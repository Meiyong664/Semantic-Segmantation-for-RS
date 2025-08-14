import torch
import torch.nn as nn 
from torchvision import models 
import torch.nn.functional as F

from einops import rearrange, repeat

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class ConvBNReLu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.BN = nn.BatchNorm2d(out_ch)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.ReLU(self.BN(self.conv1(x)))
        return x


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Encoder(nn.Module):
    """
    采用预训练的ResNet18， 下采样倍数为32倍， 同时返回各个下采样层的结果，用于Skip connnection
    """
    def __init__(self, pretrained = True):
        super().__init__()

        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        self.ResBlock1 = nn.Sequential(
            resnet18.conv1,             # dowansample x2
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,           # downsample x4
            resnet18.layer1             # not change the resolution
        )

        self.ResBlock2 = resnet18.layer2     # downsample x8
        self.ResBlock3 = resnet18.layer3     # downsample x16
        self.ResBlock4 = resnet18.layer4     # downsample x32

    def forward(self, x):

        x0 = self.ResBlock1(x)          # B x 64 x H/4 x W/4
        x1 = self.ResBlock2(x0)         # B x 128 x H/8 x W/8
        x2 = self.ResBlock3(x1)         # B x 256 x H/16 x W/16
        x3 = self.ResBlock4(x2)         # B x 512 x H/32 x W/32

        return x0, x1, x2, x3
    

class Upsample(nn.Module):
    """
    支持反卷积（转置卷积）或双线性插值的上采样方法，可通过参数 method 选择。
    mention：
        上采样过程最好仅改变图像大小而不改变通道数 
    """
    def __init__(self, in_ch=None, out_ch=None, method="bilinear"):
        super().__init__()
        
        self.method = method

        if method == "deconv":
            assert in_ch is not None and out_ch is not None, "deconv 需要指定 in_ch 和 out_ch"
            self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        elif method == "bilinear":
            self.deconv = None
        else:
            raise ValueError("method 仅支持 'deconv' 或 'bilinear'")

        self.ConvBNReLU = ConvBNReLu(in_ch, out_ch)

    def forward(self, x):
        
        if self.method == "deconv":
            x = self.deconv(x)
        elif self.method == "bilinear":
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        
        x = self.ConvBNReLU(x)

        return x
    

class WeightSum(nn.Module):
    """
    in this part, the features come from Encoder and Decoder has been dealed so that they have the same resolution
    we only do the WightSum
    """
    def __init__(self, in_ch, decoder_ch, eps=1e-8):
        super().__init__()
        
        self.pre_conv_change_ch = nn.Conv2d(in_ch, decoder_ch, kernel_size=1, bias=False)   # 用于对齐通道数 使两个特征相加时具有相同维度

        self.alpha = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad = True)  # 权重参数 可学习 随梯度更新而更新 
        self.eps = eps          # 防止除零
        
        self.post_conv = ConvBNReLu(decoder_ch, decoder_ch)         # 用于特征加权求和后进一步提取特征 

    def forward(self, Encoder_feature, Decoder_feature):

        alpha = nn.ReLU()(self.alpha)       # 限制加权求和时出现负数
        alpha = alpha / (torch.sum(alpha, dim=0) + self.eps)    # 归一化 分别给出两个权重 

        # 加权求和 Fused Features = alpha * Encoder_Features + (1 - alpha) * Decoder Features
        Decoder_feature = alpha[0] * self.pre_conv_change_ch(Encoder_feature) + alpha[1] * Decoder_feature 

        Decoder_feature = self.post_conv(Decoder_feature)
        return Decoder_feature
    

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    """
    Decoder part
        输入transformer层之前， 编码器已经进行了足够的缩放，        B x C x H/16 x W/16     其中 C = 512, 命 s1 = H/16, s2 = W/16 
        记窗口大小为w, 由于是在特征图之上进行窗口划分 patch size = 1
    """
    def __init__(self, 
                 window_size, 
                 dim,
                 num_heads = 8,
                 qkv_bias = False,
                 relative_pos_embed = True):
        super().__init__()

        # Local Branch
        self.Local_Branch_Conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim) )
        
        self.Local_Branch_Conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim) )
        
        # Global Branch
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim / num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=qkv_bias)       # B x 3C x s1 x s2, qkv联合矩阵
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)
        
        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embed = relative_pos_embed

        if self.relative_pos_embed:

            self.relative_pos_bias_table = nn.Parameter(
                torch.zeros(((2 * window_size - 1) * (2 * window_size -1), num_heads)) )     # 每个注意力头都需要一张映射表， 每个表有 2W -1 * 2W-1 个参数

            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)       # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]   # 2 x Wh*Ww x Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()     # W*W x W*W x 2
            relative_coords[:, :, 0] += self.window_size -1
            relative_coords[:, :, 1] += self.window_size -1
            relative_coords[:, :, 0] *= 2 * self.window_size -1
            relative_pos_index = relative_coords.sum(-1)
            self.register_buffer("relative_pos_index", relative_pos_index)

            trunc_normal_(self.relative_pos_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')      # 上 下 左 右   下填充 因为左上为原点
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')    # 右填充
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x
    
    def forward(self, x):
        B, C, H, W = x.shape

        # Local Branch
        local_conv1 = self.Local_Branch_Conv1(x)
        local_conv3 = self.Local_Branch_Conv2(x)

        local = local_conv1 + local_conv3

        # Global Branch
        x = self.pad(x, self.window_size)    # 防止无法划分window
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.window_size, ww=Wp//self.window_size, qkv=3, ws1=self.window_size, ws2=self.window_size)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embed:
            relative_position_bias = self.relative_pos_bias_table[self.relative_pos_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.window_size, ww=Wp//self.window_size, ws1=self.window_size, ws2=self.window_size)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out
    

class TransformerBlock(nn.Module):
    def __init__(self, 
                 window_size, 
                 dim,
                 MLP_hidden_dim = None,
                 num_heads = 8,
                 MLP_dropout = 0.2,
                 drop_path = 0.2):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.MLP = MLP(dim, hidden_features=MLP_hidden_dim, drop=MLP_dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attn = GlobalLocalAttention(window_size=window_size, dim=dim, num_heads=num_heads)

    def forward(self, x):
        attn = self.attn(self.norm1(x))
        x = self.drop_path(attn) + x  # 如果 drop_path=0，则直接跳过 DropPath
        MLP = self.MLP(self.norm2(x))
        x = self.drop_path(MLP) + x  # 同样跳过 DropPath
        return x
    

class FeatureRefinementHead(nn.Module):
    def __init__(self, in_ch =64, out_ch = 64):
        super().__init__()

        self.Channel_path_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch//16, in_ch, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.Spatial_pah_attn = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
            nn.Sigmoid()
        )

        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.proj = nn.Sequential(
            SeparableConvBN(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)

        channel_path = self.Channel_path_attn(x) * x
        Spatial_path = self.Spatial_pah_attn(x) * x

        x = channel_path + Spatial_path
        x = self.proj(x)
        x = x + shortcut
        x = self.act(x)

        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLu(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=False)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat
    

class Decoder(nn.Module):
    def __init__(self, 
                 num_classes,
                 window_size = 8,
                 dropout = 0.2,
                 Encoder_ch = [64, 128, 256, 512],
                 Decoder_ch = 64
                 ):

        super().__init__()
        
        self.pre_conv = nn.Sequential(nn.Conv2d(Encoder_ch[-1], Decoder_ch, 3, 1, 1),
                                      nn.BatchNorm2d(Decoder_ch))
        
        self.GLTB1 = TransformerBlock(window_size, Decoder_ch, MLP_dropout=dropout, drop_path = dropout) 
        
        self.up1 = Upsample(Decoder_ch, Decoder_ch)
        self.ws1 = WeightSum(Encoder_ch[-2], Decoder_ch)
        self.GLTB2 = TransformerBlock(window_size, Decoder_ch, MLP_dropout=dropout, drop_path = dropout)
        
        self.up2 = Upsample(Decoder_ch, Decoder_ch)
        self.ws2 = WeightSum(Encoder_ch[1], Decoder_ch)
        self.GLTB3 = TransformerBlock(window_size, Decoder_ch, MLP_dropout=dropout, drop_path = dropout)
        
        self.up3 = Upsample(Decoder_ch, Decoder_ch)
        self.ws3 = WeightSum(Encoder_ch[0], Decoder_ch)
        self.frh = FeatureRefinementHead(Decoder_ch, Decoder_ch)

        self.segmention = nn.Sequential(
            ConvBNReLu(Decoder_ch, Decoder_ch),
            nn.Dropout2d(p=dropout, inplace=True),
            nn.Conv2d(Decoder_ch, num_classes, kernel_size=1)
        )

        if self.training:
            self.aux_up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.aux_up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.aux_head = AuxHead(Decoder_ch, num_classes)

        self.init_weight()

    def forward(self, res0, res1, res2, res3, H, W):
        
        if self.training:
            out = self.pre_conv(res3)

            out = self.GLTB1(out)
            h4 = self.aux_up4(out)

            out = self.up1(out)
            out = self.ws1(res2, out)
            out = self.GLTB2(out)
            h3 = self.aux_up3(out)

            out = self.up2(out)
            out = self.ws2(res1, out)
            out = self.GLTB3(out)
            h2 = out

            out = self.up3(out)
            out = self.ws3(res0, out)
            out = self.frh(out)

            out = self.segmention(out)
            out = F.interpolate(out, (H, W),  mode='bilinear', align_corners=False)
            
            ah = h4 + h3 + h2
            ah = self.aux_head(ah, H, W)

            return out, ah
        else: 
            out = self.pre_conv(res3)

            out = self.GLTB1(out)

            out = self.up1(out)
            out = self.ws1(res2, out)
            out = self.GLTB2(out)

            out = self.up2(out)
            out = self.ws2(res1, out)
            out = self.GLTB3(out)

            out = self.up3(out)
            out = self.ws3(res0, out)
            out = self.frh(out)

            out = self.segmention(out)
            out = F.interpolate(out, (H, W),  mode='bilinear', align_corners=False)
        
            return out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class UNetFormer(nn.Module):
    def __init__(self, 
                 num_classes,
                 decode_channels=64,
                 pretrained = True,
                 window_size = 8, dropout = 0.2):
        
        super().__init__()

        self.Encoder = Encoder(pretrained=pretrained)

        self.Decoder = Decoder(num_classes=num_classes, window_size=window_size, dropout=dropout, Decoder_ch=decode_channels)

    def forward(self, x):
        h, w = x.size()[-2:]
        res0, res1, res2, res3 = self.Encoder(x)
        if self.training:
            x, ah = self.Decoder(res0, res1, res2, res3, h, w)
            return x, ah
        else:
            x = self.Decoder(res0, res1, res2, res3, h, w)
            return x
        
if __name__ == "__main__":

    x = torch.randn(2, 3, 512, 512)
    model = UNetFormer(num_classes=3)
    model.eval()
    y = model(x)
    print(y.shape)
