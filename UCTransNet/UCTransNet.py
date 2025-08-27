# the model file is the UCTransNet refer to the research artical 
#   UCTransNet: Rethinking the Skip Connections in U-Net from a Channel-Wise Perspective with Transformer
#   https://ojs.aaai.org/index.php/AAAI/article/view/20144

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import copy
import math


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 3, stride = 1, pad = 1):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad)
        self.BN = nn.BatchNorm2d(out_ch)
        self.Relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.Relu(self.BN(self.conv(x)))

        return x
    
class Downsample(nn.Module):
    def __init__(self, conv_nums, in_ch, out_ch = None):
        super().__init__()

        if out_ch is None:
            out_ch = in_ch
        else:
            self.norm = nn.BatchNorm2d(out_ch)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.convs = nn.Sequential(ConvBNReLU(in_ch, out_ch))
        for _ in range(conv_nums-1):
            self.convs.append(ConvBNReLU(out_ch, out_ch))

    def forward(self, x):

        x = self.maxpool(x)
        x = self.convs(x)

        return x
    
class PatchEmbedding(nn.Module):
    """
    input: feature map output by the Encoder_i
            E: B x c x H x W

    output: 
            T: B x c x H/p x W/p -> B x patch_nums x C

    """
    def __init__(self, in_ch, img_size, patch_size, dropout = 0.2):
        super().__init__()

        img_size = _pair(img_size)
        patch_size = _pair(patch_size)

        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        
        self.down = nn.Conv2d(in_ch, in_ch, patch_size, patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_ch))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        T = self.down(x)   # B x C x H/p x W/p
        T = T.flatten(2)
        T = T.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = T + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
    
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, in_ch, head_nums):
        super().__init__()
        
        self.head_nums = head_nums
        self.scale = in_ch * 15

        assert in_ch % head_nums == 0, f"in_ch ({in_ch}) must be divisible by head_nums ({head_nums})"

        self.K_Linear_embedding = nn.ModuleList() 
        self.V_Linear_embedding = nn.ModuleList() 
        self.Q1_Linear_embedding = nn.ModuleList() 
        self.Q2_Linear_embedding = nn.ModuleList() 
        self.Q3_Linear_embedding = nn.ModuleList() 
        self.Q4_Linear_embedding = nn.ModuleList() 
        
        for _ in range(head_nums):
            K = nn.Linear(self.scale, self.scale, bias=False)
            V = nn.Linear(self.scale, self.scale, bias=False)
            Q1 = nn.Linear(in_ch, in_ch, bias=False)
            Q2 = nn.Linear(in_ch*2, in_ch*2, bias=False)
            Q3 = nn.Linear(in_ch*4, in_ch*4, bias=False)
            Q4 = nn.Linear(in_ch*8, in_ch*8, bias=False)
            self.K_Linear_embedding.append(copy.deepcopy(K))
            self.V_Linear_embedding.append(copy.deepcopy(V))
            self.Q1_Linear_embedding.append(copy.deepcopy(Q1))
            self.Q2_Linear_embedding.append(copy.deepcopy(Q2))
            self.Q3_Linear_embedding.append(copy.deepcopy(Q3))
            self.Q4_Linear_embedding.append(copy.deepcopy(Q4))

        self.psi = nn.InstanceNorm2d(head_nums)
        self.out1 = nn.Linear(in_ch, in_ch, bias=False)
        self.out2 = nn.Linear(in_ch*2, in_ch*2, bias=False)
        self.out3 = nn.Linear(in_ch*4, in_ch*4, bias=False)
        self.out4 = nn.Linear(in_ch*8, in_ch*8, bias=False)
        self.attn_dropout = nn.Dropout(0.1)
        self.proj_dropout = nn.Dropout(0.1)

    def forward(self, Q1, Q2, Q3, Q4, KV):

        B, N, C = Q1.shape  # N是patch数量 C是通道数 我们要在patch维度做多头切分 

        H = W = int(N ** 0.5)
        assert H * W == N, "num of patches must be a square number"

        K = V = KV.view(B, self.head_nums, N//self.head_nums, -1)    # B x heads x N//heads x sum(Ci)

        Q1 = Q1.view(B, self.head_nums, N//self.head_nums, -1)   # B x num_heads x head_dims x C1
        Q2 = Q2.view(B, self.head_nums, N//self.head_nums, -1)   # B x num_heads x head_dims x C2
        Q3 = Q3.view(B, self.head_nums, N//self.head_nums, -1)   # B x num_heads x head_dims x C3
        Q4 = Q4.view(B, self.head_nums, N//self.head_nums, -1)   # B x num_heads x head_dims x C4

        # Linear Embedding
        for idx, K_embed in enumerate(self.K_Linear_embedding):
            K[:, idx, :, :] = K_embed(K[:, idx, :, :])
        for idx, V_embed in enumerate(self.V_Linear_embedding):
            V[:, idx, :, :] = V_embed(V[:, idx, :, :])
        for idx, Q1_embed in enumerate(self.Q1_Linear_embedding):
            Q1[:, idx, :, :] = Q1_embed(Q1[:, idx, :, :])
        for idx, Q2_embed in enumerate(self.Q2_Linear_embedding):
            Q2[:, idx, :, :] = Q2_embed(Q2[:, idx, :, :])
        for idx, Q3_embed in enumerate(self.Q3_Linear_embedding):
            Q3[:, idx, :, :] = Q3_embed(Q3[:, idx, :, :])
        for idx, Q4_embed in enumerate(self.Q4_Linear_embedding):
            Q4[:, idx, :, :] = Q4_embed(Q4[:, idx, :, :])

        # 观察后面两个维度  
        # ((C_i x head_dims) @ (head_dims x C_sum)) @ (C_sum x head_dims) 
        # = (C_i x C_sum) @ (C_sum x head_dims) 
        # = (C_i x head_dims)
        attn1 = (Q1.transpose(-2, -1) @ K) / math.sqrt(self.scale)
        attn2 = (Q2.transpose(-2, -1) @ K) / math.sqrt(self.scale)
        attn3 = (Q3.transpose(-2, -1) @ K) / math.sqrt(self.scale)
        attn4 = (Q4.transpose(-2, -1) @ K) / math.sqrt(self.scale)

        attn1 = F.softmax(self.psi(attn1), dim=-1)  
        attn2 = F.softmax(self.psi(attn2), dim=-1)
        attn3 = F.softmax(self.psi(attn3), dim=-1)
        attn4 = F.softmax(self.psi(attn4), dim=-1)

        attn1 = self.attn_dropout(attn1)
        attn2 = self.attn_dropout(attn2)
        attn3 = self.attn_dropout(attn3)
        attn4 = self.attn_dropout(attn4)

        attn1 = attn1 @ V.transpose(-2, -1)
        attn2 = attn2 @ V.transpose(-2, -1)
        attn3 = attn3 @ V.transpose(-2, -1)
        attn4 = attn4 @ V.transpose(-2, -1)

        attn1 = attn1.permute(0, 1, 3, 2).contiguous().view(B, N, -1)
        attn2 = attn2.permute(0, 1, 3, 2).contiguous().view(B, N, -1)
        attn3 = attn3.permute(0, 1, 3, 2).contiguous().view(B, N, -1)
        attn4 = attn4.permute(0, 1, 3, 2).contiguous().view(B, N, -1)

        O1 = self.out1(attn1) 
        O2 = self.out2(attn2) 
        O3 = self.out3(attn3) 
        O4 = self.out4(attn4) 
        O1 = self.proj_dropout(O1) 
        O2 = self.proj_dropout(O2) 
        O3 = self.proj_dropout(O3) 
        O4 = self.proj_dropout(O4) 

        return O1, O2, O3, O4
    
class MLP(nn.Module):
    def __init__(self, in_ch, mid_expand_ratio = None, dropout = 0.2):
        super().__init__()

        if mid_expand_ratio is None:
            mid_expand_ratio = 1

        self.fc1 = nn.Linear(in_ch, in_ch * mid_expand_ratio)
        self.fc2 = nn.Linear(in_ch * mid_expand_ratio, in_ch)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class CCT(nn.Module):
    def __init__(self, in_ch, head_nums, mid_ch_expand_ratio = 4):
        super().__init__()

        self.in_ch = in_ch
        self.head_nums = head_nums

        self.LN1_1 = nn.LayerNorm(in_ch, eps=1e-6)
        self.LN1_2 = nn.LayerNorm(in_ch*2, eps=1e-6)
        self.LN1_3 = nn.LayerNorm(in_ch*4, eps=1e-6)
        self.LN1_4 = nn.LayerNorm(in_ch*8, eps=1e-6)
        self.LN1_KV = nn.LayerNorm(in_ch*15, eps=1e-6)

        self.LN2_1 = nn.LayerNorm(in_ch, eps=1e-6)
        self.LN2_2 = nn.LayerNorm(in_ch*2, eps=1e-6)
        self.LN2_3 = nn.LayerNorm(in_ch*4, eps=1e-6)
        self.LN2_4 = nn.LayerNorm(in_ch*8, eps=1e-6)

        self.MCA = MultiHeadCrossAttention(in_ch, head_nums)
        
        self.mlp1 = MLP(in_ch, mid_ch_expand_ratio)
        self.mlp2 = MLP(in_ch*2, mid_ch_expand_ratio)
        self.mlp3 = MLP(in_ch*4, mid_ch_expand_ratio)
        self.mlp4 = MLP(in_ch*8, mid_ch_expand_ratio)

    def forward(self, x1, x2, x3, x4):

        q1 = self.LN1_1(x1)
        q2 = self.LN1_2(x2)
        q3 = self.LN1_3(x3)
        q4 = self.LN1_4(x4)

        kv = torch.cat((x1, x2, x3, x4), dim = 2)
        kv = self.LN1_KV(kv)

        attn1, attn2, attn3, attn4 = self.MCA(q1, q2, q3, q4, kv)

        attn1 += q1
        attn2 += q2
        attn3 += q3
        attn4 += q4

        o1 = self.LN2_1(attn1)
        o2 = self.LN2_2(attn2)
        o3 = self.LN2_3(attn3)
        o4 = self.LN2_4(attn4)

        o1 = self.mlp1(o1)
        o2 = self.mlp2(o2)
        o3 = self.mlp3(o3)
        o4 = self.mlp4(o4)

        o1 += attn1
        o2 += attn2
        o3 += attn3
        o4 += attn4

        return o1, o2, o3, o4

class Reconstruct(nn.Module):
    def __init__(self, in_ch, out_ch, scale, kernel_size):
        super().__init__()

        if kernel_size == 3:
            pad = 1
        else:
            pad = 0
        
        self.ConvBNRelu = ConvBNReLU(in_ch, out_ch, kernel_size, pad = pad)
        self.scale = scale

    def forward(self, x):
        B, N, C = x.shape
        h, w = int(N**0.5), int(N**0.5)

        x = x.view(B, h, w, C).permute(0, 3, 1, 2).contiguous()
        x = nn.Upsample(scale_factor=self.scale, mode="bilinear", align_corners=True)(x)
        x = self.ConvBNRelu(x)
        return x
    
class ChannelWiseCrossTransformer(nn.Module):
    def __init__(self, in_ch, img_size, patch_size, head_nums, CCT_layer_nums, mlp_mid_expand_ratio = 4):
        super().__init__()

        self.embedding1 = PatchEmbedding(in_ch, img_size, patch_size)
        self.embedding2 = PatchEmbedding(in_ch*2, img_size//2, patch_size//2)
        self.embedding3 = PatchEmbedding(in_ch*4, img_size//4, patch_size//4)
        self.embedding4 = PatchEmbedding(in_ch*8, img_size//8, patch_size//8)

        self.CCTs = nn.ModuleList()

        for _ in range(CCT_layer_nums):
            self.CCTs.append(CCT(in_ch, head_nums, mlp_mid_expand_ratio))

        self.LN1 = nn.LayerNorm(in_ch, eps = 1e-6)
        self.LN2 = nn.LayerNorm(in_ch*2, eps = 1e-6)
        self.LN3 = nn.LayerNorm(in_ch*4, eps = 1e-6)
        self.LN4 = nn.LayerNorm(in_ch*8, eps = 1e-6)

        self.reconstruct1 = Reconstruct(in_ch, in_ch, patch_size, 3)
        self.reconstruct2 = Reconstruct(in_ch*2, in_ch*2, patch_size//2, 3)
        self.reconstruct3 = Reconstruct(in_ch*4, in_ch*4, patch_size//4, 3)
        self.reconstruct4 = Reconstruct(in_ch*8, in_ch*8, patch_size//8, 3)

    def forward(self, x1, x2, x3, x4):

        t1 = self.embedding1(x1)
        t2 = self.embedding2(x2)
        t3 = self.embedding3(x3)
        t4 = self.embedding4(x4)

        for cct in self.CCTs:
            t1, t2, t3, t4 = cct(t1, t2, t3, t4)
        o1, o2, o3, o4 = t1, t2, t3, t4

        o1 = self.LN1(o1)
        o2 = self.LN2(o2)
        o3 = self.LN3(o3)
        o4 = self.LN4(o4)

        o1 = self.reconstruct1(o1)
        o2 = self.reconstruct2(o2)
        o3 = self.reconstruct3(o3)
        o4 = self.reconstruct4(o4)

        # 这里 其实相当于保留了原始的Unet结构 x_i什么都不做 就相当于原skip
        o1 += x1
        o2 += x2
        o3 += x3
        o4 += x4

        return o1, o2, o3, o4

class CCA(nn.Module):
    def __init__(self, in_ch_x, in_ch_g):
        super().__init__()
        
        self.in_ch_x = in_ch_x  # O_i (skip connection特征)
        self.in_ch_g = in_ch_g  # D_i (decoder特征)

        # for D_i: GAP -> MLP
        self.GAP_g = nn.AdaptiveAvgPool2d((1,1))
        self.mlp_g = nn.Linear(in_ch_g, in_ch_g)  # ← 修复：输入应该是 in_ch_g

        # for O_i: upsample+convs -> GAP -> MLP  
        self.GAP_x = nn.AdaptiveAvgPool2d((1,1))
        self.mlp_x = nn.Linear(in_ch_x, in_ch_g)  # ← 修复：输出应该是 in_ch_g 以便相加

        self.relu = nn.ReLU(inplace=True)
        self.sigmod = nn.Sigmoid()

    def forward(self, g, x):
        """
        g: decoder特征 [B, in_ch_g, H, W]
        x: skip特征   [B, in_ch_x, H, W]
        """
        gap_g = self.GAP_g(g)   # [B, in_ch_g, 1, 1]
        gap_g = gap_g.permute(0, 2, 3, 1).contiguous()  # [B, 1, 1, in_ch_g]
        channel_attn_g = self.mlp_g(gap_g)  # [B, 1, 1, in_ch_g]

        gap_x = self.GAP_x(x)   # [B, in_ch_x, 1, 1]
        gap_x = gap_x.permute(0, 2, 3, 1).contiguous()  # [B, 1, 1, in_ch_x]
        channel_attn_x = self.mlp_x(gap_x)  # [B, 1, 1, in_ch_g]

        channel_attn = (channel_attn_g + channel_attn_x) / 2.0  # [B, 1, 1, in_ch_g]
        channel_attn = channel_attn.permute(0, 3, 1, 2).contiguous()  # [B, in_ch_g, 1, 1]
        channel_attn = channel_attn.expand_as(g)    # [B, in_ch_g, H, W] - 注意这里应该是g的形状
        channel_attn = self.sigmod(channel_attn)

        out = g * channel_attn  # ← 修复：应该对g应用注意力，不是x
        out = self.relu(out)

        return out

class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, convs_num):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        
        self.cca = CCA(in_ch//2, in_ch//2)  # (skip_channels, decoder_channels)

        self.nconvs = nn.Sequential(ConvBNReLU(in_ch, out_ch))

        for _ in range(convs_num - 1):
            self.nconvs.append(ConvBNReLU(out_ch, out_ch))

    def forward(self, x, skip):
        """
        x: decoder特征 [B, in_ch//2, H, W]  
        skip: 跳跃连接特征 [B, skip_ch, H*2, W*2]
        """
        up = self.up(x)  # [B, in_ch//2, H*2, W*2]
        
        # 确保skip和up的空间尺寸匹配
        if skip.size(2) != up.size(2) or skip.size(3) != up.size(3):
            skip = F.interpolate(skip, size=(up.size(2), up.size(3)), mode='bilinear', align_corners=True)
        
        cca = self.cca(up, skip)  # CCA处理
        x = torch.cat((cca, up), dim=1)  # 通道拼接
        x = self.nconvs(x)

        return x
    
class UCTransNet(nn.Module):
    def __init__(self, 
                 in_ch = 3, 
                 num_classes = 2,
                 img_size = 224,
                 num_heads = 8,
                 Encoder_channel = 64,
                 patch_size = 56,
                 CCT_nums = 12):
        
        super().__init__()

        self.num_classes = num_classes

        self.Encoder1 = ConvBNReLU(3, Encoder_channel)
        self.down1 = Downsample(2, Encoder_channel, Encoder_channel*2)
        self.down2 = Downsample(2, Encoder_channel*2, Encoder_channel*4)
        self.down3 = Downsample(2, Encoder_channel*4, Encoder_channel*8)
        self.down4 = Downsample(2, Encoder_channel*8, Encoder_channel*8)

        self.mtc = ChannelWiseCrossTransformer(Encoder_channel, img_size, patch_size, num_heads, CCT_nums)

        self.up4 = UpsampleBlock(Encoder_channel*16, Encoder_channel*4, 2)
        self.up3 = UpsampleBlock(Encoder_channel*8, Encoder_channel*2, 2)
        self.up2 = UpsampleBlock(Encoder_channel*4, Encoder_channel, 2)
        self.up1 = UpsampleBlock(Encoder_channel*2, Encoder_channel, 2)

        self.out = nn.Conv2d(Encoder_channel, num_classes, 1, 1)
        self.last_activation = nn.Sigmoid() # if using BCELoss
    def forward(self, x):

        x1 = self.Encoder1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        bottom = self.down4(x4)
        x1, x2, x3, x4 = self.mtc(x1, x2, x3, x4)
        x = self.up4(bottom, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        if self.num_classes == 1:
            logits = self.last_activation(self.out(x))
        else:
            logits = self.out(x) # if nusing BCEWithLogitsLoss or class>1

        return logits
    
if __name__ == "__main__":

    x = torch.randn(2, 3, 224, 224)

    model = UCTransNet()

    y = model(x)

    print(y.shape)