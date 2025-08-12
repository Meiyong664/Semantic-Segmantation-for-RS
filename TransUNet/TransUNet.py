import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class UpSampleBlock(nn.Module):
    # cobined with four mian step:
    #   1. upsample the input, with the resolution double eg. H/16, W/16  -> W/8, H/8
    #   2. double the channels of input 
    #   if need the skip connection
    #   3. skip connection (cat)
    #   4. decline the channels to 1/2 
    def __init__(self, in_ch, out_ch, skip_ch = None, need_skip_connection = True):
        super().__init__()
        # in_ch: channels from previous decoder layer
        # skip_ch: channels from corresponding encoder skip
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = DoubleConv(in_ch, out_ch)
        if need_skip_connection:
            self.conv2 = DoubleConv(out_ch + skip_ch, out_ch)
        self.need_skip_connection = need_skip_connection

    def forward(self, x, skip = None):
        x = self.up(x)      # double the resolution
        x = self.conv1(x)   # decline the channels 

        # if need the skip connection:
        if self.need_skip_connection:
            x = torch.cat([x, skip], dim=1)
            x = self.conv2(x)
        
        return x


class ResNet50Encoder(nn.Module):
    """Extract multiscale features from a pretrained ResNet50.
    Returns features at stages: after conv1/maxpool (stage0), layer1, layer2, layer3, layer4
    Spatial downsample ratios (approx):  /4, /4, /8, /16, /32  depending on exact ops
    """
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        # Keep initial layers
        self.initial = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu 
        )
        # ResNet stages
        self.layer1 = nn.Sequential(
            resnet.maxpool, resnet.layer1
        )                            # usually output stride 4
        self.layer2 = resnet.layer2  # stride 8
        self.layer3 = resnet.layer3  # stride 16

    def forward(self, x):
        # x: B x 3 x H x W
        x0 = self.initial(x)      # B x 64 x H/2 x W/2
        x1 = self.layer1(x0)      # B x 256 x H/4 x W/4
        x2 = self.layer2(x1)      # B x 512 x H/8 x W/8
        x3 = self.layer3(x2)      # B x 1024 x H/16 x W/16
        return x0, x1, x2, x3


class VisionTransformerEncoder(nn.Module):
    """A thin ViT encoder that takes CNN feature map tokens as input.
    Tokenization: flatten spatial positions of CNN feature map (no extra patching)
    """
    def __init__(self, in_ch, embed_dim=768, depth=12, num_heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_ch, embed_dim)  # project per-spatial-vector to emb dim

        # positional embedding will be created dynamically for variable spatial sizes
        self.pos_embed = None

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=mlp_dim, dropout=dropout,
                                                   activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, feat):
        # feat: B x C x H x W
        B, _, H, W = feat.shape
        assert H % 16 == 0 and W % 16 == 0, "Image size must be divisivle by 16"
        N = H * W
        x = feat.flatten(2).transpose(1, 2)  # B x N x C
        x = self.proj(x)  # B x N x embed_dim

        # create pos embedding if needed
        if (self.pos_embed is None) or (self.pos_embed.shape[1] != N):
            # learnable positional embedding for current N
            # note: we register a buffer so that state dict will include it; but recreate if size changes
            self.pos_embed = nn.Parameter(torch.zeros(1, N, self.embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            # register_parameter dynamically
            self.register_parameter('pos_embed_param', self.pos_embed)

        x = x + self.pos_embed
        x = self.transformer(x)  # B x N x embed_dim
        x = x.transpose(1, 2).view(B, self.embed_dim, H, W)  # B x Embed_dim x H x W
        return x


class TransUNet(nn.Module):
        
    """
    Forward pass of the module.
    Args:
        feat (torch.Tensor): Input feature map of shape (B, C, H, W), where
            B is batch size, C is number of channels, H and W are spatial dimensions.
            The height (H) and width (W) must be divisible by 16 to allow for 16x downsampling.
    Returns:
        torch.Tensor: Output tensor of shape (B, embed_dim, H, W) after projection,
            positional embedding, and transformer processing.
    Raises:
        AssertionError: If H or W is not divisible by 16, indicating the image size
            is incompatible with the required scaling factor.
    """
    
    def __init__(self, num_classes=1, img_ch=3,
                 pretrained_backbone=True,
                 vit_embed_dim=768, vit_depth=12, vit_heads=12, vit_mlp_dim=3072,
                 decoder_channels=(512, 256, 128, 64)):
        super().__init__()
        # Encoder (ResNet50)
        self.encoder = ResNet50Encoder(pretrained=pretrained_backbone)

        # channels from ResNet50 layers
        # After inspection: layer1->256, layer2->512, layer3->1024
        enc_chs = [64, 256, 512, 1024]

        # Transformer applied to deepest feature map (x3)   
        self.vit = VisionTransformerEncoder(in_ch=enc_chs[-1], embed_dim=vit_embed_dim,
                                           depth=vit_depth, num_heads=vit_heads,
                                           mlp_dim=vit_mlp_dim)     # B x embed_dim x H/16 x W/16

        # Project transformer output to decoder starting channels
        # We'll use a conv to map vit_embed_dim -> decoder_channels[0]
        self.project = nn.Sequential(
            nn.Conv2d(vit_embed_dim, decoder_channels[0], kernel_size=1, bias=False),   # B x 512 x h/16 x W/16
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )

        # Reduce encoder skip channels to match decoder concat shapes
        # self.reduce_x3 = nn.Conv2d(enc_chs[3], decoder_channels[0], kernel_size=1)      # 1024 -> 512
        self.reduce_x2 = nn.Conv2d(enc_chs[2], decoder_channels[1], kernel_size=1)      # 512 -> 256    H/8 x W/8
        self.reduce_x1 = nn.Conv2d(enc_chs[1], decoder_channels[2], kernel_size=1)      # 256 -> 128    H/4 x W/4
        self.reduce_x0 = nn.Conv2d(enc_chs[0], decoder_channels[3], kernel_size=1)      # 64 -> 64      H/2 x W/2

        # Decoder (upsample blocks)
        # top decoder level consumes projected transformer (decoder_channels[0]) and skip from x3
        self.up1 = UpSampleBlock(decoder_channels[0], decoder_channels[1], decoder_channels[1])     # [(512 -> 256) + 256] -> 256   skip with H/8 x W/8
        self.up2 = UpSampleBlock(decoder_channels[1], decoder_channels[2], decoder_channels[2])     # [(256 -> 128) + 128] -> 128   skip with H/4 x W/4
        self.up3 = UpSampleBlock(decoder_channels[2], decoder_channels[3], decoder_channels[3])     # [(128 -> 64) + 64] -> 64   skip with H/2 x W/2
        self.up4 = UpSampleBlock(decoder_channels[3], decoder_channels[3], need_skip_connection=False)     # (64 -> 64) -> 64   no skip connection

        # final segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        x0, x1, x2, x3 = self.encoder(x)
        # Transformer on deepest feature
        t = self.vit(x3)  # B x E x H/16 x W/16 (same spatial dims as x4)
        d0 = self.project(t)  # B x 512 x H/16 x W/16

        # prepare reduced skip features
        # s3 = self.reduce_x3(x3)  # B x 512 x H/16 x W/16   转换跳连的通道数 1024 -> 512 
        s2 = self.reduce_x2(x2)  # B x 256 x H/8 x W/8
        s1 = self.reduce_x1(x1)  # B x 128 x H/4 x W/4
        s0 = self.reduce_x0(x0)  # B x 64 x H/2 x W/2

        # Decoder: progressively upsamples and fuses with skips
        u1 = self.up1(d0, s2)  # H/16
        u2 = self.up2(u1, s1)  # H/8
        u3 = self.up3(u2, s0)  # H/4
        u4 = self.up4(u3)  # H/2 or H/4 depending on initial conv

        out = self.head(u4)
        # If input spatial dims are divisible by 32, output will be smaller. Up-sample to input size.
        # out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out


if __name__ == '__main__':
    # quick smoke test
    model = TransUNet(num_classes=3, img_ch=3, pretrained_backbone=False)
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        y = model(x)
    print('output shape:', y.shape)  # expect B x num_classes x H x W