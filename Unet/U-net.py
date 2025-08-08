import torch 
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        return x
    
class UNetModel(nn.Module):

    def __init__(self, in_channels, num_classes=2):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # The Encoder Part, conbined with for Encoder, passing each with the shape downsample and Channels doubled
        self.Encoder1 = BasicBlock(self.in_channels, 64)
        self.Encoder2 = BasicBlock(64, 128)
        self.Encoder3 = BasicBlock(128, 256)
        self.Encoder4 = BasicBlock(256, 512)

        # the Central Part
        self.CentralConv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # the Decoder Part, similar to the Decoder Part
        self.Upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.Decoder1 = BasicBlock(1024, 512)
        self.Upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.Decoder2 = BasicBlock(512, 256)
        self.Upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.Decoder3 = BasicBlock(256, 128)
        self.Upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.Decoder4 = BasicBlock(128, 64)
        
        self.classifier = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)

        self.Maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):   # x.shape: (B, 3, H, W)

        # Encoder Path - Downsampling
        out = self.Encoder1(x)      # (B, 64, H, W)
        copy1 = out                 # Save for skip connection
        out = self.Maxpool(out)     # (B, 64, H/2, W/2)

        out = self.Encoder2(out)    # (B, 128, H/2, W/2)
        copy2 = out                 # Save for skip connection
        out = self.Maxpool(out)     # (B, 128, H/4, W/4)

        out = self.Encoder3(out)    # (B, 256, H/4, W/4)
        copy3 = out                 # Save for skip connection
        out = self.Maxpool(out)     # (B, 256, H/8, W/8)

        out = self.Encoder4(out)    # (B, 512, H/8, W/8)
        copy4 = out                 # Save for skip connection
        out = self.Maxpool(out)     # (B, 512, H/16, W/16)

        # Bottleneck
        out = self.CentralConv(out) # (B, 1024, H/16, W/16)

        # Decoder Path - Upsampling with Skip Connections
        out = self.Upconv1(out)     # (B, 512, H/8, W/8)
        out = torch.cat((copy4, out), dim=1)    # (B, 1024, H/8, W/8) - skip connection
        out = self.Decoder1(out)    # (B, 512, H/8, W/8)

        out = self.Upconv2(out)     # (B, 256, H/4, W/4)
        out = torch.cat((copy3, out), dim=1)    # (B, 512, H/4, W/4) - skip connection
        out = self.Decoder2(out)    # (B, 256, H/4, W/4)

        out = self.Upconv3(out)     # (B, 128, H/2, W/2)
        out = torch.cat((copy2, out), dim=1)    # (B, 256, H/2, W/2) - skip connection
        out = self.Decoder3(out)    # (B, 128, H/2, W/2)

        out = self.Upconv4(out)     # (B, 64, H, W)
        out = torch.cat((copy1, out), dim=1)    # (B, 128, H, W) - skip connection
        out = self.Decoder4(out)    # (B, 64, H, W)

        # Final Classification Layer
        out = self.classifier(out)  # (B, num_classes, H, W)

        return out


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型 (3通道输入RGB图像，2类分割)
    model = UNetModel(in_channels=3, num_classes=2).to(device)
    
    # 测试输入 (batch_size=2, channels=3, height=256, width=256)
    test_input = torch.randn(2, 3, 256, 256).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"输入尺寸: {test_input.shape}")
    
    # 前向传播测试
    with torch.no_grad():
        output = model(test_input)
        print(f"输出尺寸: {output.shape}")
        
    print("✅ U-Net模型测试通过！")