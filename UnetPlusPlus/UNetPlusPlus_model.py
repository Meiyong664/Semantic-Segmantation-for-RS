import torch.nn as nn
import torch 

class BasicBlock(nn.Module):
    """
    The basic conv block 

    args:
        in_channels
        out_channels 

    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # conv -> BN -> Relu
        x = self.relu(self.norm(self.conv1(x)))
        x = self.relu(self.norm(self.conv2(x)))

        return x

class UnetPlusPlus(nn.Module):

    def __init__(self, num_classes, in_channels, filters = None, deep_supervision = False):
        super().__init__()

        if filters is None:
            filters = [64, 128, 256, 512, 1024]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.depth = len(filters) - 1  # deepest index, e.g., 4 if filters len=5
        

        # 记录初始编码器部分 和原始UNet一致 可以按名取用
        # --- Encoder convs: X_{i,0} for i=0..depth ---
        self.conv_x = nn.ModuleDict()
        for i in range(self.depth + 1): # 左闭右开
            if i == 0:
                in_channels = in_channels
            else:
                in_channels = filters[i-1]
            self.conv_x[f"{i}_0"] = BasicBlock(in_channels, filters[i]) # x_{0,0}是(3, 64)
        # 下采样 最大池化
        self.Maxpool = nn.MaxPool2d(2,2)

        # 上采样部分 用于连接各中间节点 
        # --- Up-sample ops between levels (i+1 -> i). We'll reuse these ops in forward ---
        self.Ups = nn.ModuleDict()
        for i in range(self.depth):
            # 需要注意的是通道数没有发生变化 因此下面关于节点的计算通道数要特别注意
            self.Ups[f"up_{i+1}_to_{i}"] = nn.ConvTranspose2d(filters[i+1], filters[i+1], kernel_size=2, stride=2) 

        # 计算拼接节点的卷积层 即 x_{i, j} where j>=1
        # 此时 in_channels = [X_{i, 0}, X_{i, 1}, ..., X_{i, j-1}, Up(X_{i+1, j-1})], where [] refer to torch.cat(, dim = 1) 
        # 由于Up操作不改变通道数 channels_up = X_{i+1} = filter[i + 1] 而sigma[x_{i, j}, for j = 0 ... j-1] = j*filter[i]
        self.mid_point_convx = nn.ModuleDict()
        for j in range(1, self.depth+1):
            for i in range(0, self.depth+1-j):
                in_ch = j * filters[i] + filters[i + 1]
                out_ch = filters[i]
                self.mid_point_convx[f'conv_x_{i}_{j}'] = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

        # --- Final 1x1 convs for each top-level prediction X_{0,1}..X_{0,depth} ---
        if deep_supervision:
            self.final_convs = nn.ModuleList()
            for j in range(1, self.depth + 1):
                # each X_{0,j} has channels = filters[0] (we designed conv blocks to output filters[i])
                self.final_convs.append(nn.Conv2d(filters[0], num_classes, kernel_size=1))
        else:
            self.final_convs = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        
        X = {}

        # Encoder Part, the nodes x_{i, 0}, i in [0, depth]
        current = x
        for i in range(self.depth+1):
            current = self.conv_x[f"{i}_0"](current)   # 先记录当前值 因为后续要进行池化操作 若直接存储 则无法池化
            X[(i, 0)] = current                 # 输入节点存储
            if i < self.depth:
                current = self.Maxpool(current) # 高宽减半

        # Decoder part, the nodes x_{i, j}, i in [0, depth-1] j in [1, depth]
        # 必须注意的是 UNet++中 必须先计算j列的节点 第j+1列的节点才得以计算 必须按列遍历(j)
        for j in range(1, self.depth+1):
            for i in range(0, self.depth+1-j):
                up = self.Ups[f"up_{i+1}_to_{i}"](X[(i+1, j-1)])
                cats = [X[(i, k)] for k in range(0, j)] + [up]
                cat = torch.cat(cats, dim=1)
                X[(i, j)] = self.mid_point_convx[f'conv_x_{i}_{j}'](cat)

        # Output Part, Use the conbined output or only use the finnal output
        if self.deep_supervision:
            outs = []
            for j in range(1, self.depth+1):
                outs.append(self.final_convs[j-1](X[(0, j)]))
            return tuple(outs)      # (B, C, H, W) x depth
        else:
            logits = self.final_convs(X[(0, self.depth)])
            return logits
        
if __name__ == "__main__":
    # 简单自测
    model = UnetPlusPlus(num_classes=21, in_channels=3, deep_supervision=False)
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    print("out:", y.shape)