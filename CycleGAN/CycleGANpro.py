import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchsummary import summary
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, height, width = x.size()
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)  # [B, HW, C//8]
        key = self.key(x).view(batch, -1, height * width)  # [B, C//8, HW]
        
        # 注意力分数 (query x key^T)
        attention = torch.bmm(query, key)  # [B, HW, HW]
        
        # 按 sqrt 应用缩放
        d_k = query.size(-1)  # This is the size of the last dimension of query (and key)
        attention = attention / (d_k ** 0.5)  # Scale the attention scores
        
        # 应用 softmax 以获取注意力权重
        attention = F.softmax(attention, dim=-1)  
        
        value = self.value(x).view(batch, -1, height * width)  # [B, C, HW]
        
        # 将注意力权重应用于值
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(batch, channels, height, width)  # [B, C, H, W]
        
        # Return the weighted output with a residual connection
        return self.gamma * out + x


# 定义残差块
class Resnet_block(nn.Module):
    def __init__(self, in_channels):
        super(Resnet_block, self).__init__()
        block = []
        # 两层conv连接
        block += [
            nn.ReflectionPad2d(1), # 边界像素为对称轴填充(理解为桶形)
            nn.Conv2d(in_channels, in_channels, 3, 1, 0),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(True),
            nn.Dropout(0.5),  # 第一层使用Dropout    
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3, 1, 0),
            nn.InstanceNorm2d(in_channels)
            ]
        self.block = nn.Sequential(*block)
 
    def forward(self, x):
        # 残差块的输入输出相加
        out = x + self.block(x)
        return out
 
class ResAttention_block(nn.Module):
    def __init__(self, in_channels):
        super(ResAttention_block, self).__init__()
        self.res_block = Resnet_block(in_channels)
        self.attention_block = SelfAttention(in_channels)
        
    def forward(self, x):
        x = self.res_block(x)
        x = self.attention_block(x)
        return x

# 定义生成器网络
class Cycle_Gan_G(nn.Module):
    def __init__(self):
        super(Cycle_Gan_G, self).__init__()
        # 定义生成器网络结构
        DownSample_net_dic = OrderedDict()  # 网络结构字典
        # 第一次下采样
        DownSample_net_dic['down_sample1'] = nn.Sequential(
            nn.ReflectionPad2d(3),  # [3,256,256]  ->  [3,262,262]
            nn.Conv2d(3, 256, 7, stride=1, padding=0),  # [3,262,262]  ->[256,256,256]
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )
        
        # 第二次下采样
        DownSample_net_dic['down_sample2'] = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),  # [256,128,128]
            nn.InstanceNorm2d(128),
            nn.Dropout(0.3),
            nn.ReLU(True)
        )
        
        # 第三次下采样
        DownSample_net_dic['down_sample3'] = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),  # [512,64,64]
            nn.InstanceNorm2d(256),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, True)
        )

        self.net_DownSample = nn.Sequential(DownSample_net_dic)

        # 中间特征提取，增加更多的ResAttention块
        self.FeatureExtract_net = nn.Sequential(
            ResAttention_block(256),
            ResAttention_block(256),
            ResAttention_block(256),
            ResAttention_block(256)
        )
            
        UpSample_net_dic = OrderedDict()
        # 第一次上采样
        UpSample_net_dic['up_sample1'] = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),  # [256, 128, 128]
            nn.ReLU(True)
        )

        # 第二次上采样
        UpSample_net_dic['up_sample2'] = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),  # [128, 256, 256]
            nn.ReLU(True)
        )

        # 第三次上采样
        UpSample_net_dic['up_sample3'] = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, 1),
            nn.Tanh()  # 生成器的输出范围为[-1,1]，格式为3*256*256
        )
 
        self.net_UpSample = nn.Sequential(UpSample_net_dic)
        self.init_weight()
    
    # 初始化权重
    def init_weight(self):
        # 遍历所有模块
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # 使用零中心正态分布初始化权重，标准差为0.02
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    # 初始化偏置为零
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # 批归一化的权重初始化
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
 
    def forward(self, x):
        x = self.net_DownSample(x)  # [batch,512,64,64]
        x = self.FeatureExtract_net(x)  # [batch,512,64,64]
        x = self.net_UpSample(x) 
        return x

if __name__ == '__main__':
    G = Cycle_Gan_G().to('cuda')
    summary(G, (3, 256, 256))
