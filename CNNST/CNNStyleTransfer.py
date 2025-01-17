import os

import torch
import torchvision
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt

# 计算内容损失
def calc_contentloss(Y_hat, Y):
    # 计算所有通道对应矩阵的差的平方和，再除以所有元素个数
    return torch.square(Y_hat - Y.detach()).mean()

# 计算Gram矩阵
# 输入是vgg某层输出的特征图，尺寸为（c,h,w）
def calc_gram(x):
    c = x.shape[1]    # c是输出的风格特征图的通道数
    hw = x.shape[2] * x.shape[3]    
    x = x.reshape((c, hw))    # 将（c,h,w）变换为（c,h*w）
    return torch.matmul(x, x.T) / (c * hw)    

# 计算风格损失
def calc_styleloss(Y_hat, gram_Y):
    return torch.square(calc_gram(Y_hat) - gram_Y.detach()).mean()

# 计算全变分损失
def calc_tvloss(Y_hat):
    # [:, :, 1:, :]表示取各通道图像矩阵的第1行至最后一行(起始行为0行)
    # [:, :, :-1, :]表示取各通道图像矩阵的第0行至倒数第二行
    # 矩阵相减取绝对值再在各通道上取平均(所有元素加起来除以总元素数)
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

# 计算总的损失函数值
def compute_loss(X, content_Y, content_Y_hat, style_Y_gram, style_Y_hat):
    content_weight, style_weight, tv_weight = 1, 10000, 10
    # 分别计算内容损失、风格损失和全变分损失
    # 对1对y,y_hat求内容损失，乘以权重后添加到列表中
    content_l = [calc_contentloss(Y_hat, Y) * content_weight for Y_hat, Y in zip(content_Y_hat, content_Y)]
    # 对5对y,y_hat分别求风格损失，乘以权重后添加到列表中
    style_l = [calc_styleloss(Y_hat, Y) * style_weight for Y_hat, Y in zip(style_Y_hat, style_Y_gram)]
    # 求总变差损失，乘以权重
    tv_l = calc_tvloss(X) * tv_weight
    # 对所有损失求和(5个风格损失，1个内容损失，1个总变差损失)
    l = sum(style_l + content_l + [tv_l])
    return content_l, style_l, tv_l, l

class CNNStyleTransfer:
    def __init__(self, img_size = 256):
        self.file_path = os.path.abspath(__file__)
        self.dir_path = os.path.dirname(self.file_path)
        self.img_size = img_size    
        # ImageNet先验归一化参数
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406])
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225])
        self.content_idx, self.style_idx, self.net = self.get_model()
        
    def preprocess(self, img, image_shape):
        # 图像预处理 
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_shape),
            # ToTensor()将其图像由(h,w,c)转置为(c,h,w)，再把像素值从[0,255]变换到[0,1]
            torchvision.transforms.ToTensor(),
            # 标准化
            torchvision.transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)])
        return transforms(img).unsqueeze(0) # unsqueeze将图像升至4维，增加批次数=1，便于后续图像处理可以更好地进行批操作

    # 在训练过程可视化中，通常需要反归一化，以显示能用人眼看得懂的正常的图
    def postprocess(self, img):
        # 将图像从训练的GPU环境移至CPU并转为3维(c,h,w)
        img = img[0].to(self.rgb_std.device)
        # 先将(c,h,w)转化为(h,w,c)实施反归一化，并将输出范围限制到[0,1]
        img = torch.clamp(img.permute(1, 2, 0) * self.rgb_std + self.rgb_mean, 0, 1) 
        # 再将(h,w,c)转化为(c,h,w)，ToPILImage()将Tensor的每个元素乘以255；将数据由Tensor转化成Uint8
        return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

    # 构建网络
    def get_model(self):
        # 生成预训练的VGG-19模型
        pretrained_net = torchvision.models.vgg19(pretrained=True)
        # 各卷积块中的第一个卷积层的索引分别是Sequential中的0, 5, 10, 19, 28
        style_idx = [0, 5, 10, 19, 28]
        # 第4个卷积块中最后一个卷积层的索引是Sequential中的25
        content_idx = [25]

        # 提取所需的层,简化网络结构
        layers = []
        # max(content_layers + style_layers)求用到的最深层的索引
        for i in range(max(style_idx + content_idx) + 1):
            # 逐层加入到所需的layers列表中
            layers.append(pretrained_net.features[i])
        # 将layers逐元素加入到Sequential中
        net = nn.Sequential(*layers)
        net = net.cuda()
        
        return content_idx, style_idx, net


    # 提取合成图像的内容特征和风格特征
    def extract_features(self, x, content_idx, style_idx):
        content_features = []
        style_features = []
        for i in range(len(self.net)):
            # 计算当前层输出
            temp_layer = self.net[i]
            x = temp_layer(x)
            # 若当前层索引在风格索引列表中
            if i in style_idx:
                style_features.append(x)
            # 若当前层索引在内容索引列表中
            if i in content_idx:
                content_features.append(x)
        return content_features, style_features

    # 提取内容图像的内容特征
    def get_content_features(self, content_img, image_shape):
        # 对content_img先进行预处理并移至gpu，便于直接输入网络
        content_x = self.preprocess(content_img, image_shape).cuda()
        # 提取内容图像的内容特征
        content_features_x, _ = self.extract_features(content_x, self.content_idx, self.style_idx)
        # 输出内容图像的预处理结果后面作为合成图像的初始化图像
        return content_x, content_features_x

    # 提取风格图像的风格特征
    def get_style_features(self, style_img, image_shape):
        # 对style_img先进行预处理并移至gpu，便于直接输入网络
        style_x = self.preprocess(style_img, image_shape).cuda()
        # 提取风格图像的风格特征
        _, style_features_x = self.extract_features(style_x, self.content_idx, self.style_idx)
        return style_x, style_features_x

    def train(self, X, content_Y, style_Y, lr = 0.3, num_epochs = 500, lr_decay_epoch = 50):
        # X是初始化的合成图像，style_Y_gram是原始风格图像的格拉姆矩阵列表
        X, style_Y_gram, trainer = get_inits(X, lr, style_Y)
        
        # 用于存储每个epoch的损失
        content_losses = []
        style_losses = []
        tv_losses = []
        total_losses = []
        
        # 定义学习率下降调节器
        scheduler = optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
        for epoch in range(num_epochs):
            trainer.zero_grad()
            # Y_hat是用合成图像计算出的特征图
            content_Y_hat, style_Y_hat = self.extract_features(X, self.content_idx, self.style_idx)
            content_l, style_l, tv_l, l = compute_loss(X, content_Y, content_Y_hat, style_Y_gram, style_Y_hat)
            
            content_losses.append(sum(content_l).item())
            style_losses.append(sum(style_l).item())
            tv_losses.append(tv_l.item())
            total_losses.append(l.item())
            
            # 反向传播误差
            l.backward()
            # 更新一次合成图像的像素参数
            trainer.step()
            # 更新学习率超参数
            scheduler.step()
            # 每5个epoch记录一次loss信息
            if (epoch + 1) % 5 == 0:
                print('迭代次数：{}    内容损失：{:.9f}    风格损失：{:.9f}    总变差损失：{:.9f}' .format(epoch+1, sum(content_l).item(), sum(style_l).item(), tv_l.item()))
                
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_epochs), content_losses, label='Content Loss', color='blue')
        plt.plot(range(num_epochs), style_losses, label='Style Loss', color='red')
        plt.plot(range(num_epochs), tv_losses, label='TV Loss', color='green')
        plt.plot(range(num_epochs), total_losses, label='Total Loss', color='black', linestyle='--')
        
        plt.title('Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(os.path.join(self.dir_path, "resource", 'loss.png'))
            
        # 训练结束后返回合成图像
        return X
    
    def predict(self, 
                content_img_path, 
                style_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resource", "style.png")):
        # 读取图像
        content_img = Image.open(content_img_path)
        style_img = Image.open(style_img_path)
        # 给出合成图像的尺寸
        image_shape = (self.img_size, self.img_size)
        
        # 计算内容图像的预处理结果（因为我们将内容图像作为合成图像的初始化图像作为网络的初始输入）和抽取到的内容特征
        X, content_features_Y = self.get_content_features(content_img, image_shape)
        # 计算风格图像抽取到的风格特征
        _, style_features_Y = self.get_style_features(style_img, image_shape)
        
        # 开始训练
        output = self.train(X, content_features_Y, style_features_Y)
        # 调用后处理函数处理最终的合成图像，将其转换为正常格式的可视化图像
        output = self.postprocess(output)
        result_path = os.path.join(self.dir_path, "resource", "CNNStyleTransfer_result.png")
        output.save(result_path)
        
        return result_path

class SynthesizedImage(nn.Module):
    def __init__(self, img_shape):
        super(SynthesizedImage, self).__init__()
        # 随机初始化合成图像的weight参数，并将其转换为可训练参数
        self.weight = nn.Parameter(torch.rand(*img_shape))
    # forward方法返回训练参数矩阵
    def forward(self):
        return self.weight

def get_inits(X, lr, style_Y):
    # X是内容图像的预处理结果
    gen_img = SynthesizedImage(X.shape).cuda()
    # 将初始化的weight参数改为已有的图像X的参数（即像素）
    gen_img.weight.data.copy_(X.data)
    # 定义优化器
    trainer = optim.Adam(gen_img.parameters(), lr=lr)
    # 对各风格特征图计算其格拉姆矩阵，并依次存于列表中
    style_Y_gram = [calc_gram(Y) for Y in style_Y]
    return gen_img(), style_Y_gram, trainer

if __name__ == '__main__':
    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)
    
    CNNStyleTransfer_model = CNNStyleTransfer(256)
    content_img_path = os.path.join(os.path.dirname(dir_path), "resource", "cropped.png")
    result = CNNStyleTransfer_model.predict(content_img_path)
    print(result)