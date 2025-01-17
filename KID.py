from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

# 定义KID计算函数
def compute_kid(Inception_model, real_path, synthesis_path, kernel=2.0):
    """
    计算KID值
    :param real_path: 真实图像的路径
    :param synthesis_path: 生成图像的路径
    :param kernel: 高斯核的宽度
    :return: KID值
    """
    # 打开并提取图像特征
    real_image = Image.open(real_path).convert('RGB')
    fake_image = Image.open(synthesis_path).convert('RGB')
    
    real_feature = extract_features(Inception_model, real_image)
    fake_feature = extract_features(Inception_model, fake_image)
    
    # 计算高斯核
    kernel_real_real = rbf_kernel(real_feature.numpy(), real_feature.numpy())
    kernel_fake_fake = rbf_kernel(fake_feature.numpy(), fake_feature.numpy())
    kernel_real_fake = rbf_kernel(real_feature.numpy(), fake_feature.numpy())

    # 计算KID
    mmd = np.mean(kernel_real_real) + np.mean(kernel_fake_fake) - 2 * np.mean(kernel_real_fake)
       
    return mmd

# 计算特征提取函数
def extract_features(inception_model, img):
    """
    提取图像的InceptionV3特征
    :param inception_model: 预训练的InceptionV3模型
    :param img: 需要提取特征的图像
    :return: 提取的特征
    """    
    # 预处理图像
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img).unsqueeze(0)  # 添加batch维度
    
    # 使用InceptionV3模型提取特征
    with torch.no_grad():
        features = inception_model(img)  # 直接进行前向传播，得到全局特征
        features = features.view(features.size(0), -1)  # 展平特征
    
    return features

if __name__ == "__main__":
    # 加载预训练的InceptionV3模型
    Inception_model = models.inception_v3(pretrained=True, transform_input=False)
    Inception_model.eval()  # 设置为评估模式

    # 计算KID值
    kid = compute_kid(Inception_model, r"resource/image4.jpg", r"UGATIT/resource/UGATIT_result.png")
    print("KID: ", kid)
