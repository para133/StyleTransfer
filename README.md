# 方嘉彬+2022214636+智能应用系统设计
题目序号 3  
任务名称：图像的卡通、动漫化风格生成系统

## 环境搭建
### 1. 创建conda环境，实验在python==3.11下进行
```
conda create -n fjb python==3.11
conda activate fjb
```
### 2. 实验在torch2.1.0+CUDA12.2下进行
```
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```
### 3. 按照requirements.txt安装
```
pip install -r requirements.txt
```

## 模型权重
### CNNST的风格图片、CycleGANpro、UGATIT、yolov8人脸检测、APISR模型权重可在以下链接下载
[谷歌网盘](https://drive.google.com/drive/folders/18N8M3FeXt5gO1Ftxgt6nxSWORhwyTJfm?usp=drive_link)

模型权重路径组织如下,请放至对应文件夹下    
1.style.png:  
CNNST\resource\style.png  
2.CycleGANpro.pth:  
CycleGAN\resource\CycleGANpro.pth   
3.UGATIT\resource\UGATIT.pt:  
UGATIT\resource\UGATIT.pt  
4.FaceDetect.pt:  
FaceDetect\resource\FaceDetect.pt  
5.APISR.pth:  
APISR\resource\APISR.pth  

### DiffStyler模型权重请前往官方代码库下载  
[DiffStyler](https://github.com/haha-lisa/Diffstyler) 
模型权重放至 Diffstyler\checkpoints 文件夹下

## CycleGANpro
CycleGANpro的训练代码请移步[CycleGANpro]()   
## 参考的仓库  
本项目参考了以下仓库：
- [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [U-GAT-IT](https://github.com/znxlwm/UGATIT-pytorch?tab=readme-ov-file)
- [DiffStyler](https://github.com/haha-lisa/Diffstyler)
- [APISR](https://github.com/Kiteretsu77/APISR)

## 注意事项
运行代码时请不要在含有中文的路径下运行
