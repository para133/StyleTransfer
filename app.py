import os

import streamlit as st
from PIL import Image
import torchvision.models as models
import subprocess

from KID import compute_kid
from FaceDetect.FaceDetect import YOLO_FaceDetect
from CNNST.CNNStyleTransfer import CNNStyleTransfer
from CycleGAN.CycleGANStyleTransfer import CycleGANStyleTransfer
from UGATIT.UGATITStyleTransfer import UGATITStyleTransfer
from APISR.inference import APISRInference

# 获取当前文件路径
file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)

# 使用缓存机制加载模型
@st.cache_resource
def load_model():
    FaceDetect_model = YOLO_FaceDetect()
    CNN_model = CNNStyleTransfer(256)
    CycleGAN_model = CycleGANStyleTransfer(256, os.path.join(dir_path, "CycleGAN", "resource", "CycleGANpro.pth"))  
    UGATIT_model = UGATITStyleTransfer(256, os.path.join(dir_path, "UGATIT", "resource", "UGATIT.pt"))
    Inception_model = models.inception_v3(pretrained=True, transform_input=False).eval()  # 进行KID评价
    APISR_model = APISRInference(os.path.join(dir_path, 'APISR', 'resource', 'APISR.pth'))
    return FaceDetect_model, Inception_model, APISR_model, CNN_model, CycleGAN_model, UGATIT_model
    
FaceDetect_model, Inception_model, APISR_model, CNN_model, CycleGAN_model, UGATIT_model = load_model()

# Streamlit 界面设计
st.title("人物头像动画化")
st.sidebar.header("上传图像")

# 用户选择图像路径
uploaded_file = st.sidebar.file_uploader("选择图像文件", type=["png", "jpg", "jpeg"])

# 缓存预测图像
if 'predicted_image' not in st.session_state:
    st.session_state.predicted_image = None
if 'kid' not in st.session_state:
    st.session_state.kid = None
    
# 如果上传了图像，处理图像
if uploaded_file is not None:
    # 打开图像
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="原始图像", use_column_width=True)

    # 人脸检测
    highest_confidence_face, cropped_image_path = FaceDetect_model.FaceDetect(uploaded_file)
    
    # 可视化
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="原始图像", use_column_width=True)

    if highest_confidence_face > 0:
        cropped_image = Image.open(cropped_image_path)
        with col2:
            st.image(cropped_image, caption=f"提取出的人脸 置信度{highest_confidence_face:.2f}", use_column_width=True)
    else:
        cropped_image = None
        with col2:
            st.write("未检测到人脸!")
            
    if cropped_image is None:
        st.write("未检测到人脸!")
    else:
         # 下拉框选择模型
        model_choice = st.selectbox("选择模型", ["CNN", "CycleGANpro", "UGATIT", "APISR", "Diffstyler"]) 
        
        # 设置生成按钮
        generate_button = st.button("生成图像")
        
        # 模拟根据选择的模型做预测，并显示预测结果的图片
        # 清空之前生成的图像
        if not st.session_state.predicted_image:
            st.session_state.predicted_image = None
        if not st.session_state.kid:    
            st.session_state.kid = None 
            
        if model_choice == "CNN":
            if generate_button:                
                with st.spinner("生成预测图像中..."):
                    predicted_image_path = CNN_model.predict(cropped_image_path)
                    SR_image_path = APISR_model.predict(predicted_image_path)   
                    st.session_state.predicted_image = Image.open(SR_image_path)
                    predicted_image = st.session_state.predicted_image        
                    st.session_state.kid = compute_kid(Inception_model, cropped_image_path, predicted_image_path)
                
        elif model_choice == "CycleGANpro":
            if generate_button:
                with st.spinner("生成预测图像中..."):
                    predicted_image_path = CycleGAN_model.predict(cropped_image_path)
                    SR_image_path = APISR_model.predict(predicted_image_path)   
                    st.session_state.predicted_image = Image.open(SR_image_path)
                    predicted_image = st.session_state.predicted_image        
                    st.session_state.kid = compute_kid(Inception_model, cropped_image_path, predicted_image_path)
                                
        elif model_choice == "UGATIT":
            if generate_button:                
                with st.spinner("生成预测图像中..."):
                    predicted_image_path = UGATIT_model.predict(cropped_image_path)
                    SR_image_path = APISR_model.predict(predicted_image_path)   
                    st.session_state.predicted_image = Image.open(SR_image_path)
                    predicted_image = st.session_state.predicted_image        
                    st.session_state.kid = compute_kid(Inception_model, cropped_image_path, predicted_image_path)
                
        elif model_choice == "APISR":
            if generate_button:                 
                with st.spinner("生成预测图像中..."):
                    cropped_image = Image.open(cropped_image_path)
                    cropped_image.resize((256,256)).save(cropped_image_path)
                    predicted_image_path = APISR_model.predict(cropped_image_path)
                    st.session_state.predicted_image = Image.open(predicted_image_path)
                    predicted_image = st.session_state.predicted_image        
                    st.session_state.kid = compute_kid(Inception_model, cropped_image_path, predicted_image_path)
                
        elif model_choice == "Diffstyler":
            propmt = st.text_input(label="输入提示文本")
            if generate_button and propmt:                
                with st.spinner("生成预测图像中..."):
                    run_path = os.path.join(dir_path, 'Diffstyler')
                    diffusion_py = os.path.join(dir_path, 'Diffstyler', 'main.py')
                    predicted_image_path = os.path.join(dir_path, 'Diffstyler', 'resource', 'Diffusion_result.png')
                    propmt = "Anime style portrait, high quality" + propmt
                    command = f"python {diffusion_py} {cropped_image_path} {propmt} --output {predicted_image_path} -fs 0.8 -ws 0.2 -lc 3 --steps 50"
                    subprocess.run(command, check=True, cwd=run_path)
                    st.session_state.kid = compute_kid(Inception_model, cropped_image_path, predicted_image_path)
                    SR_image_path = APISR_model.predict(predicted_image_path)
                    st.session_state.predicted_image = Image.open(SR_image_path)

        # 显示当前生成的图像（如果有）
        if st.session_state.predicted_image:
            st.image(st.session_state.predicted_image, caption=f"生成结果   合成图像KID: {st.session_state.kid:.2f}", use_column_width=True)

        # 输入保存图像的路径
        save_path = st.text_input("输入保存图像的路径", value=os.path.join(dir_path, "SythesisedImage.png"))

        # 保存按钮
        if save_path:
            save_button = st.button("保存图像")
            if save_button:
                # 保存预测图像到指定路径
                st.session_state.predicted_image.save(save_path)
                st.success(f"图像已保存到 {save_path}")
                
        st.session_state.predicted = True
