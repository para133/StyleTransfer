import os
from ultralytics import YOLO
from PIL import Image

class YOLO_FaceDetect:
    def __init__(self):
        self.file_path = os.path.abspath(__file__)   
        self.dir_path = os.path.dirname(self.file_path)
        self.model = self.load_model()
            
    def load_model(self):
        # 加载模型
        model = YOLO(os.path.join(self.dir_path, "resource", 'FaceDetect.pt')).eval()
        return model

    def FaceDetect(self, image_path):  
        # 推理
        image = Image.open(image_path)
        output = self.model(image)[0]  # 获取推理结果的第一个输出
        
        # 提取检测结果
        detections = output.boxes  # 获取所有检测框
        highest_confidence = 0
        best_object = None
        
        if detections is not None:
            for box in detections.data.tolist():  # 遍历所有检测框
                x_min, y_min, x_max, y_max, confidence, class_id = box
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_object = (x_min, y_min, x_max, y_max)
        
        # 裁剪最高置信度对象, 按照正方形进行裁剪
        if best_object:
            x_min, y_min, x_max, y_max = map(int, best_object)  # 转为整数
            width = x_max - x_min
            height = y_max - y_min
            max_side = max(width, height)  # 取长宽中较大的值作为正方形的边长

            # 计算中心点
            center_x = x_min + width // 2
            center_y = y_min + height // 2

            # 计算扩大后的裁剪区域（扩大20%）
            expand_factor = 1.5  # 增加20%
            expanded_width = int(width * expand_factor)
            expanded_height = int(height * expand_factor)

            # 更新裁剪区域为扩大后的正方形
            expanded_max_side = max(expanded_width, expanded_height)  # 扩大后的正方形的边长

            # 根据扩大后的尺寸计算新的裁剪区域
            x_min_square = max(center_x - expanded_max_side // 2, 0)
            y_min_square = max(center_y - expanded_max_side // 2, 0)
            x_max_square = x_min_square + expanded_max_side
            y_max_square = y_min_square + expanded_max_side

            # 确保裁剪区域不超出图像边界
            x_max_square = min(x_max_square, image.width)
            y_max_square = min(y_max_square, image.height)
            x_min_square = max(x_min_square, 0)
            y_min_square = max(y_min_square, 0)

            # 裁剪图像
            cropped_image = image.crop((x_min_square, y_min_square, x_max_square, y_max_square))  # 裁剪区域
            target_path = os.path.join(self.dir_path, "resource", "cropped.png")
            cropped_image.save(target_path)  # 保存裁剪结果
            
        return highest_confidence, target_path
    
if __name__ == "__main__":
    # 测试
    FaceDetect_model = YOLO_FaceDetect()
    highest_confidence_face, _ = FaceDetect_model.FaceDetect(os.path.join(os.path.dirname(FaceDetect_model.dir_path), "resource", "image1.png"))
    print(f"最高置信度为{highest_confidence_face:.2f}")
                    