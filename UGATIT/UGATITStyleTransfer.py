import os
import torch
from torchvision import transforms
from PIL import Image
import cv2
from UGATIT.networks import ResnetGenerator
from UGATIT.utils import tensor2numpy, denorm, RGB2BGR
# 直接运行用下面的导入
# from networks import ResnetGenerator
# from utils import tensor2numpy, denorm, RGB2BGR

class UGATITStyleTransfer:
    def __init__(self, img_size, model_path) -> None:
        self.file_path = os.path.abspath(__file__)
        self.dir_path = os.path.dirname(self.file_path)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        
        # 加载模型
        params = torch.load(model_path)
        self.model = ResnetGenerator(3, 3, img_size=256, n_blocks=4, light=True)
        self.model.load_state_dict(params['genA2B'])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
    def predict(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img).unsqueeze(0)
        output, _, _ = self.model(img)
        output = RGB2BGR(tensor2numpy(denorm(output[0])))
        result_path = os.path.join(self.dir_path, "resource", "UGATIT_result.png")
        cv2.imwrite(result_path, output * 255.0)
        
        return result_path
    
if __name__ == "__main__":
    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)
    style_transfer = UGATITStyleTransfer(256, os.path.join(dir_path, "resource","UGATIT.pt"))
    result_path = style_transfer.predict(os.path.join(os.path.dirname(dir_path), "resource", "cropped.png"))
    print("The result is saved in", result_path)
    
    