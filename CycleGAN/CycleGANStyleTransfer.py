import os

import torch
from torchvision import transforms as transform
from PIL import Image
import torchvision

from CycleGAN.CycleGANpro import Cycle_Gan_G

class CycleGANStyleTransfer:
    def __init__(self, img_size, model_path):
        self.file_path = os.path.abspath(__file__)
        self.dir_path = os.path.dirname(self.file_path)
         
        self.model = Cycle_Gan_G()
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt['Ga_model'])
        self.model.eval()
        self.transform = transform.Compose([
            transform.ToTensor(),
            transform.Resize((img_size, img_size)),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def predict(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            img = self.model(img)
        result_path = os.path.join(self.dir_path, "resource", "CycleGAN_result.png")
        torchvision.utils.save_image(img, result_path)
        return result_path  
    
if __name__ == '__main__':
    file_path = os.path.abspath(__file__)   
    dir_path = os.path.dirname(file_path)
     
    model = CycleGANStyleTransfer(256, os.path.join(dir_path, "resource", "CycleGANpro.pth"))   
    result_path = model.predict(r'FaceDetect\resource\cropped.png')
    print(result_path)
    