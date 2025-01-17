'''
    This is file is to execute the inference for a single image or a folder input
'''
import os, sys, cv2, warnings
import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

warnings.simplefilter("default")
os.environ["PYTHONWARNINGS"] = "default"


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
# from test_utils import load_grl
from APISR.test_utils import load_grl

@torch.no_grad      # You must add these time, else it will have Out of Memory
def super_resolve_img(generator, input_path, output_path=None, weight_dtype=torch.float32, downsample_threshold=-1, crop_for_4x=True):
    ''' Super Resolve a low resolution image
    Args:
        generator (torch):              the generator class that is already loaded
        input_path (str):               the path to the input lr images
        output_path (str):              the directory to store the generated images
        weight_dtype (bool):            the weight type (float32/float16)
        downsample_threshold (int):     the threshold of height/width (short side) to downsample the input
        crop_for_4x (bool):             whether we crop the lr images to match 4x scale (needed for some situation)
    '''
    print("Processing image {}".format(input_path))
    
    # Read the image and do preprocess
    img_lr = cv2.imread(input_path)
    h, w, c = img_lr.shape


    # Downsample if needed
    short_side = min(h, w)
    if downsample_threshold != -1 and short_side > downsample_threshold:
        resize_ratio = short_side / downsample_threshold
        img_lr = cv2.resize(img_lr, (int(w/resize_ratio), int(h/resize_ratio)), interpolation = cv2.INTER_LINEAR)


    # Crop if needed
    if crop_for_4x:
        h, w, _ = img_lr.shape
        if h % 4 != 0:
            img_lr = img_lr[:4*(h//4),:,:]
        if w % 4 != 0:
            img_lr = img_lr[:,:4*(w//4),:]


    # Transform to tensor
    img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
    img_lr = ToTensor()(img_lr).unsqueeze(0).cuda()     # Use tensor format
    img_lr = img_lr.to(dtype=weight_dtype)
    
    
    # Model inference
    # print("lr shape is ", img_lr.shape)
    super_resolved_img = generator(img_lr)

    # Store the generated result
    with torch.cuda.amp.autocast():
        if output_path is not None:
            save_image(super_resolved_img, output_path)

    # Empty the cache everytime you finish processing one image
    torch.cuda.empty_cache() 
    
    return super_resolved_img

class APISRInference:
    ''' The APISR Inference class
    Args:
        weight_path (str):              the path to the weight
        scale (int):                    the scale factor
        store_dir (str):                the directory to store the generated images
        model (str):                    the model type
        downsample_threshold (int):     the threshold of height/width (short side) to downsample the input
        float16_inference (bool):       whether we use float16 inference
    '''
    def __init__(self, weight_path, scale=4, model='GRL', downsample_threshold=-1, float16_inference=False):
        self.weight_path = weight_path
        self.scale = scale
        self.store_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resource')
        self.model = model
        self.downsample_threshold = downsample_threshold
        self.float16_inference = float16_inference
        
        # Define the weight type
        if self.float16_inference:
            torch.backends.cudnn.benchmark = True
            self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32
    
        self.generator = load_grl(self.weight_path, scale=self.scale)  # GRL for Real-World SR only support 4x upscaling
        self.generator = self.generator.to(dtype=self.weight_dtype)
        self.generator.eval()
    def predict(self, input_path):
        ''' Predict the super resolution image
        Args:
            input_path (str):               the path to the input lr images
        Returns:
            output_path (str):              the directory to store the generated images
        '''
        output_path = os.path.join(self.store_dir, 'APISR_result.png')      # Output fixed to be png
        super_resolve_img(self.generator, input_path, output_path, self.weight_dtype, self.downsample_threshold, crop_for_4x=True)
        
        return output_path

if __name__ == "__main__":
    
    # Fundamental setting
    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)
    
    # input_dir = os.path.join(dir_path, 'resource', 'test.png')
    # scale = 4
    # store_dir = os.path.join(dir_path, 'resource')
    # model = 'GRL'
    # weight_path = os.path.join(dir_path, 'resource', '4x_APISR_GRL_GAN_generator.pth')
    # downsample_threshold = -1
    # float16_inference = False
    
    APISR_model = APISRInference(os.path.join(dir_path, 'resource', '4x_APISR_GRL_GAN_generator.pth'))
    output_path = APISR_model.predict(os.path.join(dir_path, 'resource', 'test.png'))
    print(f"Output path is {output_path}")
        
        