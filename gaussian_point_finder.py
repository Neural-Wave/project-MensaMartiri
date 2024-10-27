# import statements
import torch
import utils
import config
import matplotlib.pyplot as plt
from torchvision.models import resnet34
from torch.utils.data import DataLoader
from utils import normalize_brightness, histogram_eq_global
from scipy.fft import fft2, ifft2, fftshift
import cv2
import os
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
size = 256
import torch.nn as nn
# ======================================================================================================================


class features_net(nn.Module):
    def __init__(self):
        super().__init__()

        #The initial convolutional layer
        self.preparation_for_resnet = nn.Sequential(
            nn.Conv2d(in_channels = 4, out_channels = 3, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1)
        ) 
        
        #The resnet part of the network
        self.resnet = resnet34()
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-3]))

        #The "upsampling" part of the network
        self.geo_net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Conv2d(in_channels=12, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
    
        
    
    #img is 3x64x64 and depth is 1x64x64
    def forward(self, img):
        x = self.preparation_for_resnet(img)
        x = self.resnet(x)
        #print("x after resnet", x.size())
        x = self.geo_net(x)
        #print("x after geonet", x.size())

        return x

def filter_image(img, radius):
    # Assume img is in BGR format
    if img is None:
        raise ValueError("The input image is invalid.")

    # Convert the image from BGR to RGB and normalize
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    # Apply histogram equalization
    img_eq = histogram_eq_global((img_rgb * 255).astype(np.uint8)) / 255.0

    # Extract the V channel (value channel)
    v_channel = img_eq[:, :, 2]

    # Apply FFT to the V channel
    img_f = np.fft.fft2(v_channel)

    # Generate circular mask
    height, width = v_channel.shape
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    mask = np.ones((height, width), dtype=np.float32)
    mask[(y - center_y) ** 2 + (x - center_x) ** 2 < radius ** 2] = 0

    # Apply mask in the frequency domain and perform inverse FFT
    img_f_filtered = img_f * np.fft.fftshift(mask)
    f_ed = np.abs(np.fft.ifft2(img_f_filtered))

    # Resize for output compatibility
    img_rgb_resized = cv2.resize(img_rgb, (256, 256))
    f_ed_resized = cv2.resize(f_ed, (256, 256))

    # Combine RGB channels with filtered layer
    combined_image = np.concatenate([img_rgb_resized, f_ed_resized[:, :, np.newaxis]], axis=-1)
    combined_tensor = torch.tensor(combined_image, dtype=torch.float32)

    return combined_tensor  # Return only the combined tensor



class finder_class:

    def __init__(self):
        self.model = features_net()
        self.model.load_state_dict(torch.load("models/gaussian_points_finder.pth"))
        self.model.eval() 
       

    def point_finder(self, image):
        start_time = time.time()
        input = filter_image(image, 20).permute(2, 0, 1).float()
        #print("input size", input.size())
        pred = self.model(input.unsqueeze(0)).squeeze(0)
        kernel_size = 25
        stride = 1
        padding = (kernel_size - 1) // 2  # Set padding to ensure output size matches input size

        # Perform max pooling
        pooled = torch.nn.functional.max_pool2d(pred, kernel_size=kernel_size, stride=stride, padding=padding)

        # Identify locations where the pooled tensor matches the original tensor
        local_maxima_mask = (pred == pooled) & (pred > 0.2)
        local_maxima_indices = torch.nonzero(local_maxima_mask)
        end_time = time.time()

        elapsed_time = end_time - start_time
        #print("elapsed time", elapsed_time)

        '''
        # Convert `input` to H x W x 3 format in uint8 for OpenCV
        images = (input.permute(1, 2, 0).cpu().numpy()[:, :, :3] * 255).astype(np.uint8)

        for index, coordinate in enumerate(local_maxima_indices):
            # Convert `coordinate` to a NumPy array with integer type for indexing
            coordinate_np = coordinate.cpu().numpy().astype(int)
            
            # Mark the local maximum on the image in red (OpenCV uses BGR format)
            cv2.circle(images, (coordinate_np[2], coordinate_np[1]), 3, (0, 0, 255), -1)

        # If needed, convert back to float32 in [0, 1] for further processing
        images = images.astype(np.float32) / 255.0
        # Convert back to float if necessary
        # Convert `images` back to a PyTorch tensor if needed
        images_tensor = torch.from_numpy(images).permute(2, 0, 1)
        # Plotting images if bindex is divisible by 1 (every iteration)
        fig, axs = plt.subplots(1, 2, figsize=(12, 10))
        axs[0].imshow(images)
        axs[1].imshow(pred.permute(1, 2, 0).detach().cpu().numpy())
        axs[1].set_title("Point Image")

        # Ensure `pointImages[0]` is also converted for display
        
        plt.tight_layout()
        plt.savefig("plots/test.jpg")
        print("PLOT SAVED")'''
        return local_maxima_indices[:, 1:]/size

if __name__ == "__main__":
    img_path = config.DATASET_PATH + utils.get_img_path("/data/local-files/?d=teamspace/s3_connections/dtp-sbm-segmentation-video-tasks-bars-stopper-alignment-images-hackaton-usi/train_set/bad_light/img_00011.jpg")
    image = cv2.imread(img_path)
    finder = finder_class()
    print("retrieved coordinates:",finder.point_finder(image))