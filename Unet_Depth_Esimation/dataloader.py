import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import random
from PIL import Image
import os
# from dataaug import *
# from loadParam import *
# import pdb


class CustomImageDataset(Dataset):
    def __init__(self, rgb_img_dir, depth_img_dir, img_w, image_h, img_datatype, transform=None, target_transfrom=None):
        # load all the images and segmented into one large list
        self.DATA = []
        img_path = rgb_img_dir
        ground_truth_path = depth_img_dir

        for file in os.listdir(img_path):
            if file.endswith(img_datatype):
                img = Image.open(img_path + file)
                seg = Image.open(ground_truth_path + file)
                img = img.resize([img_w, image_h])
                ground_truth = seg.resize([img_w, image_h])
                self.DATA.append([img, ground_truth])
        print("Dataset initialized")

    def __len__(self):
        N = len(self.DATA)
        return N

    # we do a little trick: to make the 50 000 images, if the image requested number is 47589, we will return image 4758 and a random augmentation (1-9, 0 will return the original)
    def __getitem__(self, idx):
        # idx is from 0 to N-1                
        # Open the RGB image and ground truth label
        # convert them to tensors
        # apply any transform (blur, noise...)

        image_idx = idx // 10
        rgb, label = self.DATA[image_idx]
        # add the random noises if the image is not the original
        if idx % 10 != 0:
            rgb = self.guass_noise(rgb)
            rgb = self.blur(rgb)
            rgb = self.color_jit(rgb)
        
        # get rid of alpha in the png
        rgb = rgb.convert("RGB")
        rgb = T.ToTensor()(rgb)
        label = label.convert("L")
        label = T.ToTensor()(label)     

        return rgb, label
    
##########################################################################3
###### Data Augmentaion, feel free to add more####################################
#####################################################

    def guass_noise(self, input_img):
        inputs = T.ToTensor()(input_img)
        noise = inputs + torch.rand_like(inputs) * random.uniform(0, 1.0)
        noise = torch.clip(noise, 0, 1.)
        output_image = T.ToPILImage()
        image = output_image(noise)
        return image


    def blur(self, input_img):
        blur_transfrom = T.GaussianBlur(
            kernel_size=random.choice([3, 5, 7, 9, 11]), sigma=(0.1, 1.5))
        return blur_transfrom(input_img)


    def color_jit(self, input_img):
        color_jitter = T.ColorJitter(
            brightness=(0.5, 2.0), contrast=(0.33, 3.0),
            saturation=(0.5, 2.0), hue=(-0.35, 0.35))
        return color_jitter(input_img)


# # verify the dataloader
# if __name__ == "__main__":
#     dataset = WindowDataset(ds_path=DS_PATH)
#     dataloader = DataLoader(dataset)

#     rgb, label = dataset[0]
