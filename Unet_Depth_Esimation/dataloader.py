import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import random
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
from utils import * 


# from dataaug import *
# from loadParam import *
# import pdb


class CustomImageDataset(Dataset):
    def __init__(self, rgb_img_dir, depth_img_dir, img_w, img_h, img_datatype, transform=None, target_transfrom=None):
        # load all the images and segmented into one large list
        self.DATA = []
        img_path = rgb_img_dir
        ground_truth_path = depth_img_dir

        for file in os.listdir(img_path):
            if file.endswith(img_datatype):
                img = Image.open(img_path + file)
                seg = Image.open(ground_truth_path + file)
                img = img.resize([img_w, img_h])
                ground_truth = seg.resize([img_w, img_h])
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

class NYUNativeTrain(Dataset):
    """
    root/nyu2_train/<scene>/*.jpg and matching *.png (same stem) for depth
    """
    def __init__(self, root, img_w=320, img_h=240, center_crop=True, jitter=False):
        self.root = Path(root)
        self.img_w =img_w
        self.img_h = img_h
        # self.resize_hw = resize_hw
        self.center_crop = center_crop
        self.jitter = jitter

        self.pairs = []
        scenes = [d for d in self.root.glob("*") if d.is_dir()]
        for s in tqdm(scenes, desc="Scan train scenes"):
            jpgs = sorted(s.glob("*.jpg"))
            for rgbp in jpgs:
                dep = rgbp.with_suffix(".png")
                if dep.exists():
                    self.pairs.append((rgbp, dep))

        # standard NYU crop box on 480x640
        self.crop_box = (41,45,601,471)

    def __len__(self): 
        return len(self.pairs)

    def _apply_crop(self, rgb_pil, depth_np):
        l,t,r,b = self.crop_box
        return rgb_pil.crop((l,t,r,b)), depth_np[t:b, l:r]

    def _color_jitter(self, input_img):
        color_jitter = T.ColorJitter(
            brightness=(0.5, 2.0), contrast=(0.33, 3.0),
            saturation=(0.5, 2.0), hue=(-0.35, 0.35))
        return color_jitter(input_img)

    def __getitem__(self, idx):
        rgbp, depp = self.pairs[idx]
        rgb = Image.open(rgbp).convert("RGB")
        depth = read_depth_png_auto(depp)

        if self.center_crop:
            rgb, depth = self._apply_crop(rgb, depth)

        rgb = rgb.resize((self.img_w,self.img_h), Image.BILINEAR)
        depth = np.array(Image.fromarray(depth).resize((self.img_w,self.img_h), Image.NEAREST), dtype=np.float32)
        rgb_t = to_tensor_img(rgb)
        print("depth", depth)
        print(f"min depth {depth.min()}, max depth {depth.max()}")
        depth_t = torch.from_numpy(depth).float().clamp(0.3,10.0)
        print("depth_t", depth_t)
        print(f"min depth_t {depth_t.min()}, max depth_t {depth_t.max()}")

        if self.jitter:
            rgb_t = self._color_jitter(rgb_t)

        return rgb_t, depth_t
    

class NYUNativeTest(Dataset):
    """
    root/nyu2_test/*_colors.png & *_depth.png
    """
    def __init__(self, root, img_w=320, img_h=240, center_crop=True, jitter=True):
        self.root = Path(root)
        self.img_w =img_w
        self.img_h = img_h
        # self.resize_hw = resize_hw
        self.center_crop = center_crop
        self.jitter = jitter

        self.pairs = []
        scenes = [d for d in self.root.glob("*") if d.is_dir()]
        for s in tqdm(scenes, desc="Scan train scenes"):
            jpgs = sorted(s.glob("*.jpg"))
            for rgbp in jpgs:
                dep = rgbp.with_suffix(".png")
                if dep.exists():
                    self.pairs.append((rgbp, dep))

        # standard NYU crop box on 480x640
        self.crop_box = (41,45,601,471)


    def __len__(self): return len(self.pairs)

    def _apply_crop(self, rgb_pil, depth_np):
        l,t,r,b = self.crop_box
        return rgb_pil.crop((l,t,r,b)), depth_np[t:b, l:r]

    def __getitem__(self, idx):
        rgbp, depp = self.files[idx]
        rgb = Image.open(rgbp).convert("RGB")
        depth = read_depth_png_auto(depp)
        if self.center_crop:
            rgb, depth = self._apply_crop(rgb, depth)
        H,W = self.resize_hw
        rgb = rgb.resize((W,H), Image.BILINEAR)
        depth = np.array(Image.fromarray(depth).resize((W,H), Image.NEAREST), dtype=np.float32)
        rgb_t = to_tensor_img(rgb)
        depth_t = torch.from_numpy(depth).float().clamp(0.3,10.0)
        return rgb_t, depth_t, rgbp.stem  # stem like "00000_colors"
    
    
    
    
    
    # # verify the dataloader
# if __name__ == "__main__":
#     dataset = WindowDataset(ds_path=DS_PATH)
#     dataloader = DataLoader(dataset)

#     rgb, label = dataset[0]
