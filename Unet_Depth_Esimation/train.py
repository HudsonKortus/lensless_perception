# IMPORTS----------------------------------------------------------------------------
# STANDARD
import sys
import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torch.utils.data import Subset
from torchvision import transforms

import wandb

# CUSTOM
# from network import UNet
from network import Network
# from utils import *
from dataloader import CustomImageDataset, NYUNativeTest, NYUNativeTrain
from params import *
# import pdb
from utils import *
from nyu_dataloader import NYUv2


# Load the parameters
# from loadParam import *


def shouldLog(batchcount=None):
    if batchcount==None:
        return LOG_WANDB=='true'
    else:
        return batchcount%LOG_BATCH_INTERVAL == 0 and LOG_WANDB=='true'
    


#  TRAIN ----------------------------------------------------------------------------
def train(dataloader, model, loss_fn, optimizer, epochstep):
    
    # dp('train started')
    model.train()
    epochloss = 0
    for batchcount, (rgb, label) in enumerate(dataloader):
        dp(' batch', batchcount)
        
        rgb = rgb.to(device)
        # print("rbg shape: ", rgb.shape)

        # if len(label.shape) == 3:
        #     label = label.unsqueeze(1)  # [8, 240, 320] -> [8, 1, 240, 320]

        label = label.to(device)
        # print("label: ", label)
        # print(f"label min: {label.min()} label max {label.max()}")

        optimizer.zero_grad()
        
        pred = model(rgb)
        # print("pred shape: ", pred.shape)

        loss = loss_fn(pred, label)        
        loss.backward()
        optimizer.step()
        # print("loss: ", loss)
        epochloss += loss.item()

        wandb.log({
            "epochstep": epochstep,
            "batch/loss/train": loss.item(),
                })
            
        if batchcount == 0: # only for the first batch every epoch
            wandb_images = []
            for (pred_single, label_single, rgb_single) in zip(pred, label, rgb):
                
                # print(f"pred_single size: {pred_single.shape}")
                # print(f"label_single size: {label_single.shape}")
                # print(f"rgb_single size: {rgb_single.shape}")

                
                combined_image_np = CombineImages(pred_single, label_single, rgb_single)

                # Create wandb.Image object and append to the list
                wandb_images.append(wandb.Image(combined_image_np))

            wandb.log(
            {
                "images/train": wandb_images,
            })
                    
    if shouldLog():
        wandb.log({
            "epoch/loss/train": epochloss,
                    })
    

# Define the val function
def val(dataloader, model, loss_fn, epochstep):
    model.eval()
    
    epochloss = 0
    with torch.no_grad():
        for batchcount, (rgb, label) in enumerate(dataloader):
            dp(' batch', batchcount)
            
            rgb = rgb.to(device)
            label = label.to(device)
            
            pred = model(rgb)
            # print(pred)
            # print(pred.shape)
            loss = loss_fn(pred, label)       

            epochloss += loss.item()
        
            wandb.log({
                "batch/loss/": loss.item(),
                    })
            
            if batchcount == 0: # only for the first batch every epoch
                
                wandb_images = []
                for (pred_single, label_single, rgb_single) in zip(pred, label, rgb):
                    combined_image_np = CombineImages(pred_single, label_single, rgb_single)
                    wandb_images.append(wandb.Image(combined_image_np))

                wandb.log(
                {
                    "images/val": wandb_images,
                })
            
    wandb.log({
        "epoch/loss/val": epochloss,
                })

# INIT LOGGER
wandb.init(
    project=MODEL_NAME,
    name=str(JOB_ID),
    
    # track hyperparameters and run metadata
    config={
    "JOB_ID":JOB_ID,
    "learning_rate": LR,
    "batchsize": BATCH_SIZE,
    "dataset": DATASET_RGB_PATH,
    }
)
#create job directoy
if os.path.exists(JOB_FOLDER):
    shutil.rmtree(JOB_FOLDER)
    print(f"deleted previous job folder from {JOB_FOLDER}")
os.mkdir(JOB_FOLDER)
os.mkdir(TRAINED_MDL_PATH)

#get device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

# DATASET ---------------------------------------------------------------------------
# datatype = torch.float32

# Define the dataset size
# dataset = CustomImageDataset(
#     rgb_img_dir=DATASET_RGB_PATH, 
#     depth_img_dir=DATASET_DEPTH_PATH, 
#     img_w=IMAGE_W, 
#     img_h=IMAGE_H,
#     img_datatype=IMAGE_TYPE,
#     )
rgb_transform = transforms.Compose([
    transforms.Resize((240, 320)),  # Resize to your target size
    transforms.ToTensor(),          # Convert PIL to tensor and normalize to [0,1]
])

depth_transform = transforms.Compose([
    transforms.Resize((240, 320)),
])


# dataset = NYUNativeTrain(root=DATASET_RGB_PATH, img_w=IMAGE_W,img_h=IMAGE_H)
dataset = NYUv2(root=DATASET_PATH,
                train=True, 
                download=False,
                rgb_transform=rgb_transform,
                depth_transform=depth_transform)

# Split the dataset into train and validation
dataset_size = len(dataset)
train_size = int(0.9 * dataset_size) #train on 90% of dataset
test_size = dataset_size - train_size
trainset, valset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
trainLoader = torch.utils.data.DataLoader(trainset, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True, 
                                        num_workers=NUM_WORKERS)

valLoader = torch.utils.data.DataLoader(valset, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True, 
                                        num_workers=NUM_WORKERS)

# Network and optimzer --------------------------------------------------------------
# model = UNet()
model = Network(in_ch=3, out_ch=1)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.001) #use ADAMW optimizer- we can choose another one too
# optimizer = torch.optim.Adam(model.parameters(), lr=LR) #another option

# STORE ORIGINAL PARAMTERS
trainedMdlPath = TRAINED_MDL_PATH + f"test.pth"
torch.save(model.state_dict(), trainedMdlPath)

# lossFn = nn.BCEWithLogitsLoss()  #nn.CrossEntropyLoss(), but that did not seem to work much; nn.BCEWithLogitsLoss() is the one that worked best
lossFn = nn.MSELoss()


for eIndex in range(EPOCHS):
    dp(f"Epoch {eIndex+1}\n")

    print(" training:")
    train(trainLoader, model, lossFn, optimizer, eIndex)
    print(" validation:")
    val(valLoader, model, lossFn, eIndex)

    trainedMdlPath = TRAINED_MDL_PATH + f"{eIndex}.pth"
    torch.save(model.state_dict(), trainedMdlPath)