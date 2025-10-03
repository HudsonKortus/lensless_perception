import os

HOME_PATH   =   os.path.expanduser("~")
JOB_ID      =   "run1"
MODEL_NAME  =   "mono_depth"
DATASET_RGB_PATH     =   "/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/nyu_data/nyu2_train"
DATASET_DEPTH_PATH     =   "/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/nyu_data/nyu2_train"
OUT_PATH    =   "/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/nyu_runs"
DATASET_PATH = "/home/hudson/Documents/MQP/lensless_perception/Unet_Depth_Esimation/data"
IMAGE_W = 320
IMAGE_H = 240
IMAGE_TYPE = '.png'

JOB_FOLDER  =   os.path.join(OUT_PATH, JOB_ID)
TRAINED_MDL_PATH    =   os.path.join(JOB_FOLDER, "parameters")
BATCH_SIZE          =   8
LR                  =   5e-6
LOG_BATCH_INTERVAL  =   1
LOG_WANDB = False
NUM_WORKERS  =   1
EPOCHS = 100
