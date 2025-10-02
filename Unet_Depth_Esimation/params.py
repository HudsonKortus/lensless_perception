import os

HOME_PATH   =   os.path.expanduser("~")
JOB_ID      =   "run2"
MODEL_NAME  =   "windowseg"
DATASET_RGB_PATH     =   "/mnt/c/Users/BAMBUSGS/Desktop/Skola/WPI/Academics/Senior/A-term/git_shared/RBE474X/rbe474x_p2/part1/images/"
DATASET_DEPTH_PATH     =   "/mnt/c/Users/BAMBUSGS/Desktop/Skola/WPI/Academics/Senior/A-term/git_shared/RBE474X/rbe474x_p2/part1/images/"


OUT_PATH    =   "/mnt/c/Users/BAMBUSGS/Desktop/Skola/WPI/Academics/Senior/A-term/git_shared/RBE474X/rbe474x_p2/part2/output/"
IMAGE_W = 320
IMAGE_H = 240
IMAGE_TYPE = '.png'

JOB_FOLDER  =   os.path.join(OUT_PATH, JOB_ID)
TRAINED_MDL_PATH    =   os.path.join(JOB_FOLDER, "parameters")
BATCH_SIZE          =   8
LR                  =   5e-6
LOG_BATCH_INTERVAL  =   1
LOG_WANDB = True
NUM_WORKERS  =   1
EPOCHS = 100