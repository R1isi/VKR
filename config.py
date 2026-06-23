import torch

IMAGES_DIR = "images"

MASKS_DIR = "top_masks"           
VALID_MASKS_DIR = "top_masks"     
LABELS_DIR = "labels"             
SESSIONS_JSON = "sessions.json"

SPLIT_RATIOS = (0.80, 0.15, 0.05)
SPLIT_SEED = 42

MODEL_SAVE_PATH = "seg_model.pth"
BEST_MODEL_SAVE_PATH = "seg_model_best.pth"

IMG_SIZE = (256, 256)
HM_SIZE = (256, 256)
MASK_THRESHOLD = 0.55

SUCTION_RADIUS_PX = 8
GRASP_SEP_ERODE = 9

BASE_CHANNELS = 32

BATCH_SIZE = 8

EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
EARLY_STOP_PATIENCE = 15

AUG_ENABLED = True
AUG_ROTATION_DEG = 20.0
AUG_TRANSLATE = 0.08
AUG_SCALE = (0.85, 1.15)
AUG_BRIGHTNESS = 0.20
AUG_CONTRAST = 0.20
AUG_HFLIP = True

PCK_THRESHOLD = 0.05

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
