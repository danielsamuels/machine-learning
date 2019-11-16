import os
import pathlib
import shutil
from datetime import datetime

import numpy as np

SOURCE_IMAGE_WIDTH = 1280
SOURCE_IMAGE_HEIGHT = 720

TARGET_IMAGE_WIDTH = 300
TARGET_IMAGE_HEIGHT = 169  # math.floor(SOURCE_IMAGE_HEIGHT / (SOURCE_IMAGE_WIDTH / TARGET_IMAGE_WIDTH))

TRAINING_BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 32

BASE_PATH = os.path.join('H:\\', 'Google Drive', 'Colab Notebooks', 'bacon-images')
CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
MODEL_DIRECTORY = os.path.join(BASE_PATH, 'models')

MODEL_FILE = os.path.join(MODEL_DIRECTORY, 'my_model_1132.h5')

TRAINING_DIRECTORY = pathlib.Path(os.path.join(BASE_PATH, 'training'))
VALIDATION_DIRECTORY = pathlib.Path(os.path.join(BASE_PATH, 'validation'))

# Clear out the logs
LOG_DIRECTORY = os.path.join(CURRENT_DIRECTORY, 'logs', datetime.utcnow().strftime("%Y%m%d-%H%M%S"))
try:
    shutil.rmtree(LOG_DIRECTORY)
except FileNotFoundError:
    pass
finally:
    os.mkdir(LOG_DIRECTORY)

CHECKPOINT_DIRECTORY = os.path.join(MODEL_DIRECTORY, datetime.utcnow().strftime('%Y%m%d-%H%M%S'))

try:
    os.mkdir(CHECKPOINT_DIRECTORY)
except FileExistsError:
    pass

CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIRECTORY, 'epoch-{epoch}_val-loss-{val_loss:.2f}.h5')
FULL_MODEL_FILE = os.path.join(CHECKPOINT_DIRECTORY, 'final.h5')

# Generate the list of class names from the training directory folders.
print('Generating class names')
CLASS_NAMES = np.array([item.name for item in TRAINING_DIRECTORY.glob('*')])
