LEARNING_RATE= 1e-2
WEIGHT_DECAY=1e-5
import os

PROJECT_NAME = 'GradientExtraction'
ROOT = os.getcwd()
ROOT = ROOT[:ROOT.find(PROJECT_NAME)+len(PROJECT_NAME)]

IMAGE_STOCK = os.path.join(ROOT, 'Image_stock')
RESULTS = os.path.join(ROOT, 'Results')
VERBOSE=True
DEMO=False
ERROR=1e-6

RADIAL_AXIS_DIFF_THRESOLD=0.2