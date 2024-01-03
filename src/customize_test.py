import os
import cv2
import time
import json
import threading
import uuid
import numpy as np
from PIL import Image

palettes = [
    "road",
    "car",
    "human",
    "traffic sign",
    "building"
]

output_path = "custom"

def start(origin_img_path, input_img_path, mask_img_path, id, time):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    result = np.zeros((512, 1024, 3))
    Image.fromarray(result.astype(np.uint8)).save(output_path + "/%s.png"%id, "png")
    print("Finish mission %s"%id)
