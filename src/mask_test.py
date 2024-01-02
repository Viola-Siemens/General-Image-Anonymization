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

radius = 8

def start(origin_img_path, input_img_path, mask_img_path, id, time):
    all_mask = np.zeros((512, 1024, 3))
    img = np.array(Image.open(origin_img_path).resize((1024, 512), Image.LANCZOS))
    for target in palettes:
        ctail = "_%s.png"%target
        mask = np.array(Image.open(mask_img_path + ctail).resize((1024, 512), Image.LANCZOS))
        all_mask += mask
    all_mask /= 255.0
    result = img * (1.0 - all_mask) + all_mask * 127.5
    Image.fromarray(result.astype(np.uint8)).save("mask/%s.png"%id, "png")
    print("Finish mission %s"%id)
