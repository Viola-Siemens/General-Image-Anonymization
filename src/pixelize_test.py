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

output_path = "pixelize"

def pixelize1(img: np.ndarray):
    ret = np.zeros((512, 1024, 3))
    for x in range(0, 1024):
        for y in range(0, 512):
            ret[y, x] = img[y // radius * radius, x // radius * radius]
    return ret

def start(origin_img_path, input_img_path, mask_img_path, id, time):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    all_mask = np.zeros((512, 1024, 3))
    img = np.array(Image.open(origin_img_path).resize((1024, 512), Image.LANCZOS))
    for target in palettes:
        ctail = "_%s.png"%target
        mask = np.array(Image.open(mask_img_path + ctail).resize((1024, 512), Image.LANCZOS))
        all_mask += mask
    all_mask /= 255.0
    pixelize = pixelize1(img)
    result = all_mask * pixelize + img * (1.0 - all_mask)
    Image.fromarray(result.astype(np.uint8)).save(output_path + "/%s.png"%id, "png")
    print("Finish mission %s"%id)
