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

radius = 10
sigma_2 = 10
sz = 2 * radius + 1
kernel = np.zeros((sz, sz, 3))
for i in range(sz):
    for j in range(sz):
        kernel[i, j, :] = np.exp(-((i - radius) ** 2 + (j - radius) ** 2) / (2 * sigma_2)) / (2 * np.pi * sigma_2)
kernel = kernel / kernel.sum() * 3

def blur1(img: np.ndarray):
    padded = np.zeros((512 + 2 * radius, 1024 + 2 * radius, 3))
    padded[radius:512+radius, radius:1024+radius, :] = img
    for i in range(radius):
        padded[i, radius:1024+radius, :] = img[0, :, :]
        padded[512 + radius + i, radius:1024+radius, :] = img[-1, :, :]
    for i in range(radius):
        padded[radius:512+radius, i, :] = img[:, 0, :]
        padded[radius:512+radius, 1024 + radius + i, :] = img[:, -1, :]
    padded[0:radius, 0:radius, :] = img[0, 0, :]
    padded[512+radius:512+2*radius, 0:radius, :] = img[-1, 0, :]
    padded[0:radius, 1024+radius:1024+2*radius, :] = img[0, -1, :]
    padded[512+radius:512+2*radius, 1024+radius:1024+2*radius, :] = img[-1, -1, :]
    ret = np.zeros((512, 1024, 3))
    for x in range(0, 1024):
        for y in range(0, 512):
            ret[y, x] = (kernel * padded[y:y+2*radius+1, x:x+2*radius+1]).sum(axis=(0, 1))
    return ret

def start(origin_img_path, input_img_path, mask_img_path, id, time):
    all_mask = np.zeros((512, 1024, 3))
    img = np.array(Image.open(origin_img_path).resize((1024, 512), Image.LANCZOS))
    for target in palettes:
        ctail = "_%s.png"%target
        mask = np.array(Image.open(mask_img_path + ctail).resize((1024, 512), Image.LANCZOS))
        all_mask += mask
    all_mask /= 255.0
    blur = blur1(img)
    result = all_mask * blur + img * (1.0 - all_mask)
    Image.fromarray(result.astype(np.uint8)).save("blur/%s.png"%id, "png")
    print("Finish mission %s"%id)

NAMESPACE_TEST = uuid.uuid3(uuid.NAMESPACE_DNS, "test")

for root, dirs, files in os.walk("dataset", topdown = False):
    for fn in files:
        uid = uuid.uuid3(NAMESPACE_TEST, fn)
        if fn.endswith(".png"):
            print("Deal with %s" % fn)
            filename = os.path.join(root, fn)
            start(filename, "segment/classes/%s_no" % uid, "segment/classes/%s_mask" % uid, uid, int(time.time()))