import os
import mmcv
import mmcv_custom
import mmseg_custom
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
from PIL import Image
import threading
import uuid
import numpy as np
import json
import time
import os.path as osp

NAMESPACE_TEST = uuid.uuid3(uuid.NAMESPACE_DNS, "test")

folder = "mask"
print("Folder:", folder)

model = init_segmentor("configs/cityscapes/upernet_internimage_l_512x1024_160k_cityscapes.py", checkpoint=None, device="cuda")
checkpoint = load_checkpoint(model, "models/upernet_internimage_l_512x1024_160k_cityscapes.pth", map_location='cpu')
palette = "cityscapes"
if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    model.CLASSES = get_classes(palette)

bins = [0 for _ in range(50)]
sum_score = 0

def start(img_path, id):
    global model
    result = inference_segmentor(model, img_path)
    # show the results
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img_path, result,
                            palette=get_palette(palette),
                            show=False, opacity=1.0)[:, :, ::-1]
    origin = np.array(Image.open("segment/%s.png"%id).resize((1024, 512)))
    img = np.uint8(img)
    diff = np.power(img - origin, 2)
    temp_mask = diff.sum(axis=2) < 25
    score = temp_mask.mean()
    i = int(score * 50)
    if i >= 50:
        i = 49
    if i >= 48:
        print(img_path)
    bins[i] += 1
    return float(score)

cnt = 0
for root, dirs, files in os.walk("dataset", topdown = False):
    for fn in files:
        uid = uuid.uuid3(NAMESPACE_TEST, fn)
        if fn.endswith(".png"):
            filename = folder + "/%s.png"%uid
            score = start(filename, uid)
            sum_score += score
            cnt += 1
print("Average Precision: " + str(sum_score/cnt))
for i in range(len(bins)):
    print("[%f, %f): %d"%(i / len(bins), (i + 1) / len(bins), bins[i]))