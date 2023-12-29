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

model = init_segmentor("configs/cityscapes/upernet_internimage_l_512x1024_160k_cityscapes.py", checkpoint=None, device="cuda")
checkpoint = load_checkpoint(model, "models/upernet_internimage_l_512x1024_160k_cityscapes.pth", map_location='cpu')
palette = "cityscapes"
if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    model.CLASSES = get_classes(palette)
is_running = False
result_imgs = {}

palette_color = {
    "road": (128, 64, 128),
    "car": (0, 0, 142),
    "truck": (0, 0, 70),
    "human": (220, 20, 60),
    "traffic sign": (220, 220, 0),
    "building": (70, 70, 70)
}

noise_std = 64

output_path = "segment"

def start(img_path, id, time):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(output_path + "/classes"):
        os.mkdir(output_path + "/classes")
    global is_running, model
    result = inference_segmentor(model, img_path)
    # show the results
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img_path, result,
                            palette=get_palette(palette),
                            show=False, opacity=1.0)[:, :, ::-1]
    Image.fromarray(np.uint8(img)).save(output_path + "/%s.png"%id, "png")
    origin = np.array(Image.open(img_path).convert("RGB"))
    shape = img.shape

    for cat in palette_color.keys():
        cat_img = np.zeros(shape)
        cat_img[:, :] = palette_color[cat]
        diff = np.power(img - cat_img, 2)
        temp_mask = (diff.sum(axis=2) < 40)[:, :, None]

        mask = np.repeat(temp_mask * 255, 3, axis=2)
        noise = np.random.laplace(0, noise_std, shape)
        target = temp_mask * noise + origin
        err0 = (target <= 0)
        err255 = (target >= 255)
        target = (1 - err0) * (1 - err255) * target + err255 * 255.0

        Image.fromarray(np.uint8(target)).save(output_path + "/classes/%s_no_%s.png"%(id, cat), "png")
        Image.fromarray(np.uint8(mask)).save(output_path + "/classes/%s_mask_%s.png"%(id, cat), "png")
    
    result_imgs[id] = time
    print("Finish mission %s"%id)
    is_running = False
