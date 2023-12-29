import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import torch
import time
import json
import threading
import uuid
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from imwatermark import WatermarkEncoder
from pathlib import Path

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


torch.set_grad_enabled(False)


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler


def make_batch_sd(
        image,
        mask,
        txt,
        untxt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "untxt": num_samples * [untxt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


def inpaint(sampler, image, mask, prompt, neg_prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)

    with torch.no_grad(), torch.autocast("cuda"):
        batch = make_batch_sd(image, mask, txt=prompt, untxt = neg_prompt,
                              device=device, num_samples=num_samples)

        c = model.cond_stage_model.encode(batch["txt"])

        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck].float()
            if ck != model.masked_image_key:
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = model.get_first_stage_encoding(
                    model.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)

        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.cond_stage_model.encode(batch["untxt"])
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        shape = [model.channels, h // 8, w // 8]
        samples_cfg, intermediates = sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=1.0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
        )
        x_samples_ddim = model.decode_first_stage(samples_cfg)

        result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                             min=0.0, max=1.0)

        result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]

def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded

def predict(input_image, prompt, neg_prompt, ddim_steps, num_samples, scale, seed):
    init_image = input_image["image"].convert("RGB")
    init_mask = input_image["mask"].convert("RGB")
    image = pad_image(init_image) # resize to integer multiple of 32
    mask = pad_image(init_mask) # resize to integer multiple of 32
    width, height = image.size
    print("Inpainting...", width, height)

    result = inpaint(
        sampler=sampler,
        image=image,
        mask=mask,
        prompt=prompt,
        neg_prompt = neg_prompt,
        seed=seed,
        scale=scale,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        h=height, w=width
    )

    return result


sampler = initialize_model("configs/stable-diffusion/v2-inpainting-inference.yaml", "models/512-inpainting-ema.ckpt")
is_running = False
result_imgs = {}

palettes = [
    "road",
    "car",
    "human",
    "traffic sign",
    "building"
]

output_path = "inpaint-old"

def lerp(tp1, tp2, w1, w2 = None):
    if w2 is None:
        w2 = 1 - w1
    return (w1 * tp1[0] + w2 * tp2[0], w1 * tp1[1] + w2 * tp2[1], w1 * tp1[2] + w2 * tp2[2])

def compute(weights, imgs, pos, pixel):
    weight = np.array(weights)
    weight /= weight.sum()
    ret = lerp((0, 0, 0), pixel, 1, weight[0])
    for i in range(len(imgs)):
        ret = lerp(ret, imgs[i][pos], 1, weight[i + 1])
    return ret

def combine(origin, anonys, masks, id):
    ans = np.zeros((512, 1024, 3))
    for h in range(512):
        for w in range(1024):
            ret2 = (0, 0, 0)
            weights = [0.0625]
            for i in range(len(palettes)):
                anony = anonys[i]
                mask = masks[i]
                p = anony[h, w]
                weights.append((sum(mask[h, w]) / (3 * 255)) ** 0.25)
            ans[h, w] = compute(weights, anonys, (h, w), origin[h, w])
    Image.fromarray(np.uint8(ans)).save(output_path + "/%s.png"%id, "png")

def start(origin_img_path, input_img_path, mask_img_path, id, time):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    global is_running
    anonys = []
    masks = []
    for target in palettes:
        ctail = "_%s.png"%target
        input_image = {
            "image": Image.open(input_img_path + ctail).resize((1024, 512), Image.LANCZOS),
            "mask": Image.open(mask_img_path + ctail).resize((1024, 512), Image.LANCZOS)
        }
        masks.append(np.array(input_image["mask"]))
        prompt = "masterpiece, realistic, cityscapes, street, %s, %s"%(target, "real" if target == "human" else "no_humans")
        neg_prompt = "watermark, username, error, anime, nsfw, low quality, lowres, %s"%("no_humans" if target == "human" else "human")
        if target == "road":
            neg_prompt = neg_prompt + ", animal, painting, poster"
        elif target == "building":
            neg_prompt = neg_prompt + ", animal, painting"
        ddim_steps = 50
        num_samples = 1
        scale = 9.0
        seed = 19260817

        results = predict(input_image, prompt, neg_prompt, ddim_steps, num_samples, scale, seed)

        anonys.append(np.array(results[0]))
    combine(np.array(Image.open(origin_img_path).resize((1024, 512), Image.LANCZOS)), anonys, masks, id)
    result_imgs[id] = time
    print("Finish mission %s"%id)
    is_running = False
