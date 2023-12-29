import argparse, os
import cv2
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


device = "cuda:0"

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model = model.to(device)
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/768-v-ema.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference-v.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.2,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    
    model = model.to(device)

    sampler = DDIMSampler(model)

    outpath = "./inpaint"
    os.makedirs(outpath, exist_ok=True)

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    prompt = "masterpiece, best quality, realistic, photo, street, cityscapes, traffic"
    data = [batch_size * [prompt]]

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    for root, dirs, files in os.walk("inpaint-old", topdown = False):
        for fn in files:
            if fn.endswith(".png"):
                filename = os.path.join(root, fn)
                assert os.path.isfile(filename)
                init_image = load_img(filename).to(device)
                init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

                with torch.no_grad():
                    with precision_scope("cuda"):
                        with model.ema_scope():
                            for n in trange(opt.n_iter, desc="Sampling"):
                                for prompts in tqdm(data, desc="data"):
                                    uc = None
                                    if opt.scale != 1.0:
                                        uc = model.get_learned_conditioning(batch_size * ["watermark, username, error, anime, nsfw, low quality, lowres, bad face"])
                                    if isinstance(prompts, tuple):
                                        prompts = list(prompts)
                                    c = model.get_learned_conditioning(prompts)

                                    # encode (scaled latent)
                                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
                                    # decode it
                                    samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc, )

                                    x_samples = model.decode_first_stage(samples)
                                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                                    for x_sample in x_samples:
                                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                        img = Image.fromarray(x_sample.astype(np.uint8))
                                        img = put_watermark(img, wm_encoder)
                                        img.save(os.path.join(outpath, fn))

                print("Finished \"" + os.path.join(outpath, fn) + "\".")


if __name__ == "__main__":
    main()