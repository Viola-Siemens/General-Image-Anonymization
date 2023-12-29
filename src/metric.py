import os

import torch
_ = torch.manual_seed(19260817)

import numpy as npy
from PIL import Image
import uuid

NAMESPACE_TEST = uuid.uuid3(uuid.NAMESPACE_DNS, "test")

from torchmetrics.image.fid import FID
from torchmetrics.image.kid import KID
from torchmetrics.image.inception import IS
from torchmetrics.image.lpip_similarity import LPIPS


device = "cuda:2"
lpips = LPIPS(net_type="vgg").to(device)
fid = FID(feature=2048).to(device)
inception = IS(feature=2048).to(device)
kid = KID(feature=2048).to(device)

folder = "mask"

if __name__ == "__main__":
    with torch.no_grad():
        for root, dirs, files in os.walk("dataset", topdown = False):
            for fn in files:
                uid = uuid.uuid3(NAMESPACE_TEST, fn)
                if fn.endswith(".png"):
                    filename1 = os.path.join(root, fn)
                    img1 = Image.open(filename1).convert("RGB").resize((1024, 512), Image.LANCZOS)
                    tensor1 = torch.tensor(npy.array(img1).transpose((2,0,1))).type(torch.uint8).to(device)[None]
                    fid.update(tensor1, real=True)
                    kid.update(tensor1, real=True)
                    filename2 = folder + "/%s.png"%uid
                    img2 = Image.open(filename2).convert("RGB")
                    tensor2 = torch.tensor(npy.array(img2).transpose((2,0,1))).type(torch.uint8).to(device)[None]
                    fid.update(tensor2, real=False)
                    kid.update(tensor2, real=False)
                    inception.update(tensor2)
                    tensor1 = torch.tensor(npy.array(img1).transpose((2,0,1)) / 127.5 - 1.0).type(torch.float).to(device)[None]
                    tensor2 = torch.tensor(npy.array(img2).transpose((2,0,1)) / 127.5 - 1.0).type(torch.float).to(device)[None]
                    lpips.update(tensor1, tensor2)

    print("FID = " + str(float(fid.compute().detach().cpu())))
    mean, std = inception.compute()
    print("IS = " + str(float(mean)))
    mean, std = kid.compute()
    print("KID = " + str(float(mean)))
    print("LPIPS = " + str(float(lpips.compute().detach())))