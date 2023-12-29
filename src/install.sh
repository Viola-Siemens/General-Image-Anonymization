pip install --upgrade pip
pip install torch==1.11.0+cu113 torchaudio==0.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch_lightning==1.4.2
pip install torchmetrics==0.6.0
pip install omegaconf
pip install opencv-python
pip install einops
pip install invisible-watermark
pip install tqdm
pip install kornia
pip install transformers
pip install open-clip-torch
pip install -U openmim
mim install mmcv
pip install mmcv-full
pip install mmsegmentation==0.30.0
pip install mmengine==0.8.3
pip install mmdet==2.28.1
cd ops_dcnv3
sh make.sh
cd ../..