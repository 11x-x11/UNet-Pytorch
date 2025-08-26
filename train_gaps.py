
import os, random, math, argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt

# --------------------- Utils ---------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# --------------------- Dataset ---------------------
class GAPsDataset(Dataset):
    """
    Expects root split like:
      root/
        train/
          imgs/*.jpg|png
          masks/*.png
        valid/
          imgs/*.jpg|png
          masks/*.png
        test/
          imgs/*.jpg|png
          masks/*.png
    Masks are binary (0 background, 255/1 crack)
    """
    def __init__(self, root: Path, split: str):
        self.img_dir = Path(root) / split / 'imgs'
        self.msk_dir = Path(root) / split / 'masks'
        self.items = sorted([p.stem for p in self.img_dir.glob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg'}])
        assert len(self.items) > 0, f"No images found in {self.img_dir}"

        self.img_tf = T.Compose([
            T.Resize((400, 400), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # [0,1]
        ])
        self.msk_tf = T.Compose([
            T.Resize((400, 400), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),                               # -> float in [0,1]
            T.Lambda(lambda t: (t > 0).float()),        # hard-binarize just in case
        ])
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        stem = self.items[idx]
        # image can be jpg/png, mask commonly png
        img_path = None
        for ext in ('.jpg','.png','.jpeg'):
            p = self.img_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            raise FileNotFoundError(f"Image not found for {stem} in {self.img_dir}")

        # mask may have png extension
        msk_path = None
        for ext in ('.png','.jpg','.jpeg'):
            p = self.msk_dir / f"{stem}{ext}"
            if p.exists():
                msk_path = p
                break
        if msk_path is None:
            raise FileNotFoundError(f"Mask not found for {stem} in {self.msk_dir}")

        img = Image.open(img_path).convert('RGB')
        msk = Image.open(msk_path).convert('L')

        img = self.img_tf(img)  # [3,400,400], 0..1
        msk = self.msk_tf(msk)      # [1,H,W], uint8
        msk = (msk > 0).float()     # any nonzero value counts as foreground
        return img, msk



if __name__ == "__main__":
    # point this to your dataset root and split
    root = Path("tinydata")  
    dataset = GAPsDataset(root, "train")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # just grab one sample
    img, msk = next(iter(loader))

    print("Image:", img.shape, img.min().item(), img.max().item(), img.dtype)
    print("Mask :", msk.shape, msk.min().item(), msk.max().item(), msk.dtype)
    print("Unique values in mask:", msk.unique())

    # quick check foreground count
    print("Foreground pixels:", int(msk.sum().item()))
    print("Total pixels:", msk.numel())