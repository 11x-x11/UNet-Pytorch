import argparse
import logging
import os
import numpy as np
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from train_gaps import GAPsDataset
import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

# === Adapter to make GAPsDataset compatible with existing loop ===
from torch.utils.data import Dataset

class DictDataset(Dataset):
    """Wrap a (img, mask) dataset to return {'image': img, 'mask': mask}."""
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, msk = self.base[idx]
        return {'image': img, 'mask': msk}
# ================================================================

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

# CUDA-only: must be set BEFORE any CUDA context is created
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# cuDNN settings (CUDA only). Safe on CPU; has no effect on MPS.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Prefer deterministic kernels; set warn_only=False to hard-enforce
torch.use_deterministic_algorithms(True, warn_only=True)

# (Optional) DataLoader seeding for reproducible shuffles/augs:
def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

dir_checkpoint = Path('./checkpoints/')

def compute_pos_weight(loader, device):
    """Return a scalar pos_weight for BCEWithLogitsLoss: (neg / pos)."""
    pos = 0.0
    neg = 0.0
    with torch.no_grad():
        for batch in loader:
            # Support both dict batches and (img, mask) tuples
            if isinstance(batch, dict):
                msks = batch['mask']
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                msks = batch[1]
            else:
                raise TypeError(f"Unexpected batch type: {type(batch)}")

            msks = msks.to(device=device, dtype=torch.float32)  # [B,1,H,W] in {0,1}
            pos += msks.sum().item()
            neg += (1.0 - msks).sum().item()

    return float(neg / pos) if pos > 0 else 1.0


def weighted_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, w_pos_t: torch.Tensor) -> torch.Tensor:
    """
    Paper-style class balancing: per-pixel weight map w(x) = w_pos for positive pixels, 1.0 for negative.
    logits: [B,1,H,W], target: [B,1,H,W] in {0,1}
    returns: scalar mean loss
    """
    # elementwise BCE, no reduction
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction='none')  # [B,1,H,W]

    # build weight map: positive pixels -> w_pos, negatives -> 1.0
    # w_pos_t is a scalar tensor on the right device/dtype
    w_map = torch.where(target > 0.5, w_pos_t, torch.ones_like(target))
    return (raw * w_map).mean()


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1.0,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1) Datasets / loaders
    root = Path(args.data_dir)
    train_ds = DictDataset(GAPsDataset(root, "train"))
    val_ds   = DictDataset(GAPsDataset(root, "valid"))

    n_train = len(train_ds)
    n_val   = len(val_ds)

    g = torch.Generator().manual_seed(SEED)
    # train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    # val_loader   = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_args)

    num_workers = 0  # set to 4 (or more) if you want parallel loading

    # shared args (no placeholders)
    loader_args = {
        "batch_size": int(args.batch_size),              # <- real int
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
        "generator": g,
    }

    # only add worker-specific knobs when workers > 0
    if num_workers > 0:
        loader_args.update({
            "worker_init_fn": seed_worker,
            "persistent_workers": True,   # only valid if num_workers > 0
            "prefetch_factor": 2,         # only valid if num_workers > 0
        })

    # build loaders (pass shuffle explicitly; reuse the same loader_args)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_args)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_args)

    with torch.no_grad():
        total_pos = 0
        for b in val_loader:
            m = b['mask'] if isinstance(b, dict) else b[1]
            total_pos += int(m.sum().item())
    print("Validation positive pixels:", total_pos)
    
    # 2) Compute pos_weight for class imbalance (binary)
    scan_loader = DataLoader(
        train_ds, shuffle=False, batch_size=32, num_workers=0, pin_memory=False
    )
    pw = compute_pos_weight(scan_loader, device)
    print(f"Computed pos_weight = {pw:.3f}")

    # make a scalar tensor (broadcastable) for the weight
    w_pos_t = torch.tensor(float(pw), device=device, dtype=torch.float32)

    # we'll pass a callable into evaluate() so val_loss matches train
    from functools import partial
    criterion = partial(weighted_bce_with_logits, w_pos_t=w_pos_t)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    
    use_amp = amp and device.type == 'cuda'
    try:
        from torch.amp import GradScaler
        scaler_cls = GradScaler
    except ImportError:
        # fallback for older torch
        from torch.cuda.amp import GradScaler

    grad_scaler = GradScaler(enabled=use_amp)

    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                if model.n_classes == 1:
                    true_masks = true_masks.to(device=device, dtype=torch.float32)  # binary targets as float
                else:
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                use_amp = (amp and device.type == 'cuda')
                with torch.autocast('cuda', enabled=use_amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        pred_logits = masks_pred.contiguous()           # [B,1,H,W]
                        target = true_masks.float().contiguous()   # [B,1,H,W]

                        bce  = weighted_bce_with_logits(pred_logits, target, w_pos_t)

                        dice = dice_loss(
                            torch.sigmoid(pred_logits).squeeze(1).contiguous(),  # [B,H,W]
                            target.squeeze(1).contiguous(),                      # [B,H,W]
                            multiclass=False
                        )

                        loss = bce + dice
                    else:
                         # --- multiclass path ---
                        # targets should be class indices in [0, C-1] with shape [B,H,W]
                        assert true_masks.ndim == 4 and true_masks.shape[1] == 1, \
                            f"Multiclass target must be [B,1,H,W]; got {tuple(true_masks.shape)}"
                        ce_target = true_masks.squeeze(1).long()  # -> [B,H,W]

                        # (optional) sanity: values within range
                        tmin = int(ce_target.min().item())
                        tmax = int(ce_target.max().item())
                        assert 0 <= tmin and tmax < model.n_classes, \
                            f"Target labels must be in [0,{model.n_classes-1}], got [{tmin},{tmax}]"

                        # CE loss expects [B,C,H,W] logits and [B,H,W] targets
                        ce_loss = criterion(masks_pred, ce_target)

                        # Dice (multiclass): compare [B,C,H,W] vs one-hot [B,C,H,W]
                        one_hot = F.one_hot(ce_target, num_classes=model.n_classes).permute(0, 3, 1, 2).float()
                        dice = dice_loss(F.softmax(masks_pred, dim=1), one_hot, multiclass=True)

                        loss = ce_loss + dice

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (2 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        metrics = evaluate(model, val_loader, device, amp, criterion=criterion)
                        val_score = metrics['dice']
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'val Dice': val_score,
                                'val IoU': metrics['iou'],
                                'val Precision': metrics['precision'],
                                'val Recall': metrics['recall'],
                                'val Loss': metrics['val_loss'],
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image((torch.sigmoid(masks_pred) >= 0.5).squeeze(1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = [0, 1]
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')

    parser.add_argument('--data-dir', '-d', type=Path, default=Path('data'), help='Root folder containing train/valid/test subfolders')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # pick device: MPS -> CUDA -> CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # model = model.to(memory_format=torch.channels_last)

    # logging.info(f'Network:\n'
    #              f'\t{model.n_channels} input channels\n'
    #              f'\t{model.n_classes} output channels (classes)\n'
    #              f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
