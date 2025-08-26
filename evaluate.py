import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coeff, dice_loss

def _binary_stats(pred, true):
    # pred, true: float tensors in {0,1}, shape [B,1,H,W]
    tp = (pred * true).sum(dim=(2,3))
    fp = (pred * (1 - true)).sum(dim=(2,3))
    fn = ((1 - pred) * true).sum(dim=(2,3))
    union = (pred + true - pred*true).sum(dim=(2,3))
    # avoid /0
    eps = 1e-8
    precision = (tp / (tp + fp + eps)).mean()
    recall    = (tp / (tp + fn + eps)).mean()
    iou       = (tp / (union + eps)).mean()
    return precision, recall, iou

def _multiclass_stats(pred_oh, true_oh):
    # pred_oh, true_oh: one-hot [B,C,H,W], C includes background at 0
    eps = 1e-8
    C = pred_oh.shape[1]
    classes = range(1, C)  # ignore background 0
    precisions, recalls, ious = [], [], []
    for c in classes:
        p = pred_oh[:, c:c+1]
        t = true_oh[:, c:c+1]
        tp = (p * t).sum(dim=(2,3))
        fp = (p * (1 - t)).sum(dim=(2,3))
        fn = ((1 - p) * t).sum(dim=(2,3))
        union = (p + t - p*t).sum(dim=(2,3))
        precisions.append((tp / (tp + fp + eps)).mean())
        recalls.append((tp / (tp + fn + eps)).mean())
        ious.append((tp / (union + eps)).mean())
    precision = torch.stack(precisions).mean() if precisions else torch.tensor(0.0, device=pred_oh.device)
    recall    = torch.stack(recalls).mean()    if recalls    else torch.tensor(0.0, device=pred_oh.device)
    iou       = torch.stack(ious).mean()       if ious       else torch.tensor(0.0, device=pred_oh.device)
    return precision, recall, iou

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, criterion=None):
    net.eval()
    num_val_batches = len(dataloader)

    dice_sum = 0.0
    val_loss_sum = 0.0
    prec_sum = 0.0
    rec_sum  = 0.0
    iou_sum  = 0.0

    # default criterion consistent with net.n_classes
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss() if net.n_classes == 1 else nn.CrossEntropyLoss()

    autocast_device = device.type if device.type != 'mps' else 'cpu'
    with torch.autocast(autocast_device, enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            logits = net(image)

            if net.n_classes == 1:
                # Binary: threshold at τ = 0.5
                # true in {0,1} -> float for metrics that expect float
                mask_t = mask_true.float().unsqueeze(1) if mask_true.ndim == 3 else mask_true.float()

                # loss = BCEWithLogits + Dice
                bce  = criterion(logits, mask_t)
                dcmp = dice_loss(torch.sigmoid(logits).squeeze(1), mask_t.squeeze(1), multiclass=False)
                loss = bce + dcmp

                probs  = torch.sigmoid(logits)
                pred   = (probs > 0.5).float()

                # Dice
                dice_val = float(dice_coeff(pred, mask_t, reduce_batch_first=False))

                # P/R/IoU
                p, r, j = _binary_stats(pred, mask_t)
                prec_sum += float(p)
                rec_sum  += float(r)
                iou_sum  += float(j)

            else:
                # Multiclass: argmax + one-hot, ignore background for metrics
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes

                # multiclass targets [B,H,W] long
                mask_long = mask_true.to(device=device, dtype=torch.long)

                # loss = CE + Dice(probs vs one-hot)
                ce   = criterion(logits, mask_long)
                probs = F.softmax(logits, dim=1)
                true_oh = F.one_hot(mask_long, net.n_classes).permute(0,3,1,2).float()
                dcmp = dice_loss(probs, true_oh, multiclass=True)
                loss = ce + dcmp

                
                pred_oh = F.one_hot(logits.argmax(dim=1), net.n_classes).permute(0,3,1,2).float()

                # Dice (ignore background)
                dice_val = float(multiclass_dice_coeff(pred_oh[:,1:], true_oh[:,1:], reduce_batch_first=False))

                # P/R/IoU (macro over classes 1..C-1)
                p, r, j = _multiclass_stats(pred_oh, true_oh)
                prec_sum += float(p)
                rec_sum  += float(r)
                iou_sum  += float(j)
            
            val_loss_sum += float(loss)
            dice_sum     += float(dice_val)

    net.train()

    denom = max(num_val_batches, 1)
    metrics = {
        'val_loss':  val_loss_sum / denom,
        'dice': dice_sum / denom,
        'precision': prec_sum / denom,
        'recall': rec_sum / denom,
        'iou': iou_sum / denom,
    }
    return metrics
