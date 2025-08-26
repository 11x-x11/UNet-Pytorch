import torch, contextlib
from unet import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device).eval()

def find_max_bs(model, image_shape=(3, 400, 400), start=8, device='cuda'):
    bs, ok = start, start
    # Use autocast on CUDA so ops run in mixed precision but weights stay FP32
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if device == 'cuda' else contextlib.nullcontext()
    with torch.no_grad(), amp_ctx:
        while True:
            try:
                # Make inputs FP32; autocast will downcast where safe
                x = torch.randn(bs, *image_shape, device=device, dtype=torch.float32)
                y = model(x)  # forward only
                del x, y
                torch.cuda.synchronize() if device == 'cuda' else None
                ok = bs
                bs *= 2
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache() if device == 'cuda' else None
                    break
                else:
                    raise
    # Training needs extra memory => back off by ~2×
    return max(1, ok // 2)

safe_bs = find_max_bs(model)
print(f"Recommended training batch size (with --amp): {safe_bs}")
