from __future__ import annotations
from typing import Dict, Tuple
from pathlib import Path
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader

from src.training.losses import OrdinalAxisLoss, FocalLoss, get_axis_criterions, axis_loss
from src.utils.metrics import  convert_ordinal_to_class_preds, calculate_accuracy
from src.utils.data_utils import load_static_data
from src.utils.model_utils import create_model, load_checkpoint, save_checkpoint

@torch.inference_mode(False)
def train_loop(
    model: torch.nn.Module,
    loaders: Dict[str, DataLoader],                 # {"train":..., "val":...}
    cfg: dict,                                      # YAML 로드 결과
    device: torch.device,
) -> Dict[str, list]:
    train_loader, valid_loader = loaders["train"], loaders["val"]
    epochs        = int(cfg["train"]["total_epochs"])
    lr            = float(cfg["train"]["lr"])
    weight_decay  = float(cfg["train"]["weight_decay"])
    z_loss_weight = float(cfg["train"]["z_loss_weight"])
    asym_head     = bool(cfg["train"].get("asym_head", False))
    
    save_dir = Path(cfg["model_io"]["save_dir"])
    best_path   = save_dir / Path(cfg["model_io"]["best_model"])
    latest_path = save_dir / Path(cfg["model_io"]["latest_ckpt"])
    resume = bool(cfg['model_io'].get('train_resume', False))

    model.to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        fused=torch.cuda.is_available()
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {k: [] for k in [
        "train_loss","val_loss",
        "train_acc_exact","val_acc_exact",
        "train_acc_x","val_acc_x",
        "train_acc_y","val_acc_y",
        "train_acc_z","val_acc_z",
    ]}
    start_epoch = 0
    best_val_loss = float("inf")

    if resume and latest_path.exists():
        start_epoch, history_loaded = load_checkpoint(latest_path, model, optimizer, scheduler, for_training=True)
        # history 이어붙이기(선택)
        for k in history:
            history[k].extend(history_loaded.get(k, []))

    scaler = torch.amp.GradScaler(
        'cuda' if device.type == 'cuda' else 'cpu',
        enabled=(device.type == 'cuda')
    )

    # 손실 준비
    if asym_head:
        criterion = OrdinalAxisLoss.from_cfg(cfg)
        
    for epoch in range(start_epoch, epochs):
        # -------------------- Train --------------------
        model.train()
        train_losses = []
        # CPU로 누적(메모리 안전)
        tr_lbls = []
        tr_lx, tr_ly, tr_lz = [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train|{'Ordinal' if asym_head else 'Axis'}]", mininterval=0.5, smoothing=0)
        amp_enabled = (device.type == "cuda")
        
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=amp_enabled):
                lx, ly, lz = model(inputs)          # (B,Cx),(B,Cy),(B,Cz)
                if asym_head:
                    loss, _ = criterion(lx, ly, lz, labels)
                else:
                    criterions, _ = get_axis_criterions(epoch, cfg)
                    loss, _ = axis_loss(lx, ly, lz, labels, criterions)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 누적 (CPU로 이동해 GPU 메모리 압박 방지)
            bs = inputs.size(0)
            train_losses.append((loss.detach().float().cpu(), bs))
            tr_lbls.append(labels.detach().cpu())
            tr_lx.append(lx.detach().cpu())
            tr_ly.append(ly.detach().cpu())
            tr_lz.append(lz.detach().cpu())

        # epoch train metrics
        total_train = sum(bs for _, bs in train_losses)
        avg_train_loss = sum(v.item()*bs for v, bs in train_losses) / total_train

        tr_lbls  = torch.cat(tr_lbls, dim=0)
        tr_lx    = torch.cat(tr_lx, dim=0)
        tr_ly    = torch.cat(tr_ly, dim=0)
        tr_lz    = torch.cat(tr_lz, dim=0)

        if asym_head:
            px = convert_ordinal_to_class_preds(tr_lx)
            py = convert_ordinal_to_class_preds(tr_ly)
            pz = convert_ordinal_to_class_preds(tr_lz)
            cx = (px == tr_lbls[:,0]).sum().item()
            cy = (py == tr_lbls[:,1]).sum().item()
            cz = (pz == tr_lbls[:,2]).sum().item()
            ce = ((px == tr_lbls[:,0]) & (py == tr_lbls[:,1]) & (pz == tr_lbls[:,2])).sum().item()
        else:
            cx, cy, cz, ce = calculate_accuracy(tr_lx, tr_ly, tr_lz, tr_lbls)
        train_accs = dict(
            x=cx/total_train, y=cy/total_train, z=cz/total_train, exact=ce/total_train
        )

        # -------------------- Valid --------------------
        model.eval()
        val_losses = []
        va_lbls = []
        va_lx, va_ly, va_lz = [], [], []
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp_enabled):
            for inputs, labels in valid_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                lx, ly, lz = model(inputs)
                if asym_head:
                    vloss, _ = criterion(lx, ly, lz, labels)
                else:
                    criterions, _ = get_axis_criterions(epoch, cfg)
                    vloss, _ = axis_loss(lx, ly, lz, labels, criterions)

                bs = inputs.size(0)
                val_losses.append((vloss.detach().float().cpu(), bs))
                va_lbls.append(labels.detach().cpu())
                va_lx.append(lx.detach().cpu())
                va_ly.append(ly.detach().cpu())
                va_lz.append(lz.detach().cpu())

        total_val = sum(bs for _, bs in val_losses)
        avg_val_loss = sum(v.item()*bs for v, bs in val_losses) / total_val

        va_lbls = torch.cat(va_lbls, dim=0)
        va_lx   = torch.cat(va_lx, dim=0)
        va_ly   = torch.cat(va_ly, dim=0)
        va_lz   = torch.cat(va_lz, dim=0)

        if asym_head:
            px = convert_ordinal_to_class_preds(va_lx)
            py = convert_ordinal_to_class_preds(va_ly)
            pz = convert_ordinal_to_class_preds(va_lz)
            cx = (px == va_lbls[:,0]).sum().item()
            cy = (py == va_lbls[:,1]).sum().item()
            cz = (pz == va_lbls[:,2]).sum().item()
            ce = ((px == va_lbls[:,0]) & (py == va_lbls[:,1]) & (pz == va_lbls[:,2])).sum().item()
        else:
            cx, cy, cz, ce = calculate_accuracy(va_lx, va_ly, va_lz, va_lbls)
        val_accs = dict(
            x=cx/total_val, y=cy/total_val, z=cz/total_val, exact=ce/total_val
        )

        # step scheduler
        scheduler.step()

        # 기록
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc_exact"].append(train_accs["exact"])
        history["val_acc_exact"].append(val_accs["exact"])
        history["train_acc_x"].append(train_accs["x"])
        history["val_acc_x"].append(val_accs["x"])
        history["train_acc_y"].append(train_accs["y"])
        history["val_acc_y"].append(val_accs["y"])
        history["train_acc_z"].append(train_accs["z"])
        history["val_acc_z"].append(val_accs["z"])

        # 로그
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train -> Loss: {avg_train_loss:.4f} | Acc(Exact): {train_accs['exact']:.4f} "
              f"[X:{train_accs['x']:.3f}, Y:{train_accs['y']:.3f}, Z:{train_accs['z']:.3f}]")
        print(f"  Valid -> Loss: {avg_val_loss:.4f} | Acc(Exact): {val_accs['exact']:.4f} "
              f"[X:{val_accs['x']:.3f}, Y:{val_accs['y']:.3f}, Z:{val_accs['z']:.3f}]")

        # 체크포인트
        checkpoint = {'epoch': epoch + 1,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict(),
                      'val_loss': avg_val_loss, 
                      'history': history
        }
        save_checkpoint(checkpoint, latest_path)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(checkpoint, best_path)
            torch.save({"model_state_dict": model.state_dict()}, best_path)
            print(f"✔ New best model saved (Val Loss: {best_val_loss:.4f})")

    print("\n--- 훈련 완료! ---")
    return history