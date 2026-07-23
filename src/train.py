"""
Trains the VGG16-FCN8 segmentation head on the Herlev dataset.

Usage:
    python src/train.py

Sweeps LR_CANDIDATES x EPOCH_CANDIDATES, saves a checkpoint for every
configuration to checkpoints/, keeps the best (by validation Dice) as
checkpoints/best_model.pt, and writes checkpoints/training_config.json
summarizing the winning run.

GPU memory hygiene: every checkpoint clone is moved to CPU before storing,
and the model/optimizer for each config are explicitly deleted with
torch.cuda.empty_cache() + gc.collect() between runs so a long sweep doesn't
accumulate resident VGG16 state_dicts on the GPU.
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BEST_MODEL_PATH,
    CHECKPOINT_DIR,
    DEVICE,
    SEED,
    TRAIN_CONFIG_PATH,
    require_data_root,
)
from dataset import HerlevSegDataset, scan_dataset
from model import VGG16_FCN8

# Widen back to the full paper-order sweep [1e-4, 5e-5] x [40, 50, 60] once a
# first pass has run clean end to end.
LR_CANDIDATES = [5e-5]
EPOCH_CANDIDATES = [80]
BATCH_SIZE = 8


def dice_score(pred_prob, target, eps=1e-6):
    pred = (pred_prob > 0.5).float()
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def train_fcn_head(
    model,
    train_imgs,
    train_masks,
    val_imgs,
    val_masks,
    lr=1e-5,
    epochs=40,
    batch_size=8,
    freeze_backbone=False,
):
    for p in model.parameters():
        p.requires_grad = True

    if freeze_backbone:
        for p in model.stage1.parameters():
            p.requires_grad = False
        for p in model.stage2.parameters():
            p.requires_grad = False
        for p in model.stage3.parameters():
            p.requires_grad = False

    train_dl = DataLoader(
        HerlevSegDataset(train_imgs, train_masks), batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_dl = DataLoader(
        HerlevSegDataset(val_imgs, val_masks), batch_size=batch_size, shuffle=False, num_workers=2
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    bce_criterion = nn.BCEWithLogitsLoss()

    best_dice, best_state = -1.0, None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, masks in train_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()

            out = model(imgs)
            logits = out[:, 0]

            bce_loss = bce_criterion(logits, masks)
            probs = torch.sigmoid(logits)
            intersection = (probs * masks).sum(dim=(1, 2))
            cardinality = probs.sum(dim=(1, 2)) + masks.sum(dim=(1, 2))
            dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (cardinality + 1e-6)).mean()

            loss = bce_loss + dice_loss

            loss.backward()
            opt.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / max(1, len(train_dl.dataset))

        model.eval()
        dices = []
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                out = model(imgs)
                dices.append(dice_score(torch.sigmoid(out[:, 0]), masks).item())

        val_dice = float(np.mean(dices)) if dices else 0.0
        print(f"  epoch {epoch + 1:02d}/{epochs} train_loss={train_loss:.4f} val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    del opt
    return model, best_dice


def main():
    require_data_root()
    samples = scan_dataset(require_data_root())
    print("Total matched image/mask pairs:", len(samples))
    if not samples:
        raise RuntimeError("No image/mask pairs were found under Data/. Check the folder layout.")

    image_paths = [row["image_path"] for row in samples]
    mask_paths = [row["mask_path"] for row in samples]
    labels = [row["class_name"] for row in samples]

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.15, random_state=SEED, stratify=labels
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_cfg, best_val_dice, best_model_state = None, -1.0, None
    for lr in LR_CANDIDATES:
        for ep in EPOCH_CANDIDATES:
            print(f"\n=== training LR={lr}  EPOCHS={ep} ===")
            m = VGG16_FCN8().to(DEVICE)
            m, val_dice = train_fcn_head(
                m, train_imgs, train_masks, val_imgs, val_masks, lr=lr, epochs=ep, batch_size=BATCH_SIZE
            )

            # checkpoint immediately - a later config crashing won't lose this one
            ckpt_path = CHECKPOINT_DIR / f"model_lr{lr}_ep{ep}.pt"
            torch.save({k: v.cpu() for k, v in m.state_dict().items()}, ckpt_path)
            print(f"  saved checkpoint -> {ckpt_path}  val_dice={val_dice:.4f}")

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_cfg = dict(lr=lr, epochs=ep)
                best_model_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}

            del m
            torch.cuda.empty_cache()
            gc.collect()

    print("Best training config:", best_cfg, " val_dice =", round(best_val_dice, 4))

    torch.save(best_model_state, BEST_MODEL_PATH)
    with open(TRAIN_CONFIG_PATH, "w") as f:
        json.dump({"best_cfg": best_cfg, "best_val_dice": best_val_dice}, f, indent=2)

    print(f"Saved best model -> {BEST_MODEL_PATH}")
    print(f"Saved training config -> {TRAIN_CONFIG_PATH}")

    # Save the split so tune_hyperparams.py / final_eval.py reuse the exact
    # same held-out validation set instead of re-splitting differently.
    split_path = CHECKPOINT_DIR / "train_val_split.json"
    with open(split_path, "w") as f:
        json.dump(
            {
                "train_imgs": train_imgs,
                "train_masks": train_masks,
                "val_imgs": val_imgs,
                "val_masks": val_masks,
            },
            f,
            indent=2,
        )
    print(f"Saved train/val split -> {split_path}")


if __name__ == "__main__":
    main()
