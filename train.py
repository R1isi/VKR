import argparse
import json

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from dataset import FrameMaskDataset, session_split
from model import HeatmapUNet
from losses import DiceBCELoss
from metrics import SegMetrics

def save_metrics_plot(history, path):
    ep = history["epoch"]
    if not ep:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(ep, history["train_loss"], label="train")
    axes[0].plot(ep, history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history["iou"], label="IoU", color="tab:green")
    axes[1].plot(ep, history["dice"], label="Dice/F1", color="tab:blue")
    axes[1].plot(ep, history["precision"], label="precision", color="tab:orange")
    axes[1].plot(ep, history["recall"], label="recall", color="tab:purple")
    axes[1].set_title("Pixel quality"); axes[1].set_xlabel("epoch"); axes[1].set_ylim(0, 1)
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(ep, history["ap50"], label="AP50 (object)", color="tab:red")
    axes[2].set_title("Object AP@0.5"); axes[2].set_xlabel("epoch"); axes[2].set_ylim(0, 1)
    axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

def make_loader(basenames, shuffle, augment=False):
    dataset = FrameMaskDataset(basenames, augment=augment)
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE.type == "cuda"),
    )

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    loop = tqdm(loader, desc="train", leave=False)
    for imgs, masks, _valid, _centers in loop:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        loop.set_postfix(loss=loss.item())
    return total_loss / max(n_batches, 1)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    metrics = SegMetrics(threshold_ratio=config.PCK_THRESHOLD, mask_threshold=config.MASK_THRESHOLD,
                         suction_radius=config.SUCTION_RADIUS_PX)
    loop = tqdm(loader, desc="val  ", leave=False)
    for imgs, masks, valid, centers in loop:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        preds = model(imgs)
        total_loss += criterion(preds, masks).item()
        n_batches += 1
        metrics.update(preds, masks, valid, centers)
    return total_loss / max(n_batches, 1), metrics

def parse_args():
    p = argparse.ArgumentParser(description="Обучение сегментации следующего объекта (вариант А)")
    p.add_argument("--save-path", default=config.MODEL_SAVE_PATH,
                   help="Куда сохранять последний чекпойнт")
    p.add_argument("--best-save-path", default=config.BEST_MODEL_SAVE_PATH,
                   help="Куда сохранять лучшую по PCK модель")
    p.add_argument("--plot-path", default="training_metrics.png",
                   help="Куда сохранять график метрик")
    p.add_argument("--history-path", default="training_history.json",
                   help="Куда сохранять историю метрик (JSON) для построения графиков")
    return p.parse_args()

def main():
    args = parse_args()
    device = config.DEVICE
    print(f"Using device: {device}")

    splits = session_split()
    print(f"Кадров: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} (сплит по сессиям)")

    train_loader = make_loader(splits["train"], shuffle=True, augment=config.AUG_ENABLED)
    if train_loader is None:
        raise RuntimeError("Не найдены train-данные. Сгенерируй маски: python annotate_by_diff.py")
    val_loader = make_loader(splits["val"], shuffle=False)
    if val_loader is None:
        print("Внимание: val-данные не найдены, валидация будет пропускаться")

    model = HeatmapUNet(in_channels=3, out_channels=1, base_channels=config.BASE_CHANNELS).to(device)
    criterion = DiceBCELoss()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = {k: [] for k in ("epoch", "train_loss", "val_loss", "iou",
                               "dice", "precision", "recall", "ap50", "pixel_acc", "mean_dist")}
    best_iou = -1.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        if val_loader is not None:
            val_loss, metrics = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{config.EPOCHS} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"IoU={metrics.iou:.3f} | Dice/F1={metrics.f1:.3f} | "
                f"AP50={metrics.ap50:.3f} | P={metrics.precision:.3f} | R={metrics.recall:.3f} | "
                f"lr={lr:.2e}"
            )

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["iou"].append(metrics.iou)
            history["dice"].append(metrics.dice)
            history["precision"].append(metrics.precision)
            history["recall"].append(metrics.recall)
            history["ap50"].append(metrics.ap50)
            history["pixel_acc"].append(metrics.pixel_acc)
            history["mean_dist"].append(metrics.mean_dist)
            save_metrics_plot(history, args.plot_path)
            with open(args.history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

            if metrics.iou > best_iou:
                best_iou = metrics.iou
                torch.save(model.state_dict(), args.best_save_path)
                print(f"  -> новая лучшая модель (IoU={best_iou:.3f}) сохранена в {args.best_save_path}")

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        else:
            print(f"Epoch {epoch:3d}/{config.EPOCHS} | train={train_loss:.6f}")

        torch.save(model.state_dict(), args.save_path)

        if config.EARLY_STOP_PATIENCE and epochs_no_improve >= config.EARLY_STOP_PATIENCE:
            print(f"Early stopping: val_loss не падает {epochs_no_improve} эпох подряд. "
                  f"Лучший IoU={best_iou:.3f}")
            break

    if history["epoch"]:
        save_metrics_plot(history, args.plot_path)
        print(f"График метрик сохранён: {args.plot_path}")

if __name__ == "__main__":
    main()
