import os
import json
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config

def session_split(sessions_json=config.SESSIONS_JSON, ratios=config.SPLIT_RATIOS,
                  seed=config.SPLIT_SEED):
    sessions = json.load(open(sessions_json, encoding="utf-8"))
    idx = list(range(len(sessions)))
    random.Random(seed).shuffle(idx)
    n = len(idx)
    n_tr = int(n * ratios[0])
    n_val = int(n * ratios[1])
    groups = {
        "train": idx[:n_tr],
        "val": idx[n_tr:n_tr + n_val],
        "test": idx[n_tr + n_val:],
    }
    out = {}
    for split, ids in groups.items():
        frames = []
        for si in ids:
            frames += sessions[si][:-1]
        out[split] = frames
    return out

class FrameMaskDataset(Dataset):

    def __init__(self, basenames, images_dir=config.IMAGES_DIR, masks_dir=config.MASKS_DIR,
                 valid_masks_dir=config.VALID_MASKS_DIR, labels_dir=config.LABELS_DIR,
                 img_size=config.IMG_SIZE, augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.valid_masks_dir = valid_masks_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.augment = augment
        self.items = []
        for name in basenames:
            img_p = os.path.join(images_dir, name + ".jpg")
            mask_p = os.path.join(masks_dir, name + ".png")
            if os.path.exists(img_p) and os.path.exists(mask_p):
                self.items.append((
                    img_p, mask_p,
                    os.path.join(valid_masks_dir, name + ".png"),
                    os.path.join(labels_dir, name + ".txt"),
                ))

    def __len__(self):
        return len(self.items)

    def _read_grasp_center(self, label_path):
        if not os.path.exists(label_path):
            return (-1.0, -1.0)
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                xc, yc = float(parts[1]), float(parts[2])
                return (xc * self.img_size[1], yc * self.img_size[0])
        return (-1.0, -1.0)

    def _augment(self, img, mask, valid, center):
        h, w = img.shape[:2]
        angle = random.uniform(-config.AUG_ROTATION_DEG, config.AUG_ROTATION_DEG)
        scale = random.uniform(*config.AUG_SCALE)
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        m[0, 2] += random.uniform(-config.AUG_TRANSLATE, config.AUG_TRANSLATE) * w
        m[1, 2] += random.uniform(-config.AUG_TRANSLATE, config.AUG_TRANSLATE) * h
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        valid = cv2.warpAffine(valid, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        cx, cy = center
        if cx >= 0:
            cx, cy = (m[0, 0] * cx + m[0, 1] * cy + m[0, 2], m[1, 0] * cx + m[1, 1] * cy + m[1, 2])

        if config.AUG_HFLIP and random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
            valid = cv2.flip(valid, 1)
            if cx >= 0:
                cx = w - 1 - cx

        alpha = 1.0 + random.uniform(-config.AUG_CONTRAST, config.AUG_CONTRAST)
        beta = random.uniform(-config.AUG_BRIGHTNESS, config.AUG_BRIGHTNESS) * 255
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return img, mask, valid, (cx, cy)

    def _load_mask(self, path):
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return np.zeros(self.img_size, np.uint8)
        return cv2.resize(m, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

    def __getitem__(self, idx):
        img_p, mask_p, valid_p, label_p = self.items[idx]
        img = cv2.imread(img_p)
        if img is None:
            raise ValueError(f"Cannot read image: {img_p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))

        mask = self._load_mask(mask_p)
        valid = self._load_mask(valid_p)
        center = self._read_grasp_center(label_p)

        if self.augment:
            img, mask, valid, center = self._augment(img, mask, valid, center)

        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        valid_t = torch.from_numpy((valid > 127).astype(np.float32)).unsqueeze(0)
        center_t = torch.tensor(center, dtype=torch.float32)
        return img_t, mask_t, valid_t, center_t
