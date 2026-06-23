import os
import glob
import json
import argparse
from datetime import datetime

import cv2
import numpy as np

IMAGES_DIR = "images"
LABELS_DIR = "labels"
SESSIONS_JSON = "sessions.json"
OUT_MASKS_DIR = "diff_masks"
OUT_TOP_DIR = "top_masks"
PREVIEW_DIR = "diff_preview"
TOP_PREVIEW_DIR = "top_preview"

DIFF_FLOOR = 20
GRABCUT_DILATE = 10
TOP_EXPOSED_RATIO = 0.6
TOP_SEP_ERODE = 7
GLOBAL_CHANGE_BOUNDARY = 0.20
TIME_GAP_SOFT = 120.0
TIME_GAP_HARD = 3600.0
MIN_SESSION_LEN = 3

def parse_args():
    p = argparse.ArgumentParser(description="Аннотирование по разности кадров")
    p.add_argument("--max-sessions", type=int, default=0, help="0 = все сессии")
    p.add_argument("--preview", type=int, default=40, help="сколько превью сохранить (0 — нет)")
    p.add_argument("--no-grabcut", action="store_true", help="отключить уточнение масок GrabCut")
    return p.parse_args()

def ts(name):
    return datetime.strptime(os.path.splitext(name)[0], "%Y-%m-%d %H-%M-%S-%f")

def gain_match(fa, fb):
    a = fa.reshape(-1, 3).mean(0)
    b = fb.reshape(-1, 3).mean(0)
    scale = np.where(b > 1, a / b, 1.0)
    return np.clip(fb.astype(np.float32) * scale, 0, 255).astype(np.uint8)

def read_bbox(label_path, w, h):
    if not os.path.exists(label_path):
        return None
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            xc, yc, bw, bh = (float(v) for v in parts[1:])
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            return max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return None

def diff_map(fa, fb):
    fb = gain_match(fa, fb)
    d = cv2.absdiff(fa, fb).max(axis=2).astype(np.uint8)
    return cv2.GaussianBlur(d, (5, 5), 0)

def global_changed_fraction(diff, thresh=30):
    return float((diff > thresh).mean())

def _fill_holes(mask):
    ff = mask.copy()
    h, w = mask.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 255)
    return mask | cv2.bitwise_not(ff)

def removed_object_mask(diff, bbox):
    region = np.zeros(diff.shape, np.uint8)
    if bbox is None:
        return region
    x1, y1, x2, y2 = bbox
    sub = diff[y1:y2, x1:x2]
    if sub.size == 0:
        return region

    otsu, _ = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(DIFF_FLOOR, 0.5 * otsu)
    binr = (sub > t).astype(np.uint8) * 255
    binr = cv2.morphologyEx(binr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binr = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    region[y1:y2, x1:x2] = binr

    n, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    if n <= 1:
        return region
    bbox_area = (x2 - x1) * (y2 - y1)
    keep = np.zeros_like(region)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= 0.03 * bbox_area:
            keep[labels == k] = 255
    if keep.sum() == 0:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep = (labels == largest).astype(np.uint8) * 255
    return _fill_holes(keep)

def refine_with_grabcut(img_bgr, seed_mask, bbox, iters=5):
    if bbox is None or int((seed_mask > 0).sum()) == 0:
        return seed_mask
    x1, y1, x2, y2 = bbox

    k = np.ones((GRABCUT_DILATE, GRABCUT_DILATE), np.uint8)
    grow = cv2.dilate(seed_mask, k)
    clip = np.zeros_like(seed_mask)
    clip[y1:y2, x1:x2] = 255
    grow = cv2.bitwise_and(grow, clip)

    gc = np.full(img_bgr.shape[:2], cv2.GC_BGD, np.uint8)
    gc[grow > 0] = cv2.GC_PR_BGD
    gc[seed_mask > 0] = cv2.GC_PR_FGD
    core = cv2.erode(seed_mask, np.ones((5, 5), np.uint8), iterations=2)
    gc[core > 0] = cv2.GC_FGD
    if int((gc == cv2.GC_FGD).sum()) == 0:
        return seed_mask
    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(img_bgr, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return seed_mask
    out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    region = np.zeros_like(out)
    region[y1:y2, x1:x2] = out[y1:y2, x1:x2]

    n, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    if n <= 1:
        return seed_mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return _fill_holes((labels == largest).astype(np.uint8) * 255)

def split_sessions(files):
    sessions = []
    cur = [files[0]]
    prev_img = cv2.imread(os.path.join(IMAGES_DIR, files[0]))
    for i in range(1, len(files)):
        cur_img = cv2.imread(os.path.join(IMAGES_DIR, files[i]))
        gap = (ts(files[i]) - ts(files[i - 1])).total_seconds()
        boundary = gap > TIME_GAP_HARD
        if not boundary and prev_img is not None and cur_img is not None:
            gc = global_changed_fraction(diff_map(prev_img, cur_img))
            if gc > GLOBAL_CHANGE_BOUNDARY and gap > TIME_GAP_SOFT:
                boundary = True
        if boundary:
            sessions.append(cur)
            cur = []
        cur.append(files[i])
        prev_img = cur_img
    if cur:
        sessions.append(cur)
    return [s for s in sessions if len(s) >= MIN_SESSION_LEN]

def top_objects_mask(masks_from_t):
    if not masks_from_t:
        return None
    occupied = np.zeros_like(masks_from_t[0])
    top_union = np.zeros_like(masks_from_t[0])
    kernel = np.ones((TOP_SEP_ERODE, TOP_SEP_ERODE), np.uint8) if TOP_SEP_ERODE > 0 else None
    for m in masks_from_t:
        area = int((m > 0).sum())
        if area == 0:
            continue
        exposed = cv2.bitwise_and(m, cv2.bitwise_not(occupied))
        if int((exposed > 0).sum()) >= TOP_EXPOSED_RATIO * area:
            piece = exposed
            if kernel is not None:

                piece = cv2.erode(piece, kernel)
            top_union = cv2.bitwise_or(top_union, piece)
        occupied = cv2.bitwise_or(occupied, m)
    return top_union

def main():
    args = parse_args()
    os.makedirs(OUT_MASKS_DIR, exist_ok=True)
    os.makedirs(OUT_TOP_DIR, exist_ok=True)
    if args.preview:
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        os.makedirs(TOP_PREVIEW_DIR, exist_ok=True)

    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    print(f"Кадров: {len(files)}")

    if os.path.exists(SESSIONS_JSON):
        sessions = [[b + ".jpg" for b in s] for s in json.load(open(SESSIONS_JSON, encoding="utf-8"))]
        print(f"Сессии из {SESSIONS_JSON}: {len(sessions)}")
    else:
        sessions = split_sessions(files)
        print(f"Сессий (эвристика): {len(sessions)} | длины: {sorted((len(s) for s in sessions), reverse=True)[:15]}...")

    if args.max_sessions:
        sessions = sessions[:args.max_sessions]

    prev_a = args.preview
    prev_b = args.preview
    n_a = n_b = 0
    for si, sess in enumerate(sessions):
        n = len(sess) - 1
        if n <= 0:
            continue

        masks = []
        for t in range(n):
            fa = cv2.imread(os.path.join(IMAGES_DIR, sess[t]))
            fb = cv2.imread(os.path.join(IMAGES_DIR, sess[t + 1]))
            if fa is None or fb is None:
                masks.append(None)
                continue
            h, w = fa.shape[:2]
            bbox = read_bbox(os.path.join(LABELS_DIR, os.path.splitext(sess[t])[0] + ".txt"), w, h)
            mask = removed_object_mask(diff_map(fa, fb), bbox)
            if not args.no_grabcut:
                mask = refine_with_grabcut(fa, mask, bbox)
            masks.append(mask)
            base = os.path.splitext(sess[t])[0]
            cv2.imwrite(os.path.join(OUT_MASKS_DIR, base + ".png"), mask)
            n_a += 1

            if prev_a > 0:
                color = np.zeros_like(fa); color[:, :, 1] = mask
                overlay = cv2.addWeighted(fa, 1.0, color, 0.5, 0)
                if bbox is not None:
                    cv2.rectangle(overlay, bbox[:2], bbox[2:], (0, 0, 255), 2)
                cv2.imwrite(os.path.join(PREVIEW_DIR, f"s{si:02d}_{base}.png"), overlay)
                prev_a -= 1

        for t in range(n):
            present = [m for m in masks[t:] if m is not None]
            top = top_objects_mask(present)
            if top is None:
                continue
            base = os.path.splitext(sess[t])[0]
            cv2.imwrite(os.path.join(OUT_TOP_DIR, base + ".png"), top)
            n_b += 1

            if prev_b > 0:
                fa = cv2.imread(os.path.join(IMAGES_DIR, sess[t]))
                if fa is not None:
                    color = np.zeros_like(fa); color[:, :, 1] = top
                    overlay = cv2.addWeighted(fa, 1.0, color, 0.5, 0)
                    cv2.imwrite(os.path.join(TOP_PREVIEW_DIR, f"s{si:02d}_{base}.png"), overlay)
                    prev_b -= 1

    print(f"Готово. Маски A (убранный объект): {n_a} -> {OUT_MASKS_DIR}/")
    print(f"        Маски B (верхние объекты): {n_b} -> {OUT_TOP_DIR}/")
    if args.preview:
        print(f"Превью A (убранный объект): {PREVIEW_DIR}/")
        print(f"Превью B (верхние объекты): {TOP_PREVIEW_DIR}/")

if __name__ == "__main__":
    main()
