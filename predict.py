import os
import json
import argparse

import cv2
import numpy as np
import torch

import config
from model import HeatmapUNet
from metrics import grasp_point_from_mask

def export_annotation(image_file, mask, threshold, out_dir="annotations", min_area=60):
    os.makedirs(out_dir, exist_ok=True)
    h, w = mask.shape[:2]
    binary = (mask >= threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    annotations = []
    for k in range(1, n):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (labels == k).astype(np.uint8)
        x, y, bw, bh = (int(stats[k, c]) for c in
                        (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        dt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        _, _, _, gp = cv2.minMaxLoc(dt)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly = []
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True).reshape(-1, 2)
            poly = [float(v) for xy in approx for v in xy]
        annotations.append({
            "id": len(annotations) + 1, "category_id": 1,
            "bbox": [x, y, bw, bh], "area": area, "segmentation": [poly],
            "score": round(float(mask[comp > 0].mean()), 4),
            "grasp_point": [int(gp[0]), int(gp[1])],
        })

    out = {
        "image": {"file_name": image_file, "width": w, "height": h},
        "categories": [{"id": 1, "name": "graspable_object"}],
        "annotations": annotations,
    }
    out_path = os.path.join(out_dir, os.path.splitext(image_file)[0] + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out_path, len(annotations)

def predict_mask(image_path, model, img_size=config.IMG_SIZE, device=config.DEVICE):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read {image_path}")
    h_orig, w_orig = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size[1], img_size[0]))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        mask = model(img_tensor).squeeze().cpu().numpy()

    mask = cv2.resize(mask, (w_orig, h_orig))
    return mask, float(mask.max())

def center_and_box_from_mask(mask, threshold):
    binary = (mask >= threshold).astype(np.uint8)
    point = grasp_point_from_mask(binary)
    if point is None:
        return None, None
    px, py = int(round(point[0])), int(round(point[1]))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    lbl = labels[py, px] if (0 <= py < binary.shape[0] and 0 <= px < binary.shape[1]) else 0
    if lbl > 0:
        x, y, bw, bh, _ = stats[lbl]
        box = (x, y, x + bw, y + bh)
    else:
        box = None
    return (px, py), box

def overlay_mask(img_bgr, mask, alpha=0.5):
    heat = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, colored, alpha, 0)

def parse_args():
    p = argparse.ArgumentParser(description="Просмотр предсказаний сегментации")
    p.add_argument("--images-dir", default=config.IMAGES_DIR,
                   help="Папка с изображениями для предсказания")
    p.add_argument("--model-path", default=config.MODEL_SAVE_PATH,
                   help="Путь к чекпойнту модели")
    p.add_argument("--threshold", type=float, default=config.MASK_THRESHOLD,
                   help="Порог бинаризации маски")
    return p.parse_args()

def main():
    args = parse_args()

    model = HeatmapUNet(in_channels=3, out_channels=1, base_channels=config.BASE_CHANNELS).to(config.DEVICE)
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        return
    model.load_state_dict(torch.load(args.model_path, map_location=config.DEVICE))
    print(f"Loaded model from {args.model_path}")

    if not os.path.isdir(args.images_dir):
        print(f"Directory {args.images_dir} not found")
        return
    image_files = sorted(
        f for f in os.listdir(args.images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not image_files:
        print("No images found")
        return

    idx = 0
    show_mask = False
    window_name = "Prediction (A/D: prev/next, H: mask, S: save annotation, ESC: exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        img_path = os.path.join(args.images_dir, image_files[idx])
        mask, conf = predict_mask(img_path, model)
        center, box = center_and_box_from_mask(mask, args.threshold)

        img_display = cv2.imread(img_path)
        if show_mask:
            img_display = overlay_mask(img_display, mask, alpha=0.5)

        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.drawMarker(img_display, center, (0, 0, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)

        cv2.imshow(window_name, img_display)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:
            break
        elif key in (ord('a'), ord('A')):
            idx = (idx - 1) % len(image_files)
        elif key in (ord('d'), ord('D')):
            idx = (idx + 1) % len(image_files)
        elif key in (ord('h'), ord('H')):
            show_mask = not show_mask
        elif key in (ord('s'), ord('S')):
            out_path, n_obj = export_annotation(image_files[idx], mask, args.threshold)
            print(f"Аннотация сохранена: {out_path} (объектов: {n_obj})")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
