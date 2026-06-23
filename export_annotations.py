import os
import json
import argparse
from datetime import datetime

import cv2
import numpy as np

import config
from model import HeatmapUNet
from predict import predict_mask

def grasp_point_of_component(prob, comp_bin):
    dt = cv2.distanceTransform(comp_bin, cv2.DIST_L2, 5)
    _, _, _, maxloc = cv2.minMaxLoc(dt)
    return [int(maxloc[0]), int(maxloc[1])]

def polygon_of_component(comp_bin, eps_ratio=0.01):
    cnts, _ = cv2.findContours(comp_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    eps = eps_ratio * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    return [float(v) for xy in approx for v in xy]

def parse_args():
    p = argparse.ArgumentParser(description="Экспорт аннотаций (COCO + grasp_point)")
    p.add_argument("--images-dir", default=config.IMAGES_DIR)
    p.add_argument("--model-path", default=config.MODEL_SAVE_PATH)
    p.add_argument("--threshold", type=float, default=config.MASK_THRESHOLD)
    p.add_argument("--min-area", type=int, default=60, help="мин. площадь объекта, px")
    p.add_argument("--out", default="annotations_coco.json")
    return p.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        return
    model = HeatmapUNet(in_channels=3, out_channels=1,
                        base_channels=config.BASE_CHANNELS).to(config.DEVICE)
    model.load_state_dict(__import__("torch").load(args.model_path, map_location=config.DEVICE))
    print(f"Loaded model from {args.model_path}")

    files = sorted(f for f in os.listdir(args.images_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))

    coco = {
        "info": {"description": "graspable top objects", "date": datetime.now().isoformat()},
        "categories": [{"id": 1, "name": "graspable_object", "supercategory": "object"}],
        "images": [],
        "annotations": [],
    }
    ann_id = 1
    for img_id, fname in enumerate(files, start=1):
        path = os.path.join(args.images_dir, fname)
        mask, _conf = predict_mask(path, model)
        h, w = mask.shape[:2]
        coco["images"].append({"id": img_id, "file_name": fname, "width": w, "height": h})

        binary = (mask >= args.threshold).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < args.min_area:
                continue
            comp = (labels == k).astype(np.uint8)
            x, y, bw, bh = (int(stats[k, c]) for c in
                            (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            score = float(mask[comp > 0].mean())
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x, y, bw, bh],
                "area": area,
                "segmentation": [polygon_of_component(comp)],
                "score": round(score, 4),
                "grasp_point": grasp_point_of_component(mask, comp),
            })
            ann_id += 1

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)
    print(f"Готово. Кадров: {len(coco['images'])}, объектов: {len(coco['annotations'])} -> {args.out}")

if __name__ == "__main__":
    main()
