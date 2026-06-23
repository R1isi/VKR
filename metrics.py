import math

import cv2
import numpy as np
import torch

import config

def largest_component_center(binary):
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = centroids[largest]
    return float(cx), float(cy)

def grasp_point_from_mask(binary, sep_erode=None):
    if int((binary > 0).sum()) == 0:
        return None
    if sep_erode is None:
        sep_erode = getattr(config, "GRASP_SEP_ERODE", 9)
    eroded = cv2.erode(binary, np.ones((sep_erode, sep_erode), np.uint8))
    src = eroded if int((eroded > 0).sum()) > 0 else binary

    dt = cv2.distanceTransform(src, cv2.DIST_L2, 5)
    _, _, _, maxloc = cv2.minMaxLoc(dt)
    return float(maxloc[0]), float(maxloc[1])

class SegMetrics:

    def __init__(self, threshold_ratio, mask_threshold=0.5, suction_radius=None):
        self.threshold_ratio = threshold_ratio
        self.mask_threshold = mask_threshold
        self.suction_radius = suction_radius if suction_radius is not None \
            else getattr(config, "SUCTION_RADIUS_PX", 0)
        self.reset()

    def reset(self):
        self.n = 0
        self.n_bbox = 0
        self.sum_iou = 0.0
        self.sum_dice = 0.0
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.grasp_correct = 0
        self.sum_dist = 0.0

        self.det = []
        self.n_gt_inst = 0

    @torch.no_grad()
    def update(self, pred, gt, valid_mask, gt_centers):
        b, _, h, w = gt.shape
        threshold_px = self.threshold_ratio * math.sqrt(h * h + w * w)

        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        centers = gt_centers.detach().cpu().numpy()

        for i in range(b):
            gt_bin = (gt_np[i, 0] >= 0.5).astype(np.uint8)
            if gt_bin.sum() == 0:
                continue
            self.n += 1

            pred_i = pred_np[i, 0]
            pred_bin = (pred_i >= self.mask_threshold).astype(np.uint8)

            inter = float(np.logical_and(pred_bin, gt_bin).sum())
            union = float(np.logical_or(pred_bin, gt_bin).sum())
            p_sum = float(pred_bin.sum())
            g_sum = float(gt_bin.sum())
            self.sum_iou += inter / union if union > 0 else 0.0
            self.sum_dice += 2.0 * inter / (p_sum + g_sum) if (p_sum + g_sum) > 0 else 0.0

            tp = inter
            fp = p_sum - inter
            fn = g_sum - inter
            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.tn += float(h * w) - tp - fp - fn

            self._update_instances(pred_i, pred_bin, gt_bin)

            p = grasp_point_from_mask(pred_bin)
            if p is None:
                ay, ax = np.unravel_index(int(np.argmax(pred_i)), pred_i.shape)
                p = (float(ax), float(ay))

            gx, gy = centers[i]
            if gx >= 0:
                self.n_bbox += 1
                dist = math.sqrt((p[0] - gx) ** 2 + (p[1] - gy) ** 2)
                self.sum_dist += dist
                if dist <= threshold_px:
                    self.grasp_correct += 1

    def _update_instances(self, pred_i, pred_bin, gt_bin, min_area=20):
        ng, gl, gstats, _ = cv2.connectedComponentsWithStats(gt_bin, connectivity=8)
        gt_masks = [(gl == k) for k in range(1, ng) if gstats[k, cv2.CC_STAT_AREA] >= min_area]
        self.n_gt_inst += len(gt_masks)

        npc, pl, pstats, _ = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)
        preds = []
        for k in range(1, npc):
            if pstats[k, cv2.CC_STAT_AREA] < min_area:
                continue
            pm = (pl == k)
            preds.append((float(pred_i[pm].mean()), pm))

        matched = [False] * len(gt_masks)
        for score, pm in sorted(preds, key=lambda x: -x[0]):
            best_iou, best_j = 0.0, -1
            for j, gm in enumerate(gt_masks):
                if matched[j]:
                    continue
                inter = float(np.logical_and(pm, gm).sum())
                union = float(np.logical_or(pm, gm).sum())
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_j = iou, j
            is_tp = best_iou >= 0.5
            if is_tp:
                matched[best_j] = True
            self.det.append((score, is_tp))

    @property
    def iou(self):
        return self.sum_iou / self.n if self.n else 0.0

    @property
    def dice(self):
        return self.sum_dice / self.n if self.n else 0.0

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def pixel_acc(self):
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def ap50(self):
        if not self.det or self.n_gt_inst == 0:
            return 0.0
        det = sorted(self.det, key=lambda x: -x[0])
        tp = fp = 0
        prec, rec = [], []
        for _score, is_tp in det:
            if is_tp:
                tp += 1
            else:
                fp += 1
            prec.append(tp / (tp + fp))
            rec.append(tp / self.n_gt_inst)
        mrec = np.concatenate(([0.0], rec, [rec[-1]]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    @property
    def grasp_pck(self):
        return self.grasp_correct / self.n_bbox if self.n_bbox else 0.0

    @property
    def mean_dist(self):
        return self.sum_dist / self.n_bbox if self.n_bbox else float("inf")

    def as_dict(self):
        return {"iou": self.iou, "dice": self.dice, "f1": self.f1, "ap50": self.ap50,
                "precision": self.precision, "recall": self.recall, "pixel_acc": self.pixel_acc,
                "grasp_pck": self.grasp_pck, "mean_dist": self.mean_dist, "n": self.n}
