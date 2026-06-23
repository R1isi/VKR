import os
import argparse
import tkinter as tk
from tkinter import ttk, filedialog

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

import config
from model import HeatmapUNet
from predict import predict_mask, export_annotation

DISPLAY_W = 820
MIN_AREA = 60

BG = "#1e1e2e"
PANEL = "#262638"
CARD = "#2d2d44"
ACCENT = "#5b9cff"
ACCENT_DK = "#3f7fe0"
GREEN = "#3ddc84"
ORANGE = "#ffb454"
TEXT = "#e8e8f2"
MUTED = "#9a9ab0"

def detect_objects(mask, threshold):
    binary = (mask >= threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    objs = []
    for k in range(1, n):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < MIN_AREA:
            continue
        comp = (labels == k).astype(np.uint8)
        x, y, bw, bh = (int(stats[k, c]) for c in
                        (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        dt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        _, _, _, gp = cv2.minMaxLoc(dt)
        objs.append({"bbox": (x, y, x + bw, y + bh), "grasp": (int(gp[0]), int(gp[1])),
                     "score": float(mask[comp > 0].mean()), "area": area})
    objs.sort(key=lambda o: -o["area"])
    return objs

class OperatorApp:
    def __init__(self, root, model, model_name, images_dir, threshold):
        self.root = root
        self.model = model
        self.model_name = model_name
        self.threshold = threshold
        self.show_mask = False
        self.files = []
        self.idx = 0
        self.photo = None

        root.title("Подбор товара — оператор")
        root.configure(bg=BG)
        self._init_style()
        self._build_ui()
        if os.path.isdir(images_dir):
            self._load_folder(images_dir)

    def _init_style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        s.configure("TFrame", background=BG)
        s.configure("Header.TFrame", background=PANEL)
        s.configure("Card.TFrame", background=CARD)
        s.configure("TLabel", background=BG, foreground=TEXT)
        s.configure("Card.TLabel", background=CARD, foreground=TEXT)
        s.configure("Title.TLabel", background=PANEL, foreground=ACCENT,
                    font=("Segoe UI Semibold", 17))
        s.configure("Sub.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 10))
        s.configure("Big.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI Semibold", 13))
        s.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 9))
        s.configure("Grasp.TLabel", background=CARD, foreground=GREEN, font=("Consolas", 12, "bold"))

        s.configure("TButton", background=CARD, foreground=TEXT, borderwidth=0,
                    focuscolor=CARD, padding=(12, 7), font=("Segoe UI", 10))
        s.map("TButton", background=[("active", "#3a3a55")])
        s.configure("Accent.TButton", background=ACCENT, foreground="#0e1730",
                    font=("Segoe UI Semibold", 10), padding=(12, 7))
        s.map("Accent.TButton", background=[("active", ACCENT_DK)])

        s.configure("Horizontal.TScale", background=PANEL, troughcolor="#3a3a55")

        s.configure("Treeview", background=CARD, fieldbackground=CARD, foreground=TEXT,
                    rowheight=24, borderwidth=0)
        s.configure("Treeview.Heading", background=PANEL, foreground=ACCENT,
                    font=("Segoe UI Semibold", 9), borderwidth=0)
        s.map("Treeview", background=[("selected", ACCENT_DK)])
        s.configure("Status.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))

    def _build_ui(self):

        header = ttk.Frame(self.root, style="Header.TFrame", padding=(16, 10))
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(header, text="🤖  Подбор товара", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text=f"модель: {self.model_name}", style="Sub.TLabel").pack(side=tk.RIGHT)

        bar = ttk.Frame(self.root, padding=(12, 10))
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text="📂 Открыть папку", command=self.open_folder).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="◀", width=3, command=self.prev).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="▶", width=3, command=self.next).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="🎨 Маска", command=self.toggle_mask).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="💾 Сохранить аннотацию", style="Accent.TButton",
                   command=self.save).pack(side=tk.LEFT, padx=3)
        ttk.Label(bar, text=f"порог {self.threshold:.2f}", style="TLabel",
                  font=("Consolas", 10)).pack(side=tk.RIGHT, padx=8)

        body = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        img_card = ttk.Frame(body, style="Card.TFrame", padding=6)
        img_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Label(img_card, bg=CARD, bd=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        side = ttk.Frame(body, style="Card.TFrame", padding=12, width=270)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        side.pack_propagate(False)
        ttk.Label(side, text="ДОСТУПНЫЕ ОБЪЕКТЫ", style="Muted.TLabel").pack(anchor="w")
        self.count_lbl = ttk.Label(side, text="0", style="Big.TLabel")
        self.count_lbl.pack(anchor="w", pady=(0, 10))

        ttk.Label(side, text="ПРИОРИТЕТНЫЙ ЗАХВАТ", style="Muted.TLabel").pack(anchor="w")
        self.grasp_lbl = ttk.Label(side, text="—", style="Grasp.TLabel")
        self.grasp_lbl.pack(anchor="w")
        self.score_lbl = ttk.Label(side, text="", style="Muted.TLabel")
        self.score_lbl.pack(anchor="w", pady=(0, 10))

        ttk.Label(side, text="ВСЕ ОБЪЕКТЫ", style="Muted.TLabel").pack(anchor="w")
        self.tree = ttk.Treeview(side, columns=("n", "x", "y", "score"),
                                 show="headings", height=14)
        for col, txt, w in (("n", "#", 28), ("x", "X", 56), ("y", "Y", 56), ("score", "score", 64)):
            self.tree.heading(col, text=txt)
            self.tree.column(col, width=w, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True, pady=4)

        self.status = ttk.Label(self.root, text="", style="Status.TLabel",
                                anchor="w", padding=(12, 5))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        for seq, fn in (("<Left>", self.prev), ("<Right>", self.next),
                        ("s", self.save), ("m", self.toggle_mask)):
            self.root.bind(seq, lambda e, f=fn: f())

    def open_folder(self):
        d = filedialog.askdirectory(title="Папка с кадрами")
        if d:
            self._load_folder(d)

    def _load_folder(self, d):
        self.images_dir = d
        self.files = sorted(f for f in os.listdir(d)
                            if f.lower().endswith((".jpg", ".jpeg", ".png")))
        self.idx = 0
        if self.files:
            self.render()
        else:
            self.status.config(text=f"В папке нет изображений: {d}")

    def prev(self):
        if self.files:
            self.idx = (self.idx - 1) % len(self.files)
            self.render()

    def next(self):
        if self.files:
            self.idx = (self.idx + 1) % len(self.files)
            self.render()

    def toggle_mask(self):
        self.show_mask = not self.show_mask
        self.render()

    def save(self):
        if not self.files:
            return
        path, n = export_annotation(self.files[self.idx], self._mask, self.threshold)
        self.status.config(text=f"✓ Сохранено: {path}   (объектов: {n})")

    def render(self):
        path = os.path.join(self.images_dir, self.files[self.idx])
        mask, _conf = predict_mask(path, self.model)
        self._mask = mask
        objs = detect_objects(mask, self.threshold)

        disp = cv2.imread(path)
        if self.show_mask:
            heat = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
            colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            disp = cv2.addWeighted(disp, 0.6, colored, 0.4, 0)

        for i, o in enumerate(objs):
            x1, y1, x2, y2 = o["bbox"]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 220, 80), 2)
            cv2.putText(disp, str(i + 1), (x1 + 3, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 2, cv2.LINE_AA)
            if i == 0:
                cv2.circle(disp, o["grasp"], 6, (60, 60, 255), -1)
                cv2.drawMarker(disp, o["grasp"], (60, 60, 255), cv2.MARKER_CROSS, 18, 2)
            else:
                cv2.drawMarker(disp, o["grasp"], (84, 180, 255), cv2.MARKER_TILTED_CROSS, 10, 1)

        h, w = disp.shape[:2]
        scale = DISPLAY_W / w
        disp = cv2.resize(disp, (DISPLAY_W, int(h * scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.config(image=self.photo)

        self.count_lbl.config(text=str(len(objs)))
        if objs:
            gx, gy = objs[0]["grasp"]
            self.grasp_lbl.config(text=f"({gx}, {gy})")
            self.score_lbl.config(text=f"score {objs[0]['score']:.2f}")
        else:
            self.grasp_lbl.config(text="—")
            self.score_lbl.config(text="")
        self.tree.delete(*self.tree.get_children())
        for i, o in enumerate(objs, start=1):
            gx, gy = o["grasp"]
            self.tree.insert("", tk.END, values=(i, gx, gy, f"{o['score']:.2f}"))

        self.status.config(text=f"Кадр {self.idx + 1} / {len(self.files)}   •   объектов: {len(objs)}")

def parse_args():
    p = argparse.ArgumentParser(description="Операторский интерфейс подбора товара")
    p.add_argument("--model-path", default="seg_model_best.pth")
    p.add_argument("--images-dir", default=config.IMAGES_DIR)
    return p.parse_args()

def main():
    args = parse_args()
    model = HeatmapUNet(in_channels=3, out_channels=1, base_channels=config.BASE_CHANNELS).to(config.DEVICE)
    if not os.path.exists(args.model_path):
        print(f"Модель не найдена: {args.model_path}")
        return
    model.load_state_dict(torch.load(args.model_path, map_location=config.DEVICE))
    model.eval()

    root = tk.Tk()
    root.geometry("1180x760")
    OperatorApp(root, model, os.path.basename(args.model_path), args.images_dir,
                config.MASK_THRESHOLD)
    root.mainloop()

if __name__ == "__main__":
    main()
