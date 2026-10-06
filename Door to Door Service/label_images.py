"""
Label images locally (no Roboflow) and build a YOLO dataset.

For each image in dataset_raw/:
  drag a box around the minifig, then press ENTER or SPACE to save
  press c if the minifig is NOT in the image (saved as a negative example)
Ctrl+C to stop; rerunning skips images you've already labeled.

When every image is labeled it creates:
  dataset/images/{train,val}, dataset/labels/{train,val}, dataset/data.yaml
"""
import os, glob, random, shutil, cv2

SRC = "dataset_raw"
LBL = "dataset_raw_labels"
OUT = "dataset"
CLASS = "minifig"

os.makedirs(LBL, exist_ok=True)
imgs = sorted(glob.glob(f"{SRC}/*.jpg"))

for i, p in enumerate(imgs):
    name = os.path.splitext(os.path.basename(p))[0]
    lbl = f"{LBL}/{name}.txt"
    if os.path.exists(lbl):
        continue
    img = cv2.imread(p)
    h, w = img.shape[:2]
    print(f"[{i+1}/{len(imgs)}] {p}")
    x, y, bw, bh = cv2.selectROI("drag box, ENTER=save, c=no minifig", img,
                                 showCrosshair=True)
    with open(lbl, "w") as f:
        if bw > 0 and bh > 0:   # YOLO format: class cx cy w h (normalized)
            f.write(f"0 {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}\n")
cv2.destroyAllWindows()

# 80/20 train/val split
random.seed(0)
random.shuffle(imgs)
n_val = max(1, len(imgs) // 5)
if os.path.exists(OUT):
    shutil.rmtree(OUT)
for split, subset in (("val", imgs[:n_val]), ("train", imgs[n_val:])):
    os.makedirs(f"{OUT}/images/{split}")
    os.makedirs(f"{OUT}/labels/{split}")
    for p in subset:
        name = os.path.splitext(os.path.basename(p))[0]
        shutil.copy(p, f"{OUT}/images/{split}/")
        shutil.copy(f"{LBL}/{name}.txt", f"{OUT}/labels/{split}/")

with open(f"{OUT}/data.yaml", "w") as f:
    f.write(f"path: {os.path.abspath(OUT)}\n"
            "train: images/train\nval: images/val\n"
            f"names:\n  0: {CLASS}\n")
print(f"done: {len(imgs)-n_val} train, {n_val} val -> {OUT}/data.yaml")
