"""Generate a small synthetic dataset for quick training demos.
Creates `dataset/with_mask` and `dataset/without_mask` with simple face-like images.
"""
from PIL import Image, ImageDraw
import os
import random

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
WITH_DIR = os.path.join(OUT_DIR, "with_mask")
WITHOUT_DIR = os.path.join(OUT_DIR, "without_mask")
IMG_SIZE = 224

os.makedirs(WITH_DIR, exist_ok=True)
os.makedirs(WITHOUT_DIR, exist_ok=True)

def draw_face(draw, bbox, skin_color):
    x0, y0, x1, y1 = bbox
    draw.ellipse(bbox, fill=skin_color)
    # eyes
    ex = (x0 + x1) // 3
    ey = (y0 + y1) // 3
    r = (x1 - x0) // 20
    draw.ellipse((ex-r, ey-r, ex+r, ey+r), fill=(0,0,0))
    draw.ellipse((ex*2 - ex - r, ey-r, ex*2 - ex + r, ey+r), fill=(0,0,0))


def generate_image(with_mask: bool, path: str):
    bg = (random.randint(0,40), random.randint(20,60), random.randint(30,80))
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg)
    draw = ImageDraw.Draw(img)

    # face position and color
    margin = 20
    face_bbox = (margin, margin//2, IMG_SIZE-margin, IMG_SIZE-margin)
    skin = (random.randint(200,255), random.randint(170,230), random.randint(130,200))
    draw_face(draw, face_bbox, skin)

    # mouth / mask
    fx0, fy0, fx1, fy1 = face_bbox
    mx0 = fx0 + int((fx1-fx0)*0.15)
    mx1 = fx1 - int((fx1-fx0)*0.15)
    my = fy0 + int((fy1-fy0)*0.65)
    if with_mask:
        # draw a rectangular mask across lower face
        mask_h = int((fy1-fy0)*0.25)
        draw.rectangle((mx0, my-mask_h//2, mx1, my+mask_h//2), fill=(30,144,255))
        # mask straps
        draw.line((fx0, my-mask_h//2, mx0, my-mask_h//2), fill=(80,80,80), width=2)
        draw.line((fx1, my-mask_h//2, mx1, my-mask_h//2), fill=(80,80,80), width=2)
    else:
        # mouth (smile)
        mouth_w = mx1 - mx0
        draw.arc((mx0, my-10, mx0+mouth_w, my+20), start=0, end=180, fill=(128,0,0), width=3)

    img.save(path, quality=90)


def generate(n_per_class=60):
    # clean existing files in dataset dirs
    for folder in (WITH_DIR, WITHOUT_DIR):
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except Exception:
                pass

    for i in range(n_per_class):
        generate_image(True, os.path.join(WITH_DIR, f"with_mask_{i:03d}.jpg"))
        generate_image(False, os.path.join(WITHOUT_DIR, f"without_mask_{i:03d}.jpg"))

    print(f"Generated {n_per_class} images per class in {OUT_DIR}")


if __name__ == "__main__":
    generate(60)
