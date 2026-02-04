import json
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_image(img_dir: Path, stem_or_name: str):
    p = img_dir / stem_or_name
    if p.exists():
        return p
    stem = Path(stem_or_name).stem
    for ext in IMG_EXTS:
        cand = img_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    for ext in IMG_EXTS:
        cand = next(img_dir.rglob(f"{stem}{ext}"), None)
        if cand is not None:
            return cand
    return None


def draw_boxes(img_bgr, boxes, labels=None):
    out = img_bgr.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if labels and i < len(labels) and labels[i]:
            cv2.putText(
                out,
                str(labels[i]),
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
    return out


def parse_yolo_txt(txt_path: Path, img_w: int, img_h: int):
    boxes, labels = [], []
    for ln in txt_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append((x1, y1, x2, y2))
        labels.append(str(cls))
    return boxes, labels


def parse_pascal_xml(xml_path: Path):
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="obj")
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        x1 = float(bnd.findtext("xmin", default="0"))
        y1 = float(bnd.findtext("ymin", default="0"))
        x2 = float(bnd.findtext("xmax", default="0"))
        y2 = float(bnd.findtext("ymax", default="0"))
        boxes.append((x1, y1, x2, y2))
        labels.append(name)
    return boxes, labels


def parse_json_any(json_path: Path, img_name: str):
    data = json.loads(json_path.read_text(encoding="utf-8"))

    # COCO
    if isinstance(data, dict) and "images" in data and "annotations" in data and "categories" in data:
        cat = {c["id"]: c.get("name", str(c["id"])) for c in data["categories"]}
        target = None
        for im in data["images"]:
            fn = im.get("file_name", "")
            if Path(fn).name == Path(img_name).name or Path(fn).stem == Path(img_name).stem:
                target = im
                break
        if target is None:
            return [], []
        img_id = target["id"]
        boxes, labels = [], []
        for ann in data["annotations"]:
            if ann.get("image_id") != img_id:
                continue
            if "bbox" in ann and len(ann["bbox"]) == 4:
                x, y, w, h = ann["bbox"]
                boxes.append((x, y, x + w, y + h))
                labels.append(cat.get(ann.get("category_id"), str(ann.get("category_id"))))
        return boxes, labels

    # VIA-like
    records = []
    if isinstance(data, dict) and "regions" in data:
        records = [data]
    elif isinstance(data, dict):
        records = list(data.values())
    elif isinstance(data, list):
        records = data

    tgt_base = Path(img_name).name
    tgt_stem = Path(img_name).stem
    rec = None
    for r in records:
        fn = r.get("filename") or r.get("file_name") or r.get("name")
        if not fn:
            continue
        if Path(fn).name == tgt_base or Path(fn).stem == tgt_stem:
            rec = r
            break
    if rec is None:
        return [], []

    regions = rec.get("regions", [])
    if isinstance(regions, dict):
        regions = list(regions.values())

    boxes, labels = [], []
    for reg in regions:
        sa = reg.get("shape_attributes", {}) or reg.get("region_shape_attributes", {})
        ra = reg.get("region_attributes", {}) or {}

        if all(k in sa for k in ["x", "y", "width", "height"]):
            x, y, w, h = float(sa["x"]), float(sa["y"]), float(sa["width"]), float(sa["height"])
            boxes.append((x, y, x + w, y + h))
        elif "all_points_x" in sa and "all_points_y" in sa:
            xs, ys = sa["all_points_x"], sa["all_points_y"]
            if xs and ys:
                boxes.append((min(xs), min(ys), max(xs), max(ys)))
            else:
                continue
        else:
            continue

        if isinstance(ra, dict) and ra:
            k = next(iter(ra.keys()))
            labels.append(f"{k}:{ra[k]}")
        else:
            labels.append("")
    return boxes, labels


def find_ann(ann_dir: Path, img_path: Path):
    stem = img_path.stem
    for ext in [".txt", ".xml", ".json"]:
        cand = img_path.with_suffix(ext)
        if cand.exists():
            return cand
    for ext in [".txt", ".xml", ".json"]:
        hits = list(ann_dir.rglob(f"{stem}{ext}"))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", required=True, type=str)
    ap.add_argument("--img_dir", required=True, type=str)
    ap.add_argument("--out_dir", default="boxed_outputs", type=str)
    ap.add_argument("--limit", default=10, type=int)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    ann_dir = Path(args.ann_dir)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    imgs = imgs[: args.limit]

    for img_path in tqdm(imgs, desc="Images"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        ann_path = find_ann(ann_dir, img_path)
        boxes, labels = [], []
        if ann_path:
            if ann_path.suffix.lower() == ".txt":
                boxes, labels = parse_yolo_txt(ann_path, w, h)
            elif ann_path.suffix.lower() == ".xml":
                boxes, labels = parse_pascal_xml(ann_path)
            elif ann_path.suffix.lower() == ".json":
                boxes, labels = parse_json_any(ann_path, img_path.name)

        out_bgr = draw_boxes(img_bgr, boxes, labels)
        save_path = out_dir / f"{img_path.stem}_boxed.jpg"
        cv2.imwrite(str(save_path), out_bgr)

        if args.show:
            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 7))
            plt.title(f"{img_path.name} | boxes: {len(boxes)}")
            plt.imshow(out_rgb)
            plt.axis("off")
            plt.show()

    print(f"[DONE] Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
