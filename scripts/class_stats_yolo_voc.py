from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

def yolo_box_counts(yolo_dir: Path):
    counts = Counter()
    files = list(yolo_dir.rglob("*.txt"))
    for p in files:
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            counts[cls] += 1
    return counts, len(files)

def voc_object_counts(voc_dir: Path):
    counts = Counter()
    files = list(voc_dir.rglob("*.xml"))
    for x in files:
        root = ET.parse(str(x)).getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name", default="UNKNOWN").strip()
            counts[name] += 1
    return counts, len(files)

if __name__ == "__main__":
    yolo_dir = Path("data/YOLO_darknet")
    voc_dir  = Path("data/PASCAL_VOC")

    yolo_counts, n_txt = yolo_box_counts(yolo_dir)
    voc_counts, n_xml  = voc_object_counts(voc_dir)

    print("YOLO txt files:", n_txt)
    print("YOLO total boxes:", sum(yolo_counts.values()))
    print("YOLO top:", yolo_counts.most_common(20))

    print("\nVOC xml files:", n_xml)
    print("VOC total objects:", sum(voc_counts.values()))
    print("VOC top:", voc_counts.most_common(20))
