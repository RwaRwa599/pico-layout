import json, os, shutil, subprocess, urllib.request, zipfile
from pathlib import Path
from google.colab import files
from IPython.display import Image, display

LAYOUT_RELEASE = "https://github.com/RwaRwa599/pico-layout/releases/download/layout-v1"
LAYOUT_FILES = ("boxes.json", "template.json", "config.json", "canvas.jpg")
CANVAS_PATH = Path("/content/OMRChecker/github_canvas.jpg")

os.chdir("/content/OMRChecker")

Path("src/utils/interaction.py").write_text("""from dataclasses import dataclass
from pathlib import Path
import cv2
from src.logger import logger
from src.utils.image import ImageUtils

class _HeadlessMonitor:
    width, height, x, y = 1920, 1080, 0, 0

try:
    from screeninfo import get_monitors
    monitor_window = get_monitors()[0]
except Exception:
    monitor_window = _HeadlessMonitor()

@dataclass
class ImageMetrics:
    window_width, window_height = 1920, 1080
    window_x, window_y = 0, 0
    reset_pos = [0, 0]

class InteractionUtils:
    image_metrics = ImageMetrics()

    @staticmethod
    def show(name, origin, pause=1, resize=False, reset_pos=None, config=None):
        dest = Path("outputs/pico14/layouts")
        dest.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() else "_" for ch in str(name))[:80]
        path = dest / f"{safe}.jpg"
        if origin is None:
            logger.info(f"{name}: empty image")
            return
        img = origin
        if resize:
            if not config:
                raise Exception("config required for resize")
            img = ImageUtils.resize_util(origin, config.dimensions.display_width)
        cv2.imwrite(str(path), img)
        logger.info(f"layout saved {path}")

@dataclass
class Stats:
    files_moved = 0
    files_not_moved = 0

def wait_q():
    return

def is_window_available(name):
    return False
""")

INPUT_DIR = Path("inputs/pico14")
OUTPUT_DIR = Path("outputs/pico14")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "layouts").mkdir(exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SKIP_STEMS = {"geometry_overlay", "canvas", "github_canvas"}


def fetch_layout(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    canvas = CANVAS_PATH
    for name in LAYOUT_FILES:
        url = f"{LAYOUT_RELEASE}/{name}"
        print("github", url)
        target = dest / name if name != "canvas.jpg" else canvas
        with urllib.request.urlopen(url) as r:
            target.write_bytes(r.read())
        print("saved", target.name, target.stat().st_size)
    return canvas


def portrait(gray):
    import cv2

    if gray.shape[1] > gray.shape[0]:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    return gray


def scale_to_width(gray, width):
    import cv2

    h, w = gray.shape[:2]
    if w == width:
        return gray
    h2 = int(round(h * width / w))
    return cv2.resize(gray, (width, h2), interpolation=cv2.INTER_AREA)


def align(src, ref):
    import cv2
    import numpy as np

    sift = cv2.SIFT_create(4000)
    k1, d1 = sift.detectAndCompute(src, None)
    k2, d2 = sift.detectAndCompute(ref, None)
    if d1 is None or d2 is None:
        raise SystemExit("SIFT found no descriptors; not resizing onto the canvas")
    matcher = cv2.BFMatcher()
    raw = matcher.knnMatch(d1, d2, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        raise SystemExit(f"SIFT matches {len(good)} < 12; not resizing onto the canvas")
    src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise SystemExit("homography failed")
    return cv2.warpPerspective(src, H, (ref.shape[1], ref.shape[0]))


def warp_pages(pages, canvas_path: Path) -> list[Path]:
    import cv2

    ref = cv2.imread(str(canvas_path), cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise SystemExit("could not read GitHub canvas.jpg")
    print("canvas", ref.shape[1], "x", ref.shape[0])
    out = []
    for path in pages:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise SystemExit(f"could not read {path.name}")
        gray = portrait(gray)
        print("before", path.name, gray.shape[1], "x", gray.shape[0])
        aligned = align(scale_to_width(gray, ref.shape[1]), ref)
        if aligned.shape != ref.shape:
            raise SystemExit(
                f"{path.name} warped to {aligned.shape[1]}x{aligned.shape[0]}, "
                f"canvas is {ref.shape[1]}x{ref.shape[0]}"
            )
        cv2.imwrite(str(path), aligned)
        print("aligned", path.name, aligned.shape[1], "x", aligned.shape[0])
        out.append(path)
    return out


def detect_empty_squares(gray):
    import cv2

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.08 * peri, True)
        if len(approx) != 4:
            continue
        x, y, bw, bh = map(int, cv2.boundingRect(approx))
        if bw < 8 or bh < 8 or bw > 22 or bh > 22:
            continue
        if abs(bw - bh) > 6:
            continue
        if y < 50 or y > 800:
            continue
        roi = gray[y : y + bh, x : x + bw]
        pad = max(1, min(bw, bh) // 4)
        inner = roi[pad : bh - pad, pad : bw - pad]
        if inner.size == 0 or inner.mean() < 185:
            continue
        found.append({"x": x, "y": y, "w": bw, "h": bh})
    uniq = []
    for t in found:
        cx, cy = t["x"] + t["w"] / 2, t["y"] + t["h"] / 2
        if any(
            abs(cx - (u["x"] + u["w"] / 2)) < 7 and abs(cy - (u["y"] + u["h"] / 2)) < 7
            for u in uniq
        ):
            continue
        uniq.append(t)
    return uniq


def snap_boxes(named, gray, max_dist=22):
    import numpy as np

    det = detect_empty_squares(gray)
    deltas = []
    for b in named:
        cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
        best = None
        for d in det:
            dx = d["x"] + d["w"] / 2 - cx
            dy = d["y"] + d["h"] / 2 - cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= max_dist and (best is None or dist < best[0]):
                best = (dist, dx, dy)
        if best:
            deltas.append((best[1], best[2]))
    if len(deltas) < 20:
        print("snap skipped, pairs", len(deltas), "det", len(det))
        return [dict(b) for b in named]
    mx = int(round(float(np.median([d[0] for d in deltas]))))
    my = int(round(float(np.median([d[1] for d in deltas]))))
    print("snap median", mx, my, "pairs", len(deltas), "det", len(det))
    used = set()
    out = []
    for b in named:
        nb = dict(b)
        nb["x"] = int(b["x"] + mx)
        nb["y"] = int(b["y"] + my)
        cx, cy = nb["x"] + nb["w"] / 2, nb["y"] + nb["h"] / 2
        best = None
        for i, d in enumerate(det):
            if i in used:
                continue
            dx = d["x"] + d["w"] / 2 - cx
            dy = d["y"] + d["h"] / 2 - cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= 12 and (best is None or dist < best[0]):
                best = (dist, i, d)
        if best:
            used.add(best[1])
            d = best[2]
            nb["x"], nb["y"], nb["w"], nb["h"] = d["x"], d["y"], d["w"], d["h"]
        out.append(nb)
    return out


def write_named_template(boxes, page_w, page_h, dest: Path) -> None:
    fields = {}
    for box in boxes:
        slug = box["slug"]
        bw = max(int(box["w"]), 1)
        bh = max(int(box["h"]), 1)
        ox, oy = int(box["x"]), int(box["y"])
        fields[slug] = {
            "fieldLabels": [slug],
            "origin": [ox, oy],
            "bubblesGap": bw,
            "labelsGap": bh,
            "bubbleValues": ["1"],
            "direction": "horizontal",
            "bubbleDimensions": [bw, bh],
        }
    dest.write_text(
        json.dumps(
            {
                "pageDimensions": [page_w, page_h],
                "bubbleDimensions": [15, 15],
                "emptyValue": "",
                "customLabels": {},
                "outputColumns": [],
                "preProcessors": [],
                "fieldBlocks": fields,
            },
            indent=2,
        )
    )
    print("wrote", dest, "blocks", len(fields), "page", [page_w, page_h])


def drop_overlay(dest: Path) -> None:
    for path in list(dest.iterdir()):
        if path.stem.lower() in SKIP_STEMS:
            path.unlink()
            print("drop", path.name)


def unpack_zips(dest: Path) -> None:
    for zpath in list(dest.glob("*.zip")):
        print("unpack", zpath.name)
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                inner = Path(info.filename).name
                if inner.startswith(".") or inner.startswith("__"):
                    continue
                suf = Path(inner).suffix.lower()
                if suf not in IMAGE_EXTS and suf != ".json":
                    continue
                out = dest / inner
                out.write_bytes(zf.read(info.filename))
                print("from zip", inner, out.stat().st_size)


def collect_jpgs(dest: Path) -> list[Path]:
    import cv2

    pages = []
    for path in sorted(dest.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        if path.stem.lower() in SKIP_STEMS:
            continue
        if path.suffix.lower() != ".jpg":
            img = cv2.imread(str(path))
            if img is None:
                raise SystemExit(f"could not read image {path.name}")
            jpg = dest / f"{path.stem}.jpg"
            cv2.imwrite(str(jpg), img)
            pages.append(jpg)
        else:
            pages.append(path)
    return pages


def save_batch(dest: Path) -> list[Path]:
    print(
        "Upload pico-geometry.zip from the geometry notebook, "
        "or multi-select aligned jpgs plus boxes.json / template.json."
    )
    uploaded = files.upload()
    if not uploaded:
        raise SystemExit("no file selected")
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    for name, blob in uploaded.items():
        fn = Path(name).name
        path = dest / fn
        path.write_bytes(blob)
        print("saved", fn, len(blob))
    unpack_zips(dest)
    drop_overlay(dest)
    canvas = fetch_layout(dest)
    return warp_pages(collect_jpgs(dest), canvas)


def write_template_from_boxes(boxes_path: Path, jpg: Path, dest: Path) -> None:
    import cv2

    boxes = json.loads(boxes_path.read_text())
    img = cv2.imread(str(jpg))
    h, w = img.shape[:2]
    fields = {}
    for box in boxes:
        slug = box["slug"]
        bw = max(int(box["w"]), 1)
        bh = max(int(box["h"]), 1)
        ox, oy = int(box["x"]), int(box["y"])
        fields[slug] = {
            "fieldLabels": [slug],
            "origin": [ox, oy],
            "bubblesGap": bw,
            "labelsGap": bh,
            "bubbleValues": ["1"],
            "direction": "horizontal",
            "bubbleDimensions": [bw, bh],
        }
    dest.write_text(
        json.dumps(
            {
                "pageDimensions": [w, h],
                "bubbleDimensions": [15, 15],
                "emptyValue": "",
                "customLabels": {},
                "outputColumns": [],
                "preProcessors": [],
                "fieldBlocks": fields,
            },
            indent=2,
        )
    )
    print("wrote", dest, "blocks", len(fields), "page", [w, h])


pages = save_batch(INPUT_DIR)
print("inputs", sorted(p.name for p in INPUT_DIR.iterdir()))
jpgs = pages
if not jpgs:
    raise SystemExit("no jpg in the batch")

template_path = INPUT_DIR / "template.json"
if not template_path.exists():
    raise SystemExit("GitHub layout fetch failed: no template.json")
print("using GitHub", template_path)

layout_in = Path("inputs/pico14_layout")
shutil.rmtree(layout_in, ignore_errors=True)
layout_in.mkdir()
one = jpgs[0]
shutil.copy(one, layout_in / one.name)
import cv2

named = json.loads((INPUT_DIR / "boxes.json").read_text())
gray = cv2.imread(str(one), cv2.IMREAD_GRAYSCALE)
snapped = snap_boxes(named, gray)
write_named_template(snapped, gray.shape[1], gray.shape[0], layout_in / "template.json")
cfg = INPUT_DIR / "config.json"
if cfg.exists():
    shutil.copy(cfg, layout_in / "config.json")
print("layout image", one.name)

cmd = [
    "python3",
    "main.py",
    "--setLayout",
    "--inputDir",
    str(layout_in),
    "--outputDir",
    str(OUTPUT_DIR),
]
print(" ".join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
print(r.stdout[-5000:])
print(r.stderr[-5000:])
print("exit", r.returncode)
for p in sorted((OUTPUT_DIR / "layouts").glob("*.jpg"))[:4]:
    print(p)
    display(Image(filename=str(p), width=520))
