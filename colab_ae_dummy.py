# Dummy AE: flag every named box on every geometry page so VLM reads the full set.
# No correlation, extra, missing, or skip. Does not write a tick.
# Upload pico-geometry-4.zip (or the aligned jpgs + boxes.json).

import json, re, shutil, zipfile
from pathlib import Path

from google.colab import files

WORK = Path("/content/ae-dummy")
ZIP_PATH = Path("/content/pico-ae-dummy.zip")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SKIP_STEMS = {"geometry_overlay", "github_canvas", "canvas"}


def canon(stem: str) -> str:
    return re.sub(r"\s+\(\d+\)$", "", Path(stem).stem)


def unpack_zips(dest: Path) -> None:
    for zpath in list(dest.glob("*.zip")):
        print("unpack", zpath.name)
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                inner = Path(info.filename).name
                if not inner or inner.startswith(".") or inner.startswith("__"):
                    continue
                (dest / inner).write_bytes(zf.read(info.filename))
                print("from zip", inner)


print(
    "Dummy AE: flags every box on every page. "
    "Upload pico-geometry-4.zip (pages + boxes.json). Optional gold.json."
)
uploaded = files.upload()
if not uploaded:
    raise SystemExit("no file selected")

shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir(parents=True)
for name, blob in uploaded.items():
    path = WORK / Path(name).name
    path.write_bytes(blob)
    print("saved", path.name, len(blob))
unpack_zips(WORK)

boxes_path = WORK / "boxes.json"
if not boxes_path.exists():
    raise SystemExit("need boxes.json in the geometry zip")
boxes = json.loads(boxes_path.read_text())
slugs = [b["slug"] for b in boxes]
if not slugs:
    raise SystemExit("boxes.json has no slugs")

stems = []
seen = set()
for path in sorted(WORK.iterdir()):
    if path.suffix.lower() not in IMAGE_EXTS:
        continue
    stem = canon(path.stem)
    if stem.lower() in SKIP_STEMS:
        continue
    if stem in seen:
        print("skip duplicate", path.name)
        continue
    seen.add(stem)
    stems.append(stem)

if not stems:
    raise SystemExit("no page images in the upload")

flags = {stem: list(slugs) for stem in stems}
ticks = {stem: [] for stem in stems}
pages = {
    stem: {
        "omr": [],
        "ticks": [],
        "vlm": list(slugs),
        "extra": [],
        "missing": [],
        "unmapped": [],
        "correlation": None,
        "skipped": False,
        "dummy": True,
    }
    for stem in stems
}

out_dir = WORK / "out"
out_dir.mkdir(exist_ok=True)
(out_dir / "ticks.json").write_text(json.dumps(ticks, indent=2))
(out_dir / "flags.json").write_text(json.dumps(flags, indent=2))
(out_dir / "assume.json").write_text(
    json.dumps(
        {
            "link": "dummy",
            "cutoff": None,
            "dummy": True,
            "note": "every box flagged; VLM reads the full page",
            "pages": pages,
        },
        indent=2,
    )
)
for name in ("boxes.json", "template.json", "gold.json"):
    src = WORK / name
    if src.exists():
        (out_dir / name).write_bytes(src.read_bytes())

print("pages", len(stems), "boxes", len(slugs), "flagged crops", len(stems) * len(slugs))
for stem in stems:
    print(stem, "flags", len(flags[stem]))

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out_dir.iterdir()):
        z.write(p, p.name)
        print("zip", p.name)
print("wrote", ZIP_PATH)
files.download(str(ZIP_PATH))
