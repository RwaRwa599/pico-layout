# AE2: same routing as AE, three changes.
# --link class (LOINC CLASS), so lipid/renal cliques can keep pair scores.
# n < 2 mapped ticks go to flags.json (VLM), not ticks.json.
# missing still lists catalogue slugs with a positive class-pair score.
# AE2 does not write a tick of its own.

import json, shutil, urllib.request, zipfile
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd
from google.colab import files

# Caller-supplied. class is the AE2 default. cutoff None means do not skip on mean.
LINK = "class"
CUTOFF = None

WORK = Path("/content/ae2")
ZIP_PATH = Path("/content/pico-ae2.zip")
SKIP_STEMS = {"geometry_overlay"}
CSV_SKIP = {"file_id", "input_path", "output_path", "score"}
AE_RELEASE = "https://github.com/RwaRwa599/pico-layout/releases/download/ae-v1"
AE_FILES = ("pico_to_loinc.json", "train.json")
FIELDS_URL = "https://raw.githubusercontent.com/RwaRwa599/pico-layout/main/loinc-fields.json"


@dataclass(frozen=True)
class Assumption:
    extra: tuple[str, ...]
    missing: tuple[str, ...]
    correlation: float | None
    skipped: bool


def loinc_set(mapping, slug):
    raw = mapping[slug]
    return frozenset((raw,)) if isinstance(raw, str) else frozenset(raw)


def pair_key(a, b):
    return (a, b) if a < b else (b, a)


def pair_counts_from_orders(orders):
    counts = Counter()
    for tests in orders:
        uniq = sorted(set(tests))
        for a, b in combinations(uniq, 2):
            counts[(a, b)] += 1
    return dict(counts)


def train_count(counts, a, b):
    if a == b:
        return 0
    return int(counts.get(pair_key(a, b), 0))


def shared_mapped_loinc(mapping):
    def link(a, b):
        if a == b:
            return 0
        return 1 if loinc_set(mapping, a) & loinc_set(mapping, b) else 0

    return link


def _field_sets(mapping, fields, a, b):
    sa = {fields[c] for c in loinc_set(mapping, a) if c in fields}
    sb = {fields[c] for c in loinc_set(mapping, b) if c in fields}
    return sa, sb


def shared_class(mapping, class_of):
    def link(a, b):
        if a == b:
            return 0
        sa, sb = _field_sets(mapping, class_of, a, b)
        return 1 if sa and sb and sa & sb else 0

    return link


def shared_component(mapping, component_of):
    def link(a, b):
        if a == b:
            return 0
        sa, sb = _field_sets(mapping, component_of, a, b)
        return 1 if sa and sb and sa & sb else 0

    return link


def _member_of(mapping, panel, slug):
    if panel == slug:
        return False
    parent, child = loinc_set(mapping, panel), loinc_set(mapping, slug)
    return bool(child) and child < parent


def _covered_by_ticked_panel(slug, ticks, mapping):
    return any(_member_of(mapping, t, slug) for t in ticks)


def pair_score(a, b, counts, link):
    return train_count(counts, a, b) * (1 if link(a, b) else 0)


def assume(ticks, *, mapping, catalogue, pair_counts, link, cutoff=None):
    seen = []
    for slug in ticks:
        if slug not in mapping:
            raise KeyError(f"tick {slug!r} is not in pico_to_loinc")
        if slug not in seen:
            seen.append(slug)
    n = len(seen)
    if n < 2:
        return Assumption(extra=(), missing=(), correlation=None, skipped=True)
    scores = [pair_score(a, b, pair_counts, link) for a, b in combinations(seen, 2)]
    correlation = sum(scores) / len(scores)
    skipped = cutoff is not None and correlation < cutoff
    if skipped:
        return Assumption(extra=(), missing=(), correlation=correlation, skipped=True)
    ticked = set(seen)
    extra = [
        slug
        for slug in seen
        if not any(pair_score(slug, other, pair_counts, link) > 0 for other in seen if other != slug)
        and not _covered_by_ticked_panel(slug, ticked, mapping)
    ]
    missing = [
        slug
        for slug in catalogue
        if slug in mapping
        and slug not in ticked
        and any(pair_score(t, slug, pair_counts, link) > 0 for t in seen)
        and not _covered_by_ticked_panel(slug, ticked, mapping)
        and not any(_member_of(mapping, slug, t) for t in ticked)
    ]
    return Assumption(
        extra=tuple(extra),
        missing=tuple(missing),
        correlation=correlation,
        skipped=False,
    )


def load_pico_map(path: Path) -> dict:
    data = json.loads(path.read_text("utf-8"))
    out = {}
    for row in data["maps"]:
        if row["loinc"] is None:
            raise SystemExit(f"{row['slug']} has no LOINC")
        out[row["slug"]] = row["loinc"]
    return out


def load_train_pair_counts(path: Path) -> dict:
    data = json.loads(path.read_text("utf-8"))
    return pair_counts_from_orders(row["tests"] for row in data["orders"])


def loinc_node_fields(graph_path: Path) -> dict:
    graph = json.loads(graph_path.read_text("utf-8"))
    return {n["loinc"]: n for n in graph["nodes"] if n.get("kind") == "loinc_code"}


def field_index(nodes, field: str) -> dict:
    return {code: node[field] for code, node in nodes.items() if node.get(field)}


def fetch_ae_tables(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in AE_FILES:
        url = f"{AE_RELEASE}/{name}"
        print("github", url)
        with urllib.request.urlopen(url) as r:
            (dest / name).write_bytes(r.read())
        print("saved", name, (dest / name).stat().st_size)
    if not (dest / "loinc-fields.json").exists() and not (dest / "loinc-graph.json").exists():
        print("github", FIELDS_URL)
        with urllib.request.urlopen(FIELDS_URL) as r:
            blob = r.read()
        (dest / "loinc-fields.json").write_bytes(blob)
        print("saved loinc-fields.json", len(blob))


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
                (dest / inner).write_bytes(zf.read(info.filename))
                print("from zip", inner)


def pred_from_csv(path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(path)
    id_col = df.columns[0]
    skip = CSV_SKIP | {id_col}
    pred = {}
    for _, row in df.iterrows():
        stem = Path(str(row[id_col])).stem
        if stem.lower() in SKIP_STEMS:
            print("skip overlay row", stem)
            continue
        ticks = [
            c
            for c in df.columns
            if c not in skip and str(row[c]).strip() in {"1", "1.0"}
        ]
        pred[stem] = ticks
    return pred


def pred_from_json(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text("utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path.name} is not a stem -> slugs object")
    out = {}
    for stem, ticks in data.items():
        if str(stem).lower() in SKIP_STEMS:
            continue
        if not isinstance(ticks, list):
            raise SystemExit(f"{path.name} {stem} is not a list")
        out[str(stem)] = [str(s) for s in ticks]
    return out


def load_omr(dest: Path) -> dict[str, list[str]]:
    assume_path = dest / "assume.json"
    if assume_path.exists():
        data = json.loads(assume_path.read_text("utf-8"))
        pages = data.get("pages")
        if isinstance(pages, dict):
            pred = {stem: list(p.get("omr") or []) for stem, p in pages.items()}
            print("omr from assume.json pages", len(pred))
            return pred
    for name in ("omr.json", "pred.json"):
        p = dest / name
        if p.exists():
            pred = pred_from_json(p)
            print("omr json", p.name, "pages", len(pred))
            return pred
    csvs = sorted(dest.glob("Results_*.csv")) or sorted(dest.glob("*.csv"))
    if csvs:
        pred = pred_from_csv(csvs[0])
        print("omr csv", csvs[0].name, "pages", len(pred))
        return pred
    raise SystemExit(
        "need Untitled9 Results_*.csv, omr.json, or assume.json (pages.*.omr)"
    )


def make_link(mapping, dest: Path):
    if LINK == "mapped":
        return shared_mapped_loinc(mapping)
    graph = dest / "loinc-fields.json"
    if not graph.exists():
        graph = dest / "loinc-graph.json"
    if not graph.exists():
        raise SystemExit("loinc-fields.json required for --link class/component")
    nodes = loinc_node_fields(graph)
    idx = field_index(nodes, "class" if LINK == "class" else "component")
    if LINK == "class":
        return shared_class(mapping, idx)
    if LINK == "component":
        return shared_component(mapping, idx)
    raise SystemExit(f"LINK must be mapped, class, or component, got {LINK!r}")


print(
    "AE2: class link, singletons to VLM, missing on. "
    "Pulling pico_to_loinc.json, train.json, loinc-fields.json. "
    "Upload Untitled9 Results_*.csv, omr.json, or assume.json."
)
if LINK not in {"mapped", "class", "component"}:
    raise SystemExit("set LINK to mapped, class, or component at the top of this cell")

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
fetch_ae_tables(WORK)

map_path = WORK / "pico_to_loinc.json"
train_path = WORK / "train.json"
if not map_path.exists() or not train_path.exists():
    raise SystemExit("GitHub AE table fetch failed")

mapping = load_pico_map(map_path)
pair_counts = load_train_pair_counts(train_path)
cat = list(mapping)
link = make_link(mapping, WORK)
omr = load_omr(WORK)

ticks_out = {}
flags_out = {}
pages = {}
for stem, omr_ticks in omr.items():
    unmapped = [s for s in omr_ticks if s not in mapping]
    mapped_ticks = [s for s in omr_ticks if s in mapping]
    if unmapped:
        print("unmapped", stem, unmapped)
    got = assume(
        mapped_ticks,
        mapping=mapping,
        catalogue=cat,
        pair_counts=pair_counts,
        link=link,
        cutoff=CUTOFF,
    )
    extra = list(got.extra)
    missing = list(got.missing)
    if got.skipped and got.correlation is None:
        # n < 2: do not commit; VLM reads the singleton (or empty page).
        direct = []
        vlm = list(mapped_ticks) + list(unmapped)
    elif got.skipped:
        direct = list(mapped_ticks)
        vlm = list(unmapped)
    else:
        extra_set = set(extra)
        direct = [s for s in mapped_ticks if s not in extra_set]
        vlm = extra + missing + unmapped
    ticks_out[stem] = direct
    flags_out[stem] = vlm
    pages[stem] = {
        "omr": omr_ticks,
        "ticks": direct,
        "vlm": vlm,
        "extra": extra,
        "missing": missing,
        "unmapped": unmapped,
        "correlation": got.correlation,
        "skipped": got.skipped,
    }
    print(
        stem,
        "omr",
        omr_ticks,
        "ticks",
        direct,
        "vlm",
        vlm,
        "corr",
        got.correlation,
        "skip",
        got.skipped,
    )

out_dir = WORK / "out"
out_dir.mkdir(exist_ok=True)
(out_dir / "ticks.json").write_text(json.dumps(ticks_out, indent=2))
(out_dir / "flags.json").write_text(json.dumps(flags_out, indent=2))
(out_dir / "assume.json").write_text(
    json.dumps({"link": LINK, "cutoff": CUTOFF, "pages": pages}, indent=2)
)
for name in ("boxes.json", "template.json", "gold.json"):
    src = WORK / name
    if src.exists():
        (out_dir / name).write_bytes(src.read_bytes())

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out_dir.iterdir()):
        z.write(p, p.name)
        print("zip", p.name)
print("wrote", ZIP_PATH)
files.download(str(ZIP_PATH))
