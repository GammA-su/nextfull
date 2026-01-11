import argparse
import html
import re
import sys
from collections import defaultdict
from pathlib import Path

import xxhash
from datasets import get_dataset_config_names, load_dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import setup_runtime

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
SENT_END_RE = re.compile(r"[.!?]")

SOURCES = ["comma", "fineweb_edu", "fineweb_hq", "fineweb2_hq"]


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    text = WS_RE.sub(" ", text)
    return text.strip()


def pick_text(sample):
    if not isinstance(sample, dict):
        return None
    if "text" in sample and isinstance(sample["text"], str):
        return sample["text"]
    if "content" in sample and isinstance(sample["content"], str):
        return sample["content"]
    for value in sample.values():
        if isinstance(value, str):
            return value
    return None


def alnum_ratio(text: str) -> float:
    if not text:
        return 0.0
    alnum = sum(1 for ch in text if ch.isalnum())
    return alnum / float(len(text))


def parse_weights(raw: str):
    weights = {name: 0.0 for name in SOURCES}
    if not raw:
        return weights
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid weight '{part}', expected key=value")
        key, val = part.split("=", 1)
        key = key.strip()
        if key not in weights:
            raise ValueError(f"unknown source '{key}', choices: {', '.join(SOURCES)}")
        weights[key] = float(val)
    return weights


def build_schedule(weights, scale: int = 100):
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to > 0")
    counts = {}
    for name in SOURCES:
        w = weights.get(name, 0.0)
        if w <= 0:
            counts[name] = 0
            continue
        count = int(round(scale * w / total))
        if count <= 0:
            count = 1
        counts[name] = count
    schedule = []
    counts_left = counts.copy()
    while any(v > 0 for v in counts_left.values()):
        for name in SOURCES:
            if counts_left.get(name, 0) > 0:
                schedule.append(name)
                counts_left[name] -= 1
    return schedule


def load_source(name: str, args, logger):
    if name == "comma":
        ds = load_dataset(
            "common-pile/comma_v0.1_training_dataset",
            split="train",
            streaming=True,
        )
    elif name == "fineweb_edu":
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=args.fineweb_edu_name,
            split="train",
            streaming=True,
        )
    elif name == "fineweb_hq":
        ds = load_dataset(
            "epfml/FineWeb-HQ",
            split="train",
            streaming=True,
        )
    elif name == "fineweb2_hq":
        if not args.fineweb2_hq_config:
            configs = get_dataset_config_names("epfml/FineWeb2-HQ")
            logger.error("FineWeb2-HQ configs:")
            for cfg in configs:
                logger.error("  %s", cfg)
            sys.exit(1)
        ds = load_dataset(
            "epfml/FineWeb2-HQ",
            name=args.fineweb2_hq_config,
            split="train",
            streaming=True,
        )
    else:
        raise ValueError(f"unknown source {name}")
    return iter(ds)


def log_progress(logger, written, per_source, dupe_count, filtered_count):
    denom = max(1, dupe_count + written)
    dedupe_rate = dupe_count / float(denom)
    counts = ", ".join(f"{k}={per_source.get(k,0)}" for k in SOURCES)
    logger.info(
        "written=%d dedupe_rate=%.3f filtered=%d %s",
        written,
        dedupe_rate,
        filtered_count,
        counts,
    )


def main(args):
    logger, _ = setup_runtime("00_build_raw_from_hf", threads=args.threads, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    weights = parse_weights(args.weights)
    schedule = build_schedule(weights)

    active = {}
    for name in SOURCES:
        if weights.get(name, 0.0) > 0:
            active[name] = load_source(name, args, logger)

    if not active:
        raise ValueError("no active sources; check weights")

    seen = set()
    per_source = defaultdict(int)
    errors = defaultdict(int)
    dupe_count = 0
    filtered_count = 0
    written = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        while written < args.target_lines and active:
            for name in schedule:
                if written >= args.target_lines:
                    break
                if name not in active:
                    continue

                text = None
                while text is None:
                    try:
                        sample = next(active[name])
                    except StopIteration:
                        logger.info("source_exhausted=%s", name)
                        active.pop(name, None)
                        break
                    except Exception as exc:
                        errors[name] += 1
                        logger.warning("read_error source=%s error=%s", name, exc)
                        continue

                    raw = pick_text(sample)
                    if not raw:
                        continue
                    norm = normalize_text(raw)
                    if not norm:
                        continue

                    length = len(norm)
                    if length < args.min_chars or length > args.max_chars:
                        filtered_count += 1
                        continue
                    if len(SENT_END_RE.findall(norm)) < 2:
                        filtered_count += 1
                        continue
                    if alnum_ratio(norm) < args.alnum_ratio:
                        filtered_count += 1
                        continue

                    h = xxhash.xxh64(norm).intdigest()
                    if h in seen:
                        dupe_count += 1
                        continue
                    seen.add(h)
                    text = norm

                if text is None:
                    continue

                out_f.write(text + "\n")
                written += 1
                per_source[name] += 1

                if written % args.log_every == 0:
                    log_progress(logger, written, per_source, dupe_count, filtered_count)

    log_progress(logger, written, per_source, dupe_count, filtered_count)
    if errors:
        err_counts = ", ".join(f"{k}={v}" for k, v in errors.items() if v)
        if err_counts:
            logger.warning("read_errors: %s", err_counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw.txt")
    ap.add_argument("--target_lines", type=int, default=300000)
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--max_chars", type=int, default=4000)
    ap.add_argument(
        "--weights",
        default="comma=0.6,fineweb_edu=0.3,fineweb2_hq=0.1",
    )
    ap.add_argument("--fineweb_edu_name", default="sample-10BT")
    ap.add_argument("--fineweb2_hq_config", default=None)
    ap.add_argument("--alnum_ratio", type=float, default=0.6)
    ap.add_argument("--log_every", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    main(args)
