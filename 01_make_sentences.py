import argparse
from pathlib import Path

from utils import ensure_dir, setup_runtime, split_sentences, write_jsonl


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).strip()


def main(raw_path: str, out_dir: str, max_sents: int, min_chars: int):
    ensure_dir(out_dir)
    raw = Path(raw_path).read_text(encoding="utf-8", errors="ignore").splitlines()

    sentences = []
    sequences = []
    sid = 0
    for doc_id, doc in enumerate(raw):
        sents = [normalize_whitespace(s) for s in split_sentences(doc)]
        sents = [s for s in sents if len(s) >= min_chars]
        if max_sents:
            sents = sents[:max_sents]
        if len(sents) < 2:
            continue
        sids = []
        for sent_idx, text in enumerate(sents):
            sentences.append(
                {"sid": sid, "doc_id": doc_id, "sent_idx": sent_idx, "text": text}
            )
            sids.append(sid)
            sid += 1
        sequences.append({"doc_id": doc_id, "sids": sids})

    write_jsonl(str(Path(out_dir) / "sentences.jsonl"), sentences)
    write_jsonl(str(Path(out_dir) / "sequences.jsonl"), sequences)
    return len(sentences), len(sequences)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Path to data/raw.txt")
    ap.add_argument("--out_dir", default="data", help="Output directory")
    ap.add_argument("--max_sents", type=int, default=128, help="Max sentences per doc")
    ap.add_argument("--min_chars", type=int, default=8, help="Drop sentences shorter than this")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    logger, _ = setup_runtime("01_make_sentences", threads=args.threads)
    logger.info("loading raw=%s", args.raw)
    sent_count, seq_count = main(args.raw, args.out_dir, args.max_sents, args.min_chars)
    logger.info("sentences=%d sequences=%d out=%s", sent_count, seq_count, args.out_dir)
