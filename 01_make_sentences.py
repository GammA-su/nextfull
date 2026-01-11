import argparse
from pathlib import Path

from utils import ensure_dir, split_sentences, write_jsonl


def main(raw_path: str, out_dir: str, max_sents: int):
    ensure_dir(out_dir)
    raw = Path(raw_path).read_text(encoding="utf-8", errors="ignore").splitlines()

    sentences = []
    sequences = []
    sid = 0
    for doc_id, doc in enumerate(raw):
        sents = split_sentences(doc)
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
    print(f"sentences={len(sentences)} sequences={len(sequences)} out={out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Path to data/raw.txt")
    ap.add_argument("--out_dir", default="data", help="Output directory")
    ap.add_argument("--max_sents", type=int, default=128, help="Max sentences per doc")
    args = ap.parse_args()
    main(args.raw, args.out_dir, args.max_sents)
