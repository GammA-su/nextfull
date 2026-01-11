import argparse
import sys
from pathlib import Path

from datasets import get_dataset_config_names

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import setup_runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset name")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    logger, _ = setup_runtime("00_list_hf_configs", threads=args.threads)

    try:
        configs = get_dataset_config_names(args.dataset)
    except Exception as exc:
        logger.error("error: %s", exc)
        sys.exit(1)

    for name in configs:
        print(name)


if __name__ == "__main__":
    main()
