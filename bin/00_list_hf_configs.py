import argparse
import sys

from datasets import get_dataset_config_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset name")
    args = ap.parse_args()

    try:
        configs = get_dataset_config_names(args.dataset)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    for name in configs:
        print(name)


if __name__ == "__main__":
    main()
