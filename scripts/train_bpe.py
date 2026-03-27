r"""Train a BPE tokenizer on TinyStories and save vocab/merges to disk.

Windows PowerShell example:
python .\scripts\train_bpe.py `
    --input .\data\TinyStoriesV2-GPT4-train.txt `
    --vocab-out .\tokenizer\tinystories_bpe_vocab.pkl `
    --merges-out .\tokenizer\tinystories_bpe_merges.pkl `
    --vocab-size 10000 `
    --special-tokens "<|endoftext|>"
"""

import argparse
import pathlib
import pickle
import sys
import time

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from cs336_basics.tokenizer import train_bpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("--input", type=str, required=True, help="Training text file path")
    parser.add_argument("--vocab-out", type=str, required=True, help="Output path for vocab .pkl")
    parser.add_argument("--merges-out", type=str, required=True, help="Output path for merges .pkl")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Target vocabulary size (default: 10000)")
    parser.add_argument(
        "--special-tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help='Special tokens (default: "<|endoftext|>")',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    vocab_out_path = pathlib.Path(args.vocab_out)
    merges_out_path = pathlib.Path(args.merges_out)

    if not input_path.is_file():
        sys.exit(f"[ERROR] Training file not found: {input_path}")

    input_size_gib = input_path.stat().st_size / (1024**3)
    print(
        f"Training BPE  vocab_size={args.vocab_size}  input={input_path}  "
        f"size={input_size_gib:.2f} GiB"
    )
    print("Using streaming input counting to avoid loading the full corpus into memory.")

    t0 = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )
    elapsed = time.perf_counter() - t0

    print(f"Done in {elapsed:.1f}s  |  vocab={len(vocab)}  merges={len(merges)}")

    vocab_out_path.parent.mkdir(parents=True, exist_ok=True)
    merges_out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(vocab_out_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_out_path, "wb") as f:
        pickle.dump(merges, f)

    print(f"Saved -> {vocab_out_path}")
    print(f"       -> {merges_out_path}")

    longest = max(vocab.values(), key=len)
    print(f"Longest token: {longest!r}  (len={len(longest)})")


if __name__ == "__main__":
    main()
