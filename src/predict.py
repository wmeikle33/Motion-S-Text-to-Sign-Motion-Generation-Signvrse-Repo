import argparse
import pandas as pd
from joblib import load

from .data import load_csv, save_csv


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/model.keras")
    return ap.parse_args()


def main():
    args = parse_args()

    model = load(args.model)
    X = load_csv(args.input)

    save_csv(out, args.output)
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
