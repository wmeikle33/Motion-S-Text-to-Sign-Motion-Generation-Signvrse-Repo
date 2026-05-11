
from .data import load_csv
from .model import train_eval_save
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader

from .model import train_one_epoch, evaluate



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-out", default="models/model.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)

    return ap.parse_args()


def main():
    args = parse_args()

    
    train_loader = DataLoader(
        train_ds, batch_size=C['batch_size'], shuffle=True,
        collate_fn=collate, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=C['batch_size'], shuffle=False,
        collate_fn=collate, pin_memory=True
    )


    for epoch in range(args.epochs):
  
    
if __name__ == "__main__":
    main()
