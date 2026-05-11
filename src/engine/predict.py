
@torch.no_grad()
def validate(model, loader, dev, ep, writer=None):
    """Validate base model (token-level CE + accuracy)."""
    model.eval()

    tl = ta = 0.0
    nbat = 0

    for bi, (txts, toks, lens) in enumerate(tqdm(loader, desc="Val")):
        toks = toks.to(dev)
        lens = lens.to(dev)

        bt = toks[:, 0] if len(toks.shape) == 3 else toks

        l, _, a = model(bt, txts, lens)
        tl += l.item()
        ta += a
        nbat += 1

    out = {
        'l': tl / nbat if nbat > 0 else 0,
        'a': ta / nbat if nbat > 0 else 0,
    }

    # if wandb.run:
    #     wandb.log({'epoch': ep, 'val/loss': out['l'], 'val/acc': out['a']})

    return out
