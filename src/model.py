class TokenDataset(Dataset):
    COMB = "Sentence: {sentence} Signs: {gloss}"

    def __init__(self, data, src='both', ml=80, minl=6, alll=True):
        self.src = src
        self.ml = ml
        self.alll = alll

        self.samples = []
        for sid, d in data.items():
            t = d['tokens']
            if len(t.shape) == 1:
                t = t[np.newaxis, :]
            nq, sl = t.shape
            if minl <= sl <= ml:
                self.samples.append({
                    'id': sid,
                    't': t,
                    's': d['sentence'],
                    'g': d['gloss'],
                    'l': sl,
                    'nq': nq
                })

        print(f"TokenDataset: {len(self.samples)} samples "
              f"(filtered {len(data)} -> len {minl}-{ml}), src='{src}'")

    def __len__(self):
        return len(self.samples)

    def _txt(self, item):
        s, g = item['s'], item['g']
        if self.src == 'sentence': return s
        if self.src == 'gloss': return g
        if self.src == 'both': return self.COMB.format(sentence=s, gloss=g)
        if self.src == 'random':
            return s if random.random() < 0.5 else g
        return s

    def __getitem__(self, i):
        item = self.samples[i]
        t = item['t'].copy()
        txt = self._txt(item)
        l = item['l']

        if t.shape[1] < self.ml:
            pad = np.zeros((t.shape[0], self.ml - t.shape[1]), dtype=t.dtype)
            t = np.concatenate([t, pad], 1)
        else:
            t = t[:, :self.ml]
            l = self.ml

        t = torch.from_numpy(t).long()

        return txt, t if self.alll else t[0], l


def collate(batch, alll=True):
    txts, ts, ls = zip(*batch)
    ls = torch.tensor(ls, dtype=torch.long)
    ts = torch.stack(ts)
    return list(txts), ts, ls


def train_epoch(
    base_m, loader, base_opt, base_sch, dev, ep,
    writer=None, res_m=None, res_opt=None, res_sch=None,
    vq=None, train_res=False, res_p=0.5, amp=True, base_scaler=None,res_scaler=None,
    accum=1, res_only=False, fmp=0.5, ls=0.1
):
    """
    Train one epoch.  Key defaults aligned with mogen:
      fmp  = 0.5   (full_mask_prob -- forces text dependence)
      ls   = 0.1   (label smoothing)
    """
    if not res_only:
        base_m.train()
    else:
        base_m.eval()

    if res_m is not None:
        res_m.train()

    tl = ta = rl = ra = 0.0
    nbat = nres = 0

    pbar = tqdm(loader, desc=f"Ep {ep}")

    for bi, (txts, toks, lens) in enumerate(pbar):
        toks = toks.to(dev, non_blocking=True)
        lens = lens.to(dev, non_blocking=True)

        if len(toks.shape) == 3:
            bt = toks[:, 0]
            layers = [toks[:, i] for i in range(toks.shape[1])]
        else:
            bt = toks
            layers = [toks]

        accum_step = (bi + 1) % accum != 0

        # ---- Base model ----
        blv = ba = 0.0
        if not res_only:
            with autocast('cuda', enabled=amp):
                bl, _, ba = base_m(bt, txts, lens, full_mask_prob=fmp, label_smoothing=ls)
                bl = bl / accum

            if amp and base_scaler:
                base_scaler.scale(bl).backward()
                if not accum_step:
                    base_scaler.unscale_(base_opt)
                    torch.nn.utils.clip_grad_norm_(base_m.params_no_clip(), 1.0)
                    base_scaler.step(base_opt)
                    base_scaler.update()
                    base_opt.zero_grad()
            else:
                bl.backward()
                if not accum_step:
                    torch.nn.utils.clip_grad_norm_(base_m.params_no_clip(), 1.0)
                    base_opt.step()
                    base_opt.zero_grad()

            blv = bl.item() * accum
            tl += blv
            ta += ba
            nbat += 1

        # ---- Residual model ----
        rlv = batch_ra = 0.0
        if train_res and res_m and vq and random.random() < res_p:
            nq = len(layers)
            li = random.randint(1, nq - 1)
            prev = layers[:li]
            targ = layers[li]

            with autocast('cuda', enabled=amp):
                batch_rl, _, batch_ra = res_m(prev, targ, li, txts, lens, vq)
                batch_rl = batch_rl / accum

            if amp and res_scaler:
                res_scaler.scale(batch_rl).backward()
                if not accum_step:
                    res_scaler.unscale_(res_opt)
                    torch.nn.utils.clip_grad_norm_(res_m.params_no_clip(), 1.0)
                    res_scaler.step(res_opt)
                    res_scaler.update()
                    res_opt.zero_grad()
            else:
                batch_rl.backward()
                if not accum_step:
                    torch.nn.utils.clip_grad_norm_(res_m.params_no_clip(), 1.0)
                    res_opt.step()
                    res_opt.zero_grad()

            rlv = batch_rl.item() * accum
            rl += rlv
            ra += batch_ra
            nres += 1

        # Progress bar
        post = {}
        if not res_only:
            post['l'] = f'{blv:.4f}'
            post['a'] = f'{ba:.4f}'
        if rlv > 0:
            post['rl'] = f'{rlv:.4f}'
            post['ra'] = f'{batch_ra:.4f}'
        if amp:
            post['amp'] = 'on'
        pbar.set_postfix(post)

    # Step schedulers (after epoch, like mogen)
    if base_sch: base_sch.step()
    if res_sch:  res_sch.step()

    # Epoch averages
    m = {}
    if nbat > 0:
        m['l'] = tl / nbat
        m['a'] = ta / nbat
    if nres > 0:
        m['rl'] = rl / nres
        m['ra'] = ra / nres

    # W&B logging
    # if wandb.run:
    #     wlog = {'epoch': ep}
    #     if nbat > 0:
    #         wlog.update({
    #             'train/loss': m['l'],
    #             'train/acc': m['a'],
    #             'train/lr': base_opt.param_groups[0]['lr'],
    #         })
    #     if nres > 0:
    #         wlog.update({
    #             'train/res_loss': m['rl'],
    #             'train/res_acc': m['ra'],
    #         })
    #     wandb.log(wlog)

    return m


def train_one_epoch(
    base_m, loader, base_opt, base_sch, dev, ep,
    writer=None, res_m=None, res_opt=None, res_sch=None,
    vq=None, train_res=False, res_p=0.5, amp=True, base_scaler=None,res_scaler=None,
    accum=1, res_only=False, fmp=0.5, ls=0.1
):
    """
    Train one epoch.  Key defaults aligned with mogen:
      fmp  = 0.5   (full_mask_prob -- forces text dependence)
      ls   = 0.1   (label smoothing)
    """
    if not res_only:
        base_m.train()
    else:
        base_m.eval()

    if res_m is not None:
        res_m.train()

    tl = ta = rl = ra = 0.0
    nbat = nres = 0

    pbar = tqdm(loader, desc=f"Ep {ep}")

    for bi, (txts, toks, lens) in enumerate(pbar):
        toks = toks.to(dev, non_blocking=True)
        lens = lens.to(dev, non_blocking=True)

        if len(toks.shape) == 3:
            bt = toks[:, 0]
            layers = [toks[:, i] for i in range(toks.shape[1])]
        else:
            bt = toks
            layers = [toks]

        accum_step = (bi + 1) % accum != 0

        # ---- Base model ----
        blv = ba = 0.0
        if not res_only:
            with autocast('cuda', enabled=amp):
                bl, _, ba = base_m(bt, txts, lens, full_mask_prob=fmp, label_smoothing=ls)
                bl = bl / accum

            if amp and base_scaler:
                base_scaler.scale(bl).backward()
                if not accum_step:
                    base_scaler.unscale_(base_opt)
                    torch.nn.utils.clip_grad_norm_(base_m.params_no_clip(), 1.0)
                    base_scaler.step(base_opt)
                    base_scaler.update()
                    base_opt.zero_grad()
            else:
                bl.backward()
                if not accum_step:
                    torch.nn.utils.clip_grad_norm_(base_m.params_no_clip(), 1.0)
                    base_opt.step()
                    base_opt.zero_grad()

            blv = bl.item() * accum
            tl += blv
            ta += ba
            nbat += 1

        # ---- Residual model ----
        rlv = batch_ra = 0.0
        if train_res and res_m and vq and random.random() < res_p:
            nq = len(layers)
            li = random.randint(1, nq - 1)
            prev = layers[:li]
            targ = layers[li]

            with autocast('cuda', enabled=amp):
                batch_rl, _, batch_ra = res_m(prev, targ, li, txts, lens, vq)
                batch_rl = batch_rl / accum

            if amp and res_scaler:
                res_scaler.scale(batch_rl).backward()
                if not accum_step:
                    res_scaler.unscale_(res_opt)
                    torch.nn.utils.clip_grad_norm_(res_m.params_no_clip(), 1.0)
                    res_scaler.step(res_opt)
                    res_scaler.update()
                    res_opt.zero_grad()
            else:
                batch_rl.backward()
                if not accum_step:
                    torch.nn.utils.clip_grad_norm_(res_m.params_no_clip(), 1.0)
                    res_opt.step()
                    res_opt.zero_grad()

            rlv = batch_rl.item() * accum
            rl += rlv
            ra += batch_ra
            nres += 1

        # Progress bar
        post = {}
        if not res_only:
            post['l'] = f'{blv:.4f}'
            post['a'] = f'{ba:.4f}'
        if rlv > 0:
            post['rl'] = f'{rlv:.4f}'
            post['ra'] = f'{batch_ra:.4f}'
        if amp:
            post['amp'] = 'on'
        pbar.set_postfix(post)

    # Step schedulers (after epoch, like mogen)
    if base_sch: base_sch.step()
    if res_sch:  res_sch.step()

    # Epoch averages
    m = {}
    if nbat > 0:
        m['l'] = tl / nbat
        m['a'] = ta / nbat
    if nres > 0:
        m['rl'] = rl / nres
        m['ra'] = ra / nres

    # W&B logging
    # if wandb.run:
    #     wlog = {'epoch': ep}
    #     if nbat > 0:
    #         wlog.update({
    #             'train/loss': m['l'],
    #             'train/acc': m['a'],
    #             'train/lr': base_opt.param_groups[0]['lr'],
    #         })
    #     if nres > 0:
    #         wlog.update({
    #             'train/res_loss': m['rl'],
    #             'train/res_acc': m['ra'],
    #         })
    #     wandb.log(wlog)

    return m

