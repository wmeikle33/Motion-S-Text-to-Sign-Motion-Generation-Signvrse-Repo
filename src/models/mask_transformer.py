
class MaskTransformer(nn.Module):
    def __init__(self, num_tokens, code_dim, latent_dim=384, ff_size=1024, num_layers=8,
                 num_heads=6, dropout=0.1, clip_dim=512, clip_version="ViT-B/32",
                 cond_drop_prob=0.1, device="cuda", max_seq_len=600):
        super().__init__()
        self.nt = num_tokens
        self.cd = code_dim
        self.ld = latent_dim
        self.cdp = cond_drop_prob
        self.dev = device

        self.mid = num_tokens
        self.pid = num_tokens + 1

        self.te = nn.Embedding(num_tokens + 2, code_dim)
        self.inp = InputProcess(code_dim, latent_dim)
        self.out = OutputProcess(latent_dim, num_tokens)
        self.pos = PositionalEncoding(latent_dim, dropout, max_seq_len)
        self.tcontx = TextContextualizer(clip_dim, latent_dim, dropout=dropout)
        self.tenc = CLIPTextEncoder(clip_version, device, freeze=True)
        # self.tproj = TextProjector(clip_dim, latent_dim, dropout)

        self.blks = nn.ModuleList([CrossAttentionBlock(latent_dim, num_heads, ff_size, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(latent_dim)

        self.ns = cos_sched

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(0, 0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)

    def dropc(self, c, force=False):
        if force: return torch.zeros_like(c)
        if not self.training or self.cdp == 0: return c
        bs = c.shape[0]
        m = torch.bernoulli(torch.full((bs,), self.cdp, device=c.device))
        return c * (1 - m.view(bs, 1, 1))

    def enct(self, texts):
        te, tm = self.tenc(texts, tokens=True)   # te: (B, 77, 512), tm: (B, 77)
        cpad = (tm == 0)                          # True = padding position
        ce = self.tcontx(te, cpad)               # (B, 77, latent_dim) — contextually enriched
        return ce, cpad

    def fwd_trans(self, mids, ce, cpad, mpad):
        B, sl = mids.shape
        x = self.te(mids)
        x = self.inp(x)
        x = self.pos(x)
        text = ce.permute(1, 0, 2)

        for b in self.blks:
            x = b(x, text, mpad, cpad)

        x = self.norm(x)
        return self.out(x)

    def forward(self, motion_ids, texts, m_lens, full_mask_prob=0.5, label_smoothing=0.1):
        B, sl = motion_ids.shape
        dev = motion_ids.device

        valid = lens_mask(m_lens, sl)
        pad = ~valid
        motion_ids = torch.where(valid, motion_ids, self.pid)

        ce, cpad = self.enct(texts)
        ce = self.dropc(ce)

        t = uniform((B,), dev)
        mp = self.ns(t)
        nm = (sl * mp).round().clamp(min=1)

        full = torch.bernoulli(torch.full((B,), full_mask_prob, device=dev)).bool()
        nm = torch.where(full, m_lens.float(), nm)

        scores = torch.rand(B, sl, device=dev)
        scores = scores.masked_fill(~valid, 2.0)
        mask = scores.argsort(dim=1).argsort(dim=1) < nm[:, None]
        mask &= valid

        lbl = torch.where(mask, motion_ids, self.mid)

        xids = motion_ids.clone()
        r10 = torch.bernoulli(torch.full((B, sl), 0.1, device=dev)).bool() & mask
        xids[r10] = torch.randint(0, self.nt, (B, sl), device=dev)[r10]

        m80 = torch.bernoulli(torch.full((B, sl), 0.8, device=dev)).bool() & mask & ~r10
        xids[m80] = self.mid

        logits = self.fwd_trans(xids, ce, cpad, pad)
        logits = logits.permute(0, 2, 1)

        loss = F.cross_entropy(
            logits.reshape(-1, self.nt),
            lbl.reshape(-1),
            ignore_index=self.mid,
            label_smoothing=label_smoothing
        )

        pred = logits.argmax(-1)
        acc = ((pred == motion_ids) & mask).sum().float() / mask.sum().clamp(min=1)

        return loss, pred, acc.item()

    def forward_with_cfg(self, motion_ids, cond_emb, cond_padding_mask, motion_padding_mask, cond_scale=3.0):
        lc = self.fwd_trans(motion_ids, cond_emb, cond_padding_mask, motion_padding_mask)
        lu = self.fwd_trans(motion_ids, self.dropc(cond_emb, True), cond_padding_mask, motion_padding_mask)
        return lu + cond_scale * (lc - lu)

    @torch.no_grad()
    def generate(self, texts, m_lens, timesteps=10, cond_scale=4.0, temperature=1.0, topk_filter_thres=0.9):
        self.eval()
        B = len(texts)
        dev = next(self.parameters()).device
        ml = m_lens.max().item()

        valid = lens_mask(m_lens, ml)
        pad = ~valid

        ce, cpad = self.enct(texts)

        ids = torch.where(pad, self.pid, self.mid)
        conf = torch.where(pad, 1e5, 0.)

        for s in range(timesteps):
            t = torch.tensor(s / timesteps, device=dev)
            p = self.ns(t)
            nm = (p * m_lens.float()).round().clamp(min=1).long()

            ranks = conf.argsort(1).argsort(1)
            masked = ranks < nm.unsqueeze(1)
            ids[masked] = self.mid

            logits = self.forward_with_cfg(ids, ce, cpad, pad, cond_scale)
            logits = logits.permute(0, 2, 1)

            filt = topk(logits, topk_filter_thres)
            probs = F.softmax(filt / temperature, -1)
            samp = torch.multinomial(probs.view(-1, self.nt), 1).view(B, ml)

            ids = torch.where(masked & valid, samp, ids)

            pr = F.softmax(logits, -1)
            conf = pr.gather(2, samp.unsqueeze(-1)).squeeze(-1)
            conf[~masked] = 1e5

        ids[pad] = -1
        return ids

    def params_no_clip(self):
        return [p for n,p in self.named_parameters() if 'tenc' not in n]
        
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

