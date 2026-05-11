
def cos_sched(t):
    return torch.cos(t * math.pi / 2)

def uniform(shape, dev):
    return torch.rand(shape, device=dev)

def topk(logits, th=0.9, d=-1):
    k = max(1, int((1 - th) * logits.shape[d]))
    v, i = logits.topk(k, d)
    out = torch.full_like(logits, float('-inf'))
    out.scatter_(d, i, v)
    return out

def lens_mask(lens, ml):
    dev = lens.device
    return torch.arange(ml, device=dev).expand(len(lens), ml) < lens.unsqueeze(1)
    
class CLIPTextEncoder(nn.Module):
    """
    CLIP text encoder (frozen).
    Matches mogen API: encode_text() for global, encode_text_tokens() for per-token.
    Keeps a string-level cache for speed.
    """
    def __init__(self, version="ViT-B/32", device="cuda", freeze=True, max_cache=16000):
        super().__init__()
        self.model, _ = clip.load(version, device=device)
        self.model = self.model.float()
        self.device = device
        self.embed_dim = self.model.text_projection.shape[1]
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self._cache = {}
        self._max_cache = max_cache

    def _warm(self, texts):
        """Ensure all texts in the batch are cached."""
        miss = [i for i, t in enumerate(texts) if t not in self._cache]
        if not miss:
            return
        mt = [texts[i] for i in miss]
        tok = clip.tokenize(mt, truncate=True).to(self.device)
        with torch.no_grad():
            x = self.model.token_embedding(tok)
            x += self.model.positional_embedding
            x = x.permute(1, 0, 2)
            x = self.model.transformer(x).permute(1, 0, 2)
            x = self.model.ln_final(x)
            mask = (tok != 0).float()
            e = x[torch.arange(len(mt)), tok.argmax(-1)] @ self.model.text_projection
            e = e / e.norm(dim=-1, keepdim=True)
            for j, i in enumerate(miss):
                self._cache[texts[i]] = (x[j], mask[j], e[j])
        # Evict oldest entries but PROTECT the current batch
        if len(self._cache) > self._max_cache:
            protected = set(texts)
            to_remove = len(self._cache) - self._max_cache
            removed = 0
            for k in list(self._cache.keys()):
                if removed >= to_remove:
                    break
                if k not in protected:
                    del self._cache[k]
                    removed += 1

    def encode_text(self, texts):
        """Global embeddings (B, embed_dim) — like mogen."""
        self._warm(texts)
        return torch.stack([self._cache[t][2] for t in texts])

    def encode_text_tokens(self, texts):
        """Token-level embeddings (B, 77, embed_dim) + mask — like mogen."""
        self._warm(texts)
        embs = torch.stack([self._cache[t][0] for t in texts])
        masks = torch.stack([self._cache[t][1] for t in texts])
        return embs, masks

    def forward(self, texts, tokens=False):
        """Unified forward (kept for backward compat with mask_trans/res_trans)."""
        if tokens:
            return self.encode_text_tokens(texts)
        return self.encode_text(texts)


class TextProjector(nn.Module):
    def __init__(self, din, dout, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, dout), nn.GELU(),
            nn.Dropout(p), nn.Linear(dout, dout)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)
