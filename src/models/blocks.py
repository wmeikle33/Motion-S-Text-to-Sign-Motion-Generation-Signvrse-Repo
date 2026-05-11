class CrossAttentionBlock(nn.Module):
    def __init__(self, d, h, ff=2048, p=0.1):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, h, dropout=p, batch_first=False)
        self.sa_n = nn.LayerNorm(d)
        self.sa_p = nn.Dropout(p)

        self.ca = nn.MultiheadAttention(d, h, dropout=p, batch_first=False)
        self.ca_n = nn.LayerNorm(d)
        self.ca_p = nn.Dropout(p)

        self.ff = nn.Sequential(
            nn.Linear(d, ff), nn.GELU(),
            nn.Dropout(p), nn.Linear(ff, d),
            nn.Dropout(p)
        )
        self.ff_n = nn.LayerNorm(d)

    def forward(self, x, c, xpad=None, cpad=None):
        r = x; xn = self.sa_n(x)
        out, _ = self.sa(xn, xn, xn, key_padding_mask=xpad)
        x = r + self.sa_p(out)

        r = x; xn = self.ca_n(x)
        out, _ = self.ca(xn, c, c, key_padding_mask=cpad)
        x = r + self.ca_p(out)

        r = x; xn = self.ff_n(x)
        x = r + self.ff(xn)
        return x
