class InputProcess(nn.Module):
    def __init__(self, cd, ld):
        super().__init__()
        self.e = nn.Linear(cd, ld)
    def forward(self, x):
        return self.e(x).permute(1, 0, 2)

class OutputProcess(nn.Module):
    def __init__(self, ld, nt):
        super().__init__()
        self.d = nn.Linear(ld, ld)
        self.n = nn.LayerNorm(ld)
        self.o = nn.Linear(ld, nt)
    def forward(self, x):
        x = self.d(x)
        x = F.gelu(x)
        x = self.n(x)
        return self.o(x).permute(1, 2, 0)
