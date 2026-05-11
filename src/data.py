PROJECT_ROOT = Path("/kaggle/working")
KAGGLE_INPUT = Path("/kaggle/input")
DATA_BASE = KAGGLE_INPUT / "/Users/wmeikle/Downloads/motion-s-hierarchical-text-to-motion-generation-for-sign-language"
CSV_PATH = DATA_BASE / "train.csv"
test_df = pd.read_csv('/Users/wmeikle/Downloads/motion-s-hierarchical-text-to-motion-generation-for-sign-language/test.csv')
VAE_PATH = KAGGLE_INPUT / "/kaggle/input/models/antonygithinji/motion-s-vae-rvq/pytorch/default/3/rvq_vae_best.pth"

OUTPUT_ROOT = PROJECT_ROOT

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Using device: {device}")

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
