PROJECT_ROOT = Path("/kaggle/working")
KAGGLE_INPUT = Path("/kaggle/input")
DATA_BASE = KAGGLE_INPUT / "/Users/wmeikle/Downloads/motion-s-hierarchical-text-to-motion-generation-for-sign-language"
CSV_PATH = DATA_BASE / "train.csv"
test_df = pd.read_csv('/Users/wmeikle/Downloads/motion-s-hierarchical-text-to-motion-generation-for-sign-language/test.csv')
VAE_PATH = KAGGLE_INPUT / "/kaggle/input/models/antonygithinji/motion-s-vae-rvq/pytorch/default/3/rvq_vae_best.pth"

OUTPUT_ROOT = PROJECT_ROOT

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Using device: {device}")
