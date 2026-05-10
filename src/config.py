from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    random_state: int = 42
    artifacts_dir: Path = Path("models")
    model_path: Path = artifacts_dir / "models/model.keras"
    pipeline_path: Path = artifacts_dir / "pipeline.joblib"

VAE_CONFIG = {
    'num_embeddings': 512,
    'latent_dim': 256,
    'num_quantizers': 6,
}

TRANSFORMER_CONFIG = {
    'latent_dim': 384,
    'ff_size': 1024,
    'num_layers': 8,
    'num_heads': 6,
    'dropout': 0.1,
    'cond_drop_prob': 0.2,         
    'max_token_len': 500,         
    'min_token_len': 6,             
    'text_source': 'both',

    'batch_size': 32,              
    'grad_accum': 4,                
    'lr': 2e-4,
    'weight_decay': 0.01,          
    'warmup_epochs': 10,          
    'epochs': 100,                 
    'full_mask_prob': 0.5,         
    'label_smoothing': 0.1,        
    'residual_start_epoch': 50,    
    'res_lr': 1e-4,
    'res_weight_decay': 0.01,
    'res_prob': 1.0,              
    'save_every': 50,
}
