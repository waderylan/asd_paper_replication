from dataclasses import dataclass

@dataclass
class Config:
    # General
    seed: int = 1337
    device: str = "cuda"

    # Data (edit these to your paths)
    gait_root: str = "npz_datasets/gait"   # expects xyz arrays per sample
    dream_root: str = "npz_datasets/dream"         # expects JSON/npz per sample

    # Common preprocessing
    n_joints: int = 25
    n_frames: int = 64

    # Training
    batch_size: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs_img_stream: int = 30
    epochs_skel_stream: int = 100  # for skeleton ablation or full joint training
    epochs_joint: int = 30         # joint training of two streams
    align_lambda: float = 0.5      # weight on Euclidean alignment loss

    # Model
    convnext_embed_dim: int = 512
    gcn_embed_dim: int = 512
    fused_embed_dim: int = 512

    # Task
    dataset_name: str = "gait"     # "gait" (ASD vs TD) or "dream" (NS/ASD/AUT)
