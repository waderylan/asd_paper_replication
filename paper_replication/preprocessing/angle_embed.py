import numpy as np

# Angle embedding per the paper: compute cosine-similarity (angle) matrix between joints
# for each frame, and project coordinates through that matrix to enrich features.

def angle_embed(xyz64):
    """
    xyz64: (64,25,3) float32
    returns angle-enriched coords as (3,64,25) float32 (CHW for graph stream)
    """
    # Normalize across sequence to stabilize magnitudes
    norm = np.linalg.norm(xyz64.reshape(-1, 3), axis=1).mean() + 1e-6
    X = xyz64 / norm  # (64,25,3)
    T, J, _ = X.shape

    # angles[t] = cosine-sim matrix (25x25) at frame t
    angles = np.zeros((T, J, J), dtype=np.float32)
    for t in range(T):
        v = X[t]                      # (25,3)
        dot = v @ v.T                 # (25,25)
        nv = np.linalg.norm(v, axis=1, keepdims=True)
        denom = (nv @ nv.T) + 1e-6
        angles[t] = (dot / denom).astype(np.float32)

    # Inject: (T,25,25) @ (T,25,3) -> (T,25,3)
    X_theta = np.einsum("tij,tjk->tik", angles, X)
    # Graph stream expects (C,T,V) = (3,64,25)
    return np.transpose(X_theta.astype(np.float32), (2,0,1))

# --- Public API expected by datasets/* ----------------------------------------

def embed_with_angles(xyz64):
    """
    Compatibility shim expected by datasets.gait_fullbody / datasets.dream.
    xyz64: (64,25,3) -> (3,64,25)
    """
    return angle_embed(xyz64)
