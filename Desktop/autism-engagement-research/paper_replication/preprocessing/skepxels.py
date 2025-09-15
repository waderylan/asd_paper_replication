import numpy as np
import cv2

# --- Skepxels utilities -------------------------------------------------------
# We tile 64 frames of 25x( x,y,z ) into a 5x5 "joint grid" per frame,
# then arrange the 64 tiny 5x5x3 tiles into an 8x8 grid (40x40x3) and resize to 224^2.

def _frame_to_grid(frame_xyz25):
    """
    frame_xyz25: (25,3) -> (5,5,3)
    Uses simple row-major mapping j -> (r,c) = divmod(j,5).
    If you have a custom joint layout, change mapping here.
    """
    img = np.zeros((5, 5, 3), dtype=np.float32)
    for j in range(25):
        r, c = divmod(j, 5)
        img[r, c, :] = frame_xyz25[j, :]
    return img

def _tile_64(frames_5x5x3):
    """
    frames_5x5x3: list of 64 arrays each (5,5,3) -> (40,40,3)
    """
    assert len(frames_5x5x3) == 64
    rows = []
    for r in range(8):
        row_tiles = frames_5x5x3[r*8:(r+1)*8]
        rows.append(np.concatenate(row_tiles, axis=1))  # concat along width
    big = np.concatenate(rows, axis=0)                  # concat along height -> (40,40,3)
    return big

def _normalize_per_channel(img):
    # Min-max per channel
    mn = img.min(axis=(0,1), keepdims=True)
    mx = img.max(axis=(0,1), keepdims=True)
    rng = np.maximum(mx - mn, 1e-6)
    return (img - mn) / rng

def make_skepxels(xyz64):
    """
    xyz64: (64,25,3) float32
    returns: (3,224,224) float32 (CHW)
    """
    tiles = [_frame_to_grid(xyz64[t]) for t in range(64)]
    big = _tile_64(tiles).astype(np.float32)           # (40,40,3)
    big = _normalize_per_channel(big)
    big = cv2.resize(big, (224, 224), interpolation=cv2.INTER_LINEAR)
    chw = np.transpose(big, (2,0,1)).astype(np.float32)
    return chw

# --- Public API expected by datasets/* ----------------------------------------

def build_skepxels(xyz64):
    """
    Compatibility shim expected by datasets.gait_fullbody / datasets.dream.
    xyz64: (64,25,3) -> (3,224,224)
    """
    return make_skepxels(xyz64)
