import numpy as np

# Kinect V2 25-joint names for reference
KINECT_V2_JOINTS = [
    "SPINE_BASE","SPINE_MID","NECK","HEAD",
    "SHOULDER_LEFT","ELBOW_LEFT","WRIST_LEFT","HAND_LEFT","HANDTIP_LEFT","THUMB_LEFT",
    "SHOULDER_RIGHT","ELBOW_RIGHT","WRIST_RIGHT","HAND_RIGHT","HANDTIP_RIGHT","THUMB_RIGHT",
    "HIP_LEFT","KNEE_LEFT","ANKLE_LEFT","FOOT_LEFT",
    "HIP_RIGHT","KNEE_RIGHT","ANKLE_RIGHT","FOOT_RIGHT",
    "SPINE_SHOULDER"
]

# undirected edges (bones)
EDGES = [
    (0,1),(1,24),(24,2),(2,3),
    (24,4),(4,5),(5,6),(6,7),(7,8),(7,9),
    (24,10),(10,11),(11,12),(12,13),(13,14),(13,15),
    (0,16),(16,17),(17,18),(18,19),
    (0,20),(20,21),(21,22),(22,23)
]

def adjacency(n=25):
    A = np.zeros((n,n), dtype=np.float32)
    for i,j in EDGES:
        A[i,j] = 1.0
        A[j,i] = 1.0
    # add self-loops
    for i in range(n): A[i,i] = 1.0
    # symmetric normalized A_hat = D^{-1/2} A D^{-1/2}
    D = np.diag(1.0 / np.sqrt(A.sum(1) + 1e-8))
    A_hat = D @ A @ D
    return A_hat
