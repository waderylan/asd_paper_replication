from sklearn.model_selection import KFold
import numpy as np

def make_kfold_indices(n_samples, n_splits=10, seed=1337):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    idx = np.arange(n_samples)
    for tr, te in kf.split(idx):
        folds.append({"train": tr.tolist(), "test": te.tolist()})
    return folds
