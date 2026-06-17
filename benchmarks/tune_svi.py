import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from bayesian_sparse_gmm.model import BayesianSparseGMM
import time

def tune_svi():
    print("Loading Olivetti faces...")
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = faces.data
    X = StandardScaler().fit_transform(X)
    
    lambdas = [2.0, 3.0, 4.0, 5.0, 10.0]
    thetas = [0.01, 0.1, 0.5]
    
    for l0 in lambdas:
        for t in thetas:
            gmm = BayesianSparseGMM(
                K_max=40,
                optimizer="svi",
                epochs=50,
                batch_size=128,
                lambda_0=l0,
                lambda_1=0.5,
                alpha=1.0,
                theta=t,
                backend="cuda",
                verbose=0,
                random_state=42
            )
            t0 = time.time()
            gmm.fit(X)
            n_sel = len(gmm.selected_features_)
            print(f"lambda_0={l0}, theta={t} -> Kept {n_sel}/4096 ({(n_sel/4096)*100:.2f}%) in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    tune_svi()
