import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator


def find_optimal_k(X, k_min=2, k_max=10, random_state=42, verbose=True):
    """
    Tự động tìm số cụm K tối ưu cho K-Means bằng Elbow + Silhouette.

    Quy tắc chốt K:
      1. Elbow thất bại (None) -> dùng K của Silhouette.
      2. Hai phương pháp cùng K -> chọn K đó.
      3. Khác nhau -> chọn K có Silhouette cao hơn; nếu chênh lệch < 0.02
         thì chọn K nhỏ hơn.

    Trả về: best_k (int), info (dict gồm K_range, inertias, silhouettes, k_elbow, k_sil)
    """
    K_range = list(range(k_min, k_max + 1))
    inertias, sil_scores = [], []

    for k in K_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, km.labels_))
        if verbose:
            print(f"K={k}: inertia={inertias[-1]:.1f}, silhouette={sil_scores[-1]:.3f}")

    k_elbow = KneeLocator(K_range, inertias,
                          curve="convex", direction="decreasing").elbow
    k_sil = K_range[int(np.argmax(sil_scores))]

    if k_elbow is None:
        best_k = k_sil
    elif k_elbow == k_sil:
        best_k = k_elbow
    else:
        sil_elbow = sil_scores[K_range.index(k_elbow)]
        if max(sil_scores) - sil_elbow < 0.02:
            best_k = min(k_elbow, k_sil)
        else:
            best_k = k_sil

    info = {"K_range": K_range, "inertias": inertias, "silhouettes": sil_scores,
            "k_elbow": k_elbow, "k_sil": k_sil}
    return int(best_k), info


def plot_k_selection(info, best_k, save_path="k_selection.png"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(info["K_range"], info["inertias"], "o-")
    ax[0].axvline(best_k, color="red", linestyle="--", label=f"K chọn = {best_k}")
    ax[0].set_title("Elbow Method")
    ax[0].set_xlabel("Số cụm K")
    ax[0].set_ylabel("Inertia")
    ax[0].legend()

    ax[1].plot(info["K_range"], info["silhouettes"], "o-", color="green")
    ax[1].axvline(best_k, color="red", linestyle="--")
    ax[1].set_title("Silhouette Score")
    ax[1].set_xlabel("Số cụm K")
    ax[1].set_ylabel("Silhouette")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
