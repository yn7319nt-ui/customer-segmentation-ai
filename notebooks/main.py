"""So sánh K-Means bản gốc (K chọn tay, không PCA) với bản nâng cấp (PCA + tự chọn K)."""
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from auto_k import find_optimal_k

DATA_PATH = "Mall_Customers_clean.csv"
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
RANDOM_STATE = 42
K_MANUAL = 5
PCA_COMPONENTS = 2
N_RUNS = 5


def fit_timed(X_fit, k, n_runs=N_RUNS):
    """Chạy K-Means n_runs lần, trả về mô hình và thời gian fit trung bình."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_fit)
        times.append(time.perf_counter() - start)
    return km, float(np.mean(times))


def better(old, new, higher_is_better):
    if np.isclose(old, new):
        return "Bằng nhau"
    new_wins = new > old if higher_is_better else new < old
    return "Bản nâng cấp" if new_wins else "Bản gốc"


def main():
    df = pd.read_csv(DATA_PATH)
    X = StandardScaler().fit_transform(df[FEATURES])
    X_pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE).fit_transform(X)

    # Bản gốc
    km_old, t_old = fit_timed(X, K_MANUAL)

    # Bản nâng cấp
    start = time.perf_counter()
    k_auto, _ = find_optimal_k(X_pca, verbose=False)
    km_new, t_new = fit_timed(X_pca, k_auto)
    t_pipeline = time.perf_counter() - start - t_new * (N_RUNS - 1)

    sil_old = silhouette_score(X, km_old.labels_)
    dbi_old = davies_bouldin_score(X, km_old.labels_)
    sil_new_own = silhouette_score(X_pca, km_new.labels_)
    dbi_new_own = davies_bouldin_score(X_pca, km_new.labels_)
    sil_new_same = silhouette_score(X, km_new.labels_)
    dbi_new_same = davies_bouldin_score(X, km_new.labels_)

    rows = [
        ("Số cụm K", K_MANUAL, k_auto, None),
        ("Silhouette - không gian riêng (cao = tốt)", sil_old, sil_new_own, True),
        ("Davies-Bouldin - không gian riêng (thấp = tốt)", dbi_old, dbi_new_own, False),
        ("Silhouette - cùng không gian gốc (cao = tốt)", sil_old, sil_new_same, True),
        ("Davies-Bouldin - cùng không gian gốc (thấp = tốt)", dbi_old, dbi_new_same, False),
        ("Thời gian fit K-Means (giây, thấp = tốt)", t_old, t_new, False),
    ]
    table = pd.DataFrame(
        [(name, round(o, 4), round(n, 4), "-" if hib is None else better(o, n, hib))
         for name, o, n, hib in rows],
        columns=["Chỉ số", "Bản gốc", "Bản nâng cấp", "Bên tốt hơn"],
    )

    print(table.to_string(index=False))
    print(f"\nThời gian cả quy trình nâng cấp (PCA đã tính trước + tìm K + fit): {t_pipeline:.4f} s")
    table.to_csv("bang_so_sanh.csv", index=False, encoding="utf-8-sig")
    print("Đã lưu bang_so_sanh.csv")


if __name__ == "__main__":
    main()