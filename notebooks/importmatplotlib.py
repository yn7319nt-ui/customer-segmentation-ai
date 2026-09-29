import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1. Tạo dữ liệu giả lập (3 cụm dữ liệu)
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=300, centers=centers, cluster_std=0.5, random_state=0
)

# 2. Khởi tạo và huấn luyện mô hình K-Means với 3 cụm
cmo = KMeans(n_clusters=3, random_state=0).fit(X)
labels = cmo.labels_

# 3. Trực quan hóa kết quả bằng Matplotlib
plt.figure(figsize=(8, 6))

# Vẽ các điểm dữ liệu theo từng cụm
for k, col in zip(range(3), ["red", "green", "blue"]):
    my_members = labels == k
    plt.plot(
            X[my_members, 0],
            X[my_members, 1],
            marker="o",
            color=col,
            markersize=6,
            alpha=0.6,
            label=f"Cụm {k+1}",
        )

    # Vẽ tâm cụm (Medoid) lớn hơn và màu sắc tương ứng
    medoid = cmo.cluster_centers_[k]
    plt.plot(
        medoid[0],
        medoid[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=14,
    )

plt.title("Phân cụm K-Medoids trên dữ liệu mẫu")
plt.legend()
plt.grid(True)
plt.show()
