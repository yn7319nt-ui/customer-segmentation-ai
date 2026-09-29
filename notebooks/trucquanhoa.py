import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# BƯỚC 1: ĐỌC DỮ LIỆU VÀ CHUẨN HÓA

df = pd.read_csv(r'D:\py\Mall_Customers_Cleaned.csv')

# Khởi tạo Scaler chuẩn hóa Z-score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# BƯỚC 2: NÂNG CẤP - TÍCH HỢP PCA (2D VÀ 3D)

# Khởi tạo PCA 2D (PC1, PC2)
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# Khởi tạo PCA 3D (PC1, PC2, PC3)
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# In tỷ lệ phương sai
print("THÔNG TIN PHƯƠNG SAI TÍCH LŨY")
print("PCA 2D - Tỷ lệ giữ lại:", pca_2d.explained_variance_ratio_)
print("PCA 2D - Tổng phương sai giải thích:", np.sum(pca_2d.explained_variance_ratio_) * 100, "%")
print("PCA 3D - Tỷ lệ giữ lại:", pca_3d.explained_variance_ratio_)
print("PCA 3D - Tổng phương sai giải thích:", np.sum(pca_3d.explained_variance_ratio_) * 100, "%\n")

# BƯỚC 3: PHÂN CỤM K-MEANS TRÊN KHÔNG GIANG PCA (K=4)

best_k = 4

# K-Means cho 2D
kmeans_2d = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_2d = kmeans_2d.fit_predict(X_pca_2d)

df_pca_2d = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'])
df_pca_2d['Cluster'] = labels_2d

# K-Means cho 3D
kmeans_3d = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_3d = kmeans_3d.fit_predict(X_pca_3d)

df_pca_3d = pd.DataFrame(X_pca_3d, columns=['PC1', 'PC2', 'PC3'])
df_pca_3d['Cluster'] = labels_3d

# Bảng màu hiển thị cho 4 cụm
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

# BƯỚC 4: TRỰC QUAN HÓA ĐỒNG THỜI 2D VÀ 3D

fig = plt.figure(figsize=(16, 7))

# --- 1. ĐỒ THỊ BÊN TRÁI: BIỂU ĐỒ PCA 2D ---
ax1 = fig.add_subplot(1, 2, 1)
sns.scatterplot(
    x='PC1', y='PC2', 
    hue='Cluster', style='Cluster',
    data=df_pca_2d, 
    palette=colors, 
    s=90, alpha=0.9,
    ax=ax1
)

# Vẽ Tâm cụm 2D
ax1.scatter(
    kmeans_2d.cluster_centers_[:, 0], 
    kmeans_2d.cluster_centers_[:, 1], 
    s=250, c='black', marker='X', 
    edgecolor='white', linewidth=2, 
    label='Tâm cụm (Centroids)'
)

var_2d = np.sum(pca_2d.explained_variance_ratio_) * 100
ax1.set_title(f'Nâng cấp 1: Phân cụm K-Means (K=4) trên PCA 2D\n(Giữ {var_2d:.1f}% thông tin)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Principal Component 1 (PC1) - 33.7%', fontsize=10)
ax1.set_ylabel('Principal Component 2 (PC2) - 26.2%', fontsize=10)
ax1.legend(title='Cụm Khách Hàng', loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.5)

# --- 2. ĐỒ THỊ BÊN PHẢI: BIỂU ĐỒ PCA 3D ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

for cluster in range(best_k):
    cluster_data = df_pca_3d[df_pca_3d['Cluster'] == cluster]
    ax2.scatter(
        cluster_data['PC1'], 
        cluster_data['PC2'], 
        cluster_data['PC3'], 
        c=colors[cluster], 
        label=f'Cụm {cluster}', 
        s=60, alpha=0.85
    )

# Vẽ Tâm cụm 3D
centroids_3d = kmeans_3d.cluster_centers_
ax2.scatter(
    centroids_3d[:, 0], 
    centroids_3d[:, 1], 
    centroids_3d[:, 2], 
    s=300, c='black', marker='X', 
    edgecolor='white', linewidth=2, 
    label='Tâm cụm (Centroids)'
)

var_3d = np.sum(pca_3d.explained_variance_ratio_) * 100
ax2.set_title(f'Nâng cấp 1: Phân cụm K-Means (K=4) trên PCA 3D\n(Giữ {var_3d:.1f}% thông tin)', fontsize=12, fontweight='bold')
ax2.set_xlabel('PC1 (33.7%)', fontsize=10)
ax2.set_ylabel('PC2 (26.2%)', fontsize=10)
ax2.set_zlabel('PC3 (23.3%)', fontsize=10)
ax2.legend(title='Cụm Khách Hàng', loc='upper left')

# Tự động canh chỉnh khoảng cách giữa 2 khung hình
plt.tight_layout()

# Tự động lưu file ảnh chất lượng cao để chèn vào báo cáo Word/Slide
plt.savefig(r'D:\py\PCA_2D_3D_Clustering.png', dpi=300, bbox_inches='tight')

# Hiển thị cửa sổ biểu đồ
plt.show()