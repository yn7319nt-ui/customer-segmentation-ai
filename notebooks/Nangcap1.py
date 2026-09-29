import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

#BƯỚC 1: ĐỌC DỮ LIỆU CŨ VÀ CHUẨN HÓA
df = pd.read_csv(r'D:\py\Mall_Customers_Cleaned.csv')

# Khởi tạo Scaler chuẩn hóa Z-score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# BƯỚC 2: NÂNG CẤP - TÍCH HỢP THUẬT TOÁN PCA
# Khởi tạo PCA giảm từ 4 thuộc tính xuống 2 thành phần chính (PC1, PC2)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Lưu kết quả PCA vào DataFrame để theo dõi
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
print("Tỷ lệ phương sai giữ lại bởi (PC1, PC2):", pca.explained_variance_ratio_)
print("Tổng phương sai giải thích:", np.sum(pca.explained_variance_ratio_))

# BƯỚC 3: CHẠY K-MEANS TRÊN DỮ LIỆU ĐÃ GIẢM CHIỀU (PCA)

# Tìm số cụm K tối ưu
sil_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    sil_scores.append(score)
    print(f"K = {k} | Silhouette Score = {score:.4f}")

# Chọn K = 4 (cho Silhouette Score cao nhất = 0.4164)
best_k = 4
kmeans_pca = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_pca.fit_predict(X_pca)
df_pca['Cluster'] = df['Cluster']

# BƯỚC 4: TRỰC QUAN HÓA KẾT QUẢ NÂNG CẤP

plt.figure(figsize=(9, 6))
sns.scatterplot(
    x='PC1', y='PC2', 
    hue='Cluster', 
    palette='Set1', 
    data=df_pca, 
    s=80, 
    style='Cluster'
)

# Vẽ tâm cụm (Centroids)
plt.scatter(
    kmeans_pca.cluster_centers_[:, 0], 
    kmeans_pca.cluster_centers_[:, 1], 
    s=250, c='black', marker='X', label='Tâm cụm (Centroids)'
)

plt.title('Nâng cấp 1: Phân cụm K-Means (K=4) trên không gian PCA 2D', fontsize=13)
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()