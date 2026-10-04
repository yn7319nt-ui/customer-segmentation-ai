# customer-segmentation-ai
# Phân cụm khách hàng bằng K-Means kết hợp PCA

## 1. Giới thiệu

Đề tài thực hiện phân cụm khách hàng dựa trên bộ dữ liệu Mall Customers bằng thuật toán K-Means.

Nhóm xây dựng mô hình K-Means ban đầu và thực hiện cải tiến bằng cách tích hợp phương pháp Phân tích thành phần chính (Principal Component Analysis - PCA). PCA được sử dụng nhằm giảm chiều dữ liệu và hỗ trợ trực quan hóa kết quả phân cụm.

Bên cạnh đó, nhóm tiến hành đánh giá và so sánh kết quả phân cụm trước và sau khi cải tiến.

---

## 2. Mục tiêu

- Tìm hiểu bài toán phân cụm khách hàng.
- Xây dựng mô hình phân cụm khách hàng bằng thuật toán K-Means.
- Đánh giá kết quả phân cụm bằng các chỉ số phù hợp.
- Tích hợp PCA vào quy trình phân cụm.
- Giảm chiều dữ liệu và hỗ trợ trực quan hóa kết quả bằng PCA.
- So sánh kết quả giữa mô hình K-Means ban đầu và mô hình sau khi cải tiến.

---

## 3. Phương pháp thực hiện

Quy trình thực hiện gồm các bước chính:

1. Chuẩn bị bộ dữ liệu Mall Customers.
2. Tiền xử lý và chuẩn hóa dữ liệu.
3. Xây dựng mô hình K-Means ban đầu.
4. Đánh giá kết quả phân cụm bằng Inertia và Silhouette Score.
5. Tích hợp PCA để giảm chiều dữ liệu.
6. Thực hiện K-Means trên không gian dữ liệu sau PCA.
7. Đánh giá kết quả phân cụm sau khi cải tiến.
8. Trực quan hóa và so sánh kết quả trước và sau khi cải tiến.

---

## 4. Công nghệ sử dụng

### Ngôn ngữ và môi trường

- Python
- Jupyter Notebook
- Visual Studio Code

### Thư viện

- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

### Thuật toán và phương pháp

- K-Means Clustering
- Principal Component Analysis (PCA)
- Elbow Method
- Silhouette Score

---

## 5. Cấu trúc repository

```text
customer-segmentation-ai/
│
├── data/
│   ├── Mall_Customers_Cleaned.csv
│   └── Mall_Customers_Scaled.csv
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── Nangcap1.py
│   ├── README.md
│   ├── importmatplotlib.py
│   └── trucquanhoa.py
│
├── references/
│   └── Tài liệu tham khảo
│
├── reports/
│   └── Báo cáo của đề tài
│
└── slides/
    └── Slide thuyết trình
```
## 6. Dữ liệu

Đề tài sử dụng bộ dữ liệu **Mall Customers** để thực hiện phân cụm khách hàng.

Dữ liệu sau khi được tiền xử lý và chuẩn hóa được lưu trong thư mục `data/`, gồm:

- `Mall_Customers_Cleaned.csv`: dữ liệu sau bước làm sạch và tiền xử lý.
- `Mall_Customers_Scaled.csv`: dữ liệu sau khi được chuẩn hóa.

Các file dữ liệu được sử dụng làm đầu vào cho quá trình xây dựng và đánh giá mô hình phân cụm.
## 7. Cài đặt và chạy chương trình

### 7.1. Yêu cầu môi trường

- Python 3.x
- Jupyter Notebook hoặc Visual Studio Code
- Git (nếu muốn tải repository từ GitHub)

### 7.2. Cài đặt thư viện

Mở Terminal trong Visual Studio Code hoặc Command Prompt và chạy lệnh:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Các thư viện được sử dụng trong đề tài:

- `pandas`: đọc và xử lý dữ liệu.
- `numpy`: xử lý dữ liệu số.
- `matplotlib`: trực quan hóa dữ liệu và kết quả.
- `seaborn`: hỗ trợ trực quan hóa dữ liệu.
- `scikit-learn`: sử dụng các phương pháp và thuật toán như StandardScaler, PCA, K-Means và Silhouette Score.

### 7.3. Phân tích và tiền xử lý dữ liệu

Mở file:

```text
notebooks/01_EDA_and_Preprocessing.ipynb
```

File này được sử dụng để thực hiện phân tích khám phá dữ liệu và các bước tiền xử lý dữ liệu.

Có thể mở file bằng Jupyter Notebook hoặc Visual Studio Code và chạy lần lượt các cell.

Dữ liệu sau khi xử lý được lưu trong thư mục:

```text
data/
```

gồm:

- `Mall_Customers_Cleaned.csv`: dữ liệu sau khi làm sạch và tiền xử lý.
- `Mall_Customers_Scaled.csv`: dữ liệu sau khi chuẩn hóa.

### 7.4. Chạy chương trình Nâng cấp 1

Mở file:

```text
notebooks/Nangcap1.py
```

Mở Terminal tại thư mục repository và chạy lệnh:

```bash
python notebooks/Nangcap1.py
```

Chương trình thực hiện các bước:

1. Đọc dữ liệu đã được tiền xử lý.
2. Chuẩn hóa dữ liệu bằng `StandardScaler`.
3. Giảm chiều dữ liệu bằng PCA.
4. Thực hiện phân cụm K-Means trên dữ liệu sau PCA.
5. Tính Silhouette Score với các giá trị K từ 2 đến 10.
6. Đánh giá kết quả phân cụm.
7. Trực quan hóa kết quả trên không gian PCA.

### 7.5. Chạy chương trình trực quan hóa

File trực quan hóa được lưu tại:

```text
notebooks/trucquanhhoa.py
```

Có thể chạy file bằng Visual Studio Code hoặc Terminal:

```bash
python notebooks/trucquanhhoa.py
```

File này được sử dụng để hỗ trợ trực quan hóa dữ liệu và kết quả phân tích của đề tài.

