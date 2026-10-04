"""Dự đoán cụm khách hàng. Cách dùng: python predict.py <tuổi> <thu nhập k$> <điểm chi tiêu>"""
import sys
import joblib
import pandas as pd

MODEL_PATH = "kmeans_pipeline.joblib"
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def main():
    if len(sys.argv) != 4:
        print("Cách dùng: python predict.py <tuổi> <thu nhập k$> <điểm chi tiêu>")
        sys.exit(1)

    values = [float(v) for v in sys.argv[1:4]]
    model = joblib.load(MODEL_PATH)
    cluster = int(model.predict(pd.DataFrame([values], columns=FEATURES))[0])
    print(f"Khách hàng (tuổi={values[0]:g}, thu nhập={values[1]:g}k$, chi tiêu={values[2]:g}) thuộc cụm {cluster}")


if __name__ == "__main__":
    main()