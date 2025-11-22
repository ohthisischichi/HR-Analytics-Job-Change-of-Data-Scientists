# 1. HR Analytics: Job Change Prediction
Dự án Khoa học Dữ liệu nhằm dự đoán khả năng thay đổi công việc của nhân sự ngành Data Science, được thực hiện hoàn toàn thủ công (from scratch) bằng thư viện **NumPy**.

# 2. Mục lục

1. [Tiêu đề và Mô tả ngắn gọn](#1-hr-analytics-job-change-prediction)
2. [Mục lục](#2-mục-lục)
3. [Giới thiệu](#3-giới-thiệu)
4. [Dataset](#4-dataset)
5. [Quy trình thực hiện (Methodology)](#5-quy-trình-thực-hiện-methodology)
6. [Hướng dẫn cài đặt & Sử dụng](#6-hướng-dẫn-cài-đặt--sử-dụng)
7. [Usage: Hướng dẫn cách chạy từng phần](#7-usage-hướng-dẫn-cách-chạy-từng-phần)
8. [Kết quả (Results)](#8-kết-quả-results)
9. [Project Structure: Giải thích chức năng từng file/folder](#9-project-structure-giải-thích-chức-năng-từng-filefolder)
10. [Thách thức & Giải pháp](#10-thách-thức--giải-pháp)
11. [Hướng phát triển tương lai](#11-hướng-phát-triển-tương-lai)
12. [Contributors](#12-contributors)
13. [License](#13-license)
---

# 3. Giới thiệu

**Mô tả bài toán:**
Dự án này nhằm mục đích dự đoán khả năng một ứng viên Khoa học Dữ liệu (Data Scientist) sẽ thay đổi công việc hay không, dựa trên các thông tin nhân khẩu học, học vấn và kinh nghiệm làm việc của họ. Đây là bài toán phân loại nhị phân (Binary Classification) với đầu ra là 0 (Không đổi việc) hoặc 1 (Có đổi việc).

**Động lực & Ứng dụng thực tế:**
Việc dự đoán sớm ý định nghỉ việc giúp bộ phận Nhân sự (HR) chủ động đưa ra các chính sách giữ chân nhân tài, giảm thiểu chi phí tuyển dụng mới và duy trì sự ổn định cho đội ngũ nhân sự chất lượng cao.

**Mục tiêu cụ thể:**
* Xây dựng mô hình dự đoán chính xác khả năng thay đổi công việc (Ưu tiên **Recall** để không bỏ sót nhân tài).
* Triển khai toàn bộ quy trình xử lý dữ liệu và thuật toán học máy (Logistic Regression, Random Forest) từ đầu (from scratch) **CHỈ sử dụng thư viện NumPy**.
* Phân tích các yếu tố ảnh hưởng chính đến quyết định nghỉ việc.

---

# 4. Dataset

* **Nguồn dữ liệu:** HR Analytics: Job Change of Data Scientists.
* **Đặc điểm:**
    * **Kích thước:** Tập train gồm ~19,000 dòng, tập test ~2,000 dòng.
    * **Features:** Gồm 13 đặc trưng bao gồm cả dữ liệu số (`city_development_index`, `training_hours`) và dữ liệu danh mục (`city`, `gender`, `relevent_experience`, `enrolled_university`, `education_level`, `major_discipline`, `experience`, `company_size`, `company_type`, `last_new_job`).
    * **Target:** Cột `target` (0 hoặc 1).
* **Vấn đề chính:**
    * **Mất cân bằng lớp (Class Imbalance):** Lớp 0 chiếm ~75%, lớp 1 chỉ ~25%.
    * **Dữ liệu thiếu (Missing Values):** Đặc biệt nghiêm trọng ở các cột `company_type`, `gender`.
    * **Nhiễu (Outliers):** Cột `training_hours` có phân phối lệch phải với nhiều giá trị ngoại lai.

---

# 5. Quy trình thực hiện (Methodology)

Dự án tuân thủ quy trình Khoa học Dữ liệu tiêu chuẩn:

### Bước 1: Khám phá dữ liệu (EDA)
* Phân tích phân bố của biến mục tiêu để xác định chiến lược đánh giá (ưu tiên Recall/F1-Score).
* Định lượng và trực quan hóa tỷ lệ giá trị thiếu.
* Phân tích đơn biến (Histogram, Boxplot) và đa biến (Correlation Heatmap) để tìm mối tương quan (ví dụ: City Development Index càng thấp, tỷ lệ nghỉ việc càng cao).
* Phân tích hành vi chuyên sâu: "Hội chứng Job Hopping", "Chảy máu chất xám" và "Nghịch lý đào tạo".

### Bước 2: Tiền xử lý dữ liệu (Preprocessing)
* **Data Integration:** Tải và đồng bộ hóa quy trình xử lý cho cả tập train và test để đảm bảo tính nhất quán.
* **Data Cleaning:**
    * *Imputation:* Điền giá trị thiếu bằng **Median** (cho cột số) và **'Unknown'** (cho cột danh mục).
    * *Capping:* Xử lý ngoại lai (Outliers) bằng kỹ thuật **Winsorizing** (cắt ngưỡng 1% - 99%) để giảm nhiễu cho mô hình tuyến tính.
* **Data Transformation:**
    * *Encoding:* Sử dụng **Ordinal Encoding** cho biến có thứ tự (`experience`, `company_size`) và **One-Hot Encoding** cho biến định danh.
    * *Standardization:* Chuẩn hóa dữ liệu số về dạng Z-score: $z = \frac{x - \mu}{\sigma}$.
    * *Binning:* Chuyển đổi biến liên tục `training_hours` thành các nhóm rời rạc để xử lý mối quan hệ phi tuyến tính.
* **Data Reduction:** Loại bỏ cột định danh `enrollee_id` và kiểm tra loại bỏ các đặc trưng gây đa cộng tuyến.
* **Handling Imbalance:** Tự cài đặt thuật toán **SMOTE (Synthetic Minority Over-sampling Technique)** bằng NumPy (Vectorized) để cân bằng lại tập dữ liệu huấn luyện.

### Bước 3: Xây dựng Mô hình (Modeling)
* **Thuật toán:**
    * **Logistic Regression:** Tự cài đặt từ đầu bằng NumPy (Gradient Descent, Sigmoid).
    * **Random Forest:** Tự cài đặt từ đầu bằng NumPy (Bagging, Decision Trees).
* **Kỹ thuật tối ưu:**
    * **Threshold Tuning:** Tối ưu hóa ngưỡng quyết định để cân bằng Precision-Recall.
    * **Cross-Validation:** Sử dụng K-Fold để kiểm định độ ổn định của mô hình.
* **Hàm kích hoạt:** Sigmoid. Công thức: $\sigma(z) = \frac{1}{1 + e^{-z}}$.
* **Hàm mất mát:** Binary Cross-Entropy Loss.
* **Tối ưu hóa:** Gradient Descent. Công thức cập nhật: $\theta_j := \theta_j - \alpha \frac{\partial J}{\partial \theta_j}$.

---

# 6. Hướng dẫn cài đặt & Sử dụng

**Yêu cầu:** Python 3.7+, NumPy, Matplotlib, Seaborn.

**Cài đặt:**
Cài đặt các thư viện phụ thuộc:
    ```bash
    pip install -r requirements.txt
    ```
# 7. Usage: Hướng dẫn cách chạy từng phần
**Sử dụng:**
Chạy các notebook theo thứ tự để tái lập kết quả:
1.  `01_data_exploration.ipynb`: Để hiểu dữ liệu và xem các biểu đồ phân tích.
2.  `02_preprocessing.ipynb`: Để tạo ra dữ liệu sạch (kết quả lưu vào `data/processed/`).
3.  `03_modeling.ipynb`: Để huấn luyện mô hình và tạo file `submission.csv`.

---

# 8. Kết quả (Results)

Sau khi thử nghiệm so sánh giữa Logistic Regression và Random Forest trên tập dữ liệu đã SMOTE, kết quả đánh giá trên tập Validation như sau:

| Metric | Logistic Regression | Random Forest | Đánh giá |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 71.7% | **77.5%** | RF chính xác hơn tổng thể. |
| **Precision** | 45.7% | **54.9%** | RF giảm thiểu báo động giả tốt hơn. |
| **Recall** | 70.5% | **73.3%** | RF phát hiện được nhiều người nghỉ việc hơn. |
| **F1-Score** | 0.554 | **0.624** | **RF vượt trội (+7%) về độ cân bằng.** |

**Chọn **Random Forest** làm mô hình cuối cùng để triển khai.
* **Lý do:** Random Forest xử lý tốt các mối quan hệ phi tuyến tính (như biến `training_hours` và `experience`), chống nhiễu tốt hơn và mang lại chỉ số F1 cao nhất.

---

# 9. Project Structure: Giải thích chức năng từng file/folder

project-name/
├── data/                       # Thư mục chứa dữ liệu
│   ├── raw/                    # Dữ liệu thô ban đầu (không được chỉnh sửa)
│   │   ├── aug_train.csv       # Tập huấn luyện gốc
│   │   └── aug_test.csv        # Tập kiểm tra gốc (không có nhãn)
│   ├── processed/              # Dữ liệu đã qua xử lý (sẵn sàng cho mô hình)
│   │   ├── train_processed.csv # Tập train đã làm sạch, mã hóa và chuẩn hóa
│   │   ├──test_processed.csv   # Tập test đã làm sạch, mã hóa và chuẩn hóa
│   └   └── submission.csv      # Kết quả dự đoán cuối cùng để nộp bài
│
├── notebooks/                  # Các Jupyter Notebook (chạy theo thứ tự)
│   ├── 01_data_exploration.ipynb  # Bước 1: Khám phá dữ liệu, phân tích đơn biến/đa biến
│   ├── 02_preprocessing.ipynb     # Bước 2: Xử lý missing, outliers, encoding, scaling
│   └── 03_modeling.ipynb          # Bước 3: Huấn luyện Logistic Regression & Tạo submission
│
├── src/                        # Mã nguồn Python (Core Logic)
│   ├── __init__.py             # Đánh dấu thư mục là Python package
│   ├── data_processing.py      # Các hàm xử lý dữ liệu thuần NumPy (Load, Impute, Encode)
│   ├── visualization.py        # Các hàm vẽ biểu đồ (Histogram, Boxplot, Heatmap)
│   └── models.py               # Cài đặt thuật toán Logistic Regression & KNN từ đầu
│
├── README.md                   # Tài liệu báo cáo chi tiết về dự án
└── requirements.txt            # Danh sách các thư viện cần cài đặt

---

# 10. Thách thức & Giải pháp
1.  **Mất cân bằng dữ liệu (Imbalance):**
    * *Vấn đề:* Mô hình có xu hướng dự đoán toàn bộ là lớp 0 (Không nghỉ), Accuracy cao nhưng Recall thấp.
    * *Giải pháp:* Tự cài đặt thuật toán **SMOTE (Vectorized)** để sinh dữ liệu giả lập cho lớp 1. Kết hợp điều chỉnh ngưỡng (Threshold Moving) cho Logistic Regression.

2.  **Quan hệ phi tuyến tính:**
    * *Vấn đề:* Biến `training_hours` không tuân theo quy luật tăng/giảm đều, làm giảm hiệu quả của Logistic Regression.
    * *Giải pháp:* Chuyển sang kỹ thuật **Binning** và sử dụng mô hình **Random Forest** (phi tuyến tính) để nắm bắt mẫu dữ liệu phức tạp.

3.  **Hiệu năng tính toán:**
    * *Vấn đề:* Cài đặt Random Forest và SMOTE bằng vòng lặp Python thuần rất chậm.
    * *Giải pháp:* Tối ưu hóa bằng **NumPy Broadcasting** và **Vectorization** (tính toán trên ma trận) để tăng tốc độ xử lý gấp nhiều lần.

---

# 11. Hướng phát triển tương lai

* Thử nghiệm các mô hình phi tuyến tính mạnh mẽ hơn như Gradient Boosting (XGBoost, LightGBM) để cải thiện độ chính xác hơn nữa.
* Áp dụng kỹ thuật Hyperparameter Tuning (Grid Search) để tìm bộ tham số tối ưu nhất cho Random Forest.
* Tạo thêm các đặc trưng mới (Feature Engineering) dựa trên kiến thức chuyên môn (ví dụ: tỷ lệ `experience` / `age`).

---

# 12. Contributors

* **Họ tên:** Phan Thị Phương Chi 
* **MSSV:** 23120025
* **Email:** chiphan2005@gmail.com

---
# 13. License
Dự án này được thực hiện cho mục đích giáo dục thuộc môn học CSC17104 - Lập trình Khoa học dữ liệu.