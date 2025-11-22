# src/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Thiết lập theme cho Seaborn
sns.set_theme(style="whitegrid")

# --- 1. Trực quan hóa Khám phá Dữ liệu (EDA) ---

def plot_univariate_distribution(data, col_name, is_numerical=True):
    """
    Vẽ Histogram và Boxplot cho một biến duy nhất.
    """
    if not is_numerical:
        print("Chỉ hỗ trợ Histogram cho dữ liệu số hoặc Ordinal đã mã hóa.")
        return
        
    plt.figure(figsize=(12, 5))

    # Histogram (Phân bố)
    plt.subplot(1, 2, 1)
    sns.histplot(data, kde=True, bins=30, color='skyblue')
    plt.title(f'Histogram Phân bố: {col_name}')
    plt.xlabel(col_name)
    
    # Boxplot (Giá trị ngoại lai - Outliers)
    plt.subplot(1, 2, 2)
    sns.boxplot(y=data, color='lightcoral')
    plt.title(f'Boxplot: {col_name} (Kiểm tra Outliers)')
    plt.ylabel(col_name)

    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(X, feature_names=None, title="Ma trận Tương quan Features"):
    """
    Tính toán và trực quan hóa ma trận tương quan (Correlation Heatmap) 
    CHỈ DÙNG NUMPY để tính toán Correlation Matrix.
    """
    # X là ma trận features 2D (ví dụ: X_train_final)
    
    # 1. Tính toán Ma trận Tương quan (CHỈ DÙNG NUMPY)
    # np.corrcoef(X, rowvar=False) tính toán hệ số tương quan Pearson 
    # giữa các cột (features).
    try:
        corr_matrix = np.corrcoef(X, rowvar=False)
    except Exception as e:
        print(f"LỖI: Không thể tính np.corrcoef. Đảm bảo X chỉ chứa giá trị số. Chi tiết: {e}")
        return

    # 2. Trực quan hóa Heatmap
    plt.figure(figsize=(10, 8))
    
    # Tạo mask để che phần tam giác trên (tùy chọn)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) 
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,          # Hiển thị giá trị tương quan
        fmt=".2f",
        cmap='coolwarm',     # Sắc độ màu
        cbar_kws={"shrink": .8},
        linewidths=.5,
        linecolor='black',
        xticklabels=feature_names if feature_names else 'auto',
        yticklabels=feature_names if feature_names else 'auto'
    )
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_target_distribution(targets, labels, title="Phân bố Biến Mục tiêu"):
    """
    Vẽ biểu đồ cột (Bar plot) thể hiện sự mất cân bằng lớp (Class Imbalance).
    """
    counts = [np.sum(targets == 0.0), np.sum(targets == 1.0)]
    
    plt.figure(figsize=(7, 5))
    sns.barplot(x=labels, y=counts, palette='viridis')
    plt.title(title)
    plt.xlabel('Lớp (0: Không Thay đổi / 1: Có Thay đổi)')
    plt.ylabel('Số lượng Mẫu')
    plt.show()

def plot_missing_values(missing_percentages):
    """
    Vẽ biểu đồ thanh ngang (Barh) thể hiện tỷ lệ giá trị thiếu theo cột.
    """
    if not missing_percentages:
        print("Không có giá trị thiếu để trực quan hóa.")
        return
        
    # Sắp xếp theo tỷ lệ giảm dần
    sorted_missing = sorted(missing_percentages.items(), key=lambda item: item[1], reverse=True)
    cols = [item[0] for item in sorted_missing]
    percentages = [item[1] for item in sorted_missing]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=percentages, y=cols, palette='Reds_d')
    plt.title('Tỷ lệ Giá trị Thiếu trên các Đặc trưng')
    plt.xlabel('Phần trăm Giá trị Thiếu (%)')
    plt.show()

def plot_raw_vs_standardized(raw_data, processed_data, raw_name, std_name):
    """
    Vẽ biểu đồ so sánh phân bố dữ liệu thô và dữ liệu đã chuẩn hóa.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'So sánh Phân bố: {raw_name} (Thô) vs {std_name} (Chuẩn hóa)', fontsize=14)

    # Plot 1: Dữ liệu Thô
    sns.histplot(raw_data, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f'Thô (Mean: {np.mean(raw_data):.2f})')
    axes[0].set_xlabel('Giá trị Gốc')
    
    # Plot 2: Dữ liệu Đã Chuẩn hóa
    sns.histplot(processed_data, kde=True, ax=axes[1], color='lightcoral')
    axes[1].set_title(f'Chuẩn hóa (Mean: {np.mean(processed_data):.2f})')
    axes[1].set_xlabel('Giá trị Z-score')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- 2. Trực quan hóa Mô hình ---

def plot_loss_history(loss_history, title="Lịch sử Loss trong quá trình Huấn luyện"):
    """
    Vẽ biểu đồ đường (Line Chart) thể hiện sự thay đổi của hàm mất mát.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel('Vòng lặp (x100)')
    plt.ylabel('Loss (Cross-Entropy)')
    plt.show()

def plot_confusion_matrix(metrics, title="Confusion Matrix"):
    """
    Vẽ Heatmap Confusion Matrix từ các chỉ số TP, TN, FP, FN.
    """
    cm = np.array([
        [metrics['TN'], metrics['FP']],
        [metrics['FN'], metrics['TP']]
    ])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Dự đoán 0', 'Dự đoán 1'], 
                yticklabels=['Thực tế 0', 'Thực tế 1'])
    plt.title(title)
    plt.ylabel('Thực tế (True Label)')
    plt.xlabel('Dự đoán (Predicted Label)')
    plt.show()

    # Thêm hàm này vào cuối file src/visualization.py

def plot_correlation_heatmap(X, feature_names=None, title="Ma trận Tương quan"):
    """
    Tính toán và trực quan hóa ma trận tương quan (Correlation Heatmap).
    """
    # 1. Tính toán Ma trận Tương quan (Pearson) CHỈ DÙNG NUMPY
    # rowvar=False: Mỗi cột là một biến (feature), mỗi hàng là một mẫu (sample)
    try:
        corr_matrix = np.corrcoef(X, rowvar=False)
    except Exception as e:
        print(f"LỖI: Không thể tính toán ma trận tương quan. Chi tiết: {e}")
        return

    # 2. Vẽ Heatmap bằng Seaborn
    plt.figure(figsize=(10, 8))
    
    # Tạo mask để che phần tam giác trên (để biểu đồ đỡ rối)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) 
    
    sns.heatmap(
        corr_matrix,
        mask=mask,           # Che một nửa trên
        annot=True,          # Hiển thị con số tương quan
        fmt=".2f",           # Làm tròn 2 chữ số thập phân
        cmap='coolwarm',     # Màu: Đỏ (Dương), Xanh (Âm)
        vmin=-1, vmax=1,     # Giới hạn thang màu từ -1 đến 1
        cbar_kws={"shrink": .8},
        linewidths=.5,
        linecolor='white',
        xticklabels=feature_names if feature_names else 'auto',
        yticklabels=feature_names if feature_names else 'auto'
    )
    
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right') # Xoay tên cột cho dễ đọc
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_raw_vs_capping(data_raw, data_capped, col_name):
    """
    Vẽ biểu đồ Histogram so sánh phân bố dữ liệu trước và sau khi Capping.
    
    Args:
        data_raw (array-like): Dữ liệu gốc (có thể chứa outliers).
        data_capped (array-like): Dữ liệu sau khi đã xử lý Capping.
        col_name (str): Tên cột để hiển thị.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'HIỆU QUẢ CAPPING (WINSORIZING): {col_name}', fontsize=14, fontweight='bold')
    
    # --- Plot 1: Trước khi Capping ---
    # Loại bỏ NaN để vẽ không lỗi
    valid_raw = data_raw[~np.isnan(data_raw)]
    sns.histplot(valid_raw, kde=True, ax=axes[0], color='salmon', bins=40)
    axes[0].set_title(f'TRƯỚC Capping\n(Max: {np.max(valid_raw):.2f})', color='red')
    axes[0].set_xlabel('Giá trị')
    
    # --- Plot 2: Sau khi Capping ---
    valid_capped = data_capped[~np.isnan(data_capped)]
    sns.histplot(valid_capped, kde=True, ax=axes[1], color='teal', bins=40)
    axes[1].set_title(f'SAU Capping\n(Max: {np.max(valid_capped):.2f})', color='green')
    axes[1].set_xlabel('Giá trị')
    
    plt.tight_layout()
    plt.show()