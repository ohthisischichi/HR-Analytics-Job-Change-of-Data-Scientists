# src/data_processing.py

import numpy as np
import csv
import sys
# Tăng giới hạn đệ quy nếu cần thiết cho các thao tác mảng lớn
sys.setrecursionlimit(2000) 

# --- 1. Load Dữ liệu ---
def load_data(file_path):
    """
    Load dữ liệu từ file CSV thành NumPy Structured Array.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
        
        # Load dữ liệu chính: dtype=None cho Structured Array
        data = np.genfromtxt(
            file_path,
            delimiter=',',
            skip_header=1,
            dtype=None,
            encoding='utf-8',
            usecols=range(len(header))
        )
        
        field_map = {name: data.dtype.names[i] for i, name in enumerate(header)}
        return data, header, field_map
    except Exception as e:
        print(f"LỖI LOAD DATA: Không thể tải {file_path}. Chi tiết: {e}")
        return None, None, None


# --- 2. Xử lý Missing Values (Imputation) ---
def impute_categorical(data, field_map, col_name, fill_value='Unknown'):
    """
    Điền giá trị thiếu (chuỗi rỗng) trong cột danh mục bằng fill_value.
    Sử dụng Indexing/Masking CHỈ DÙNG NUMPY.
    """
    field_name = field_map[col_name]
    column = data[field_name]
    
    # Tạo Mask: True cho vị trí thiếu ('')
    missing_mask = (column == '')
    
    # Điền giá trị mới vào vị trí bị thiếu
    data[field_name][missing_mask] = fill_value
    return data

def impute_numerical(data, field_map, col_name, median_value):
    """
    Điền giá trị thiếu (np.nan) trong cột số bằng giá trị median đã tính từ tập train.
    Sử dụng Indexing/Masking CHỈ DÙNG NUMPY.
    """
    field_name = field_map[col_name]
    column = data[field_name]
    
    # Tạo Mask: True cho vị trí np.nan
    missing_mask = np.isnan(column)
    
    # Điền giá trị mới vào vị trí bị thiếu
    data[field_name][missing_mask] = median_value
    return data


def cap_outliers(data, field_map, col_name, lower_percentile=0.01, upper_percentile=0.99):
    """
    Xử lý ngoại lai bằng phương pháp Capping (Winsorizing).
    Thay thế các giá trị nằm ngoài khoảng [1%, 99%] bằng giá trị biên tương ứng.
    """
    field = field_map[col_name]
    column = data[field]
    
    # Tính các ngưỡng biên (CHỈ DÙNG NUMPY)
    # nanquantile bỏ qua NaN để tính cho chính xác
    lower_bound = np.nanquantile(column, lower_percentile)
    upper_bound = np.nanquantile(column, upper_percentile)
    
    print(f"Capping '{col_name}': < {lower_bound:.2f} -> {lower_bound:.2f} | > {upper_bound:.2f} -> {upper_bound:.2f}")
    
    # Capping (Vectorized)
    # Dùng np.clip để giới hạn giá trị trong khoảng [lower, upper]
    # Lưu ý: np.clip không thay đổi giá trị NaN
    data[field] = np.clip(column, lower_bound, upper_bound)
    
    return data

# --- 3. Mã hóa (Encoding) ---
def ordinal_encode(data, field_name, mapping):
    """
    Mã hóa cột Ordinal (thứ tự) thành số.
    Sử dụng np.select để vector hóa việc mapping.
    """
    column = data[field_name]
    
    conditions = [(column == key) for key in mapping.keys()]
    choices = list(mapping.values())
    
    # np.select để vector hóa mapping. default=-1 cho giá trị không map được
    encoded_array = np.select(conditions, choices, default=-1) 
    return encoded_array

def one_hot_encode(data, field_map, col_name):
    """
    Tự triển khai One-Hot Encoding cho cột danh mục (Nominal).
    Sử dụng vectorization CHỈ DÙNG NUMPY.
    """
    field = field_map[col_name]
    column = data[field]
    
    unique_values = np.unique(column)
    n_samples = data.shape[0]
    n_unique = len(unique_values)
    
    ohe_matrix = np.zeros((n_samples, n_unique), dtype=np.int8)
    
    for i, val in enumerate(unique_values):
        mask = (column == val)
        ohe_matrix[mask, i] = 1
        
    # Tạo tên cột mới cho OHE
    new_col_names = [f"{col_name}_{val.decode('utf-8')}" if isinstance(val, bytes) else f"{col_name}_{val}" for val in unique_values]
    
    return ohe_matrix, new_col_names


# --- 4. Chuẩn hóa (Standardization) ---
def standardize(train_col, test_col):
    """
    Thực hiện Z-score Standardization (Mean=0, Std=1).
    Tính Mean và Std CHỈ TỪ TẬP TRAIN và áp dụng lên cả TRAIN và TEST.
    Sử dụng Vectorization CHỈ DÙNG NUMPY.
    """
    # Tính Mean và Std TỪ TẬP TRAIN
    mean_train = np.mean(train_col)
    std_train = np.std(train_col)
    
    # Tránh chia cho 0 nếu std = 0
    if std_train == 0:
        std_train = 1e-8
    
    # Áp dụng Standardization (Vectorization)
    train_standardized = (train_col - mean_train) / std_train
    test_standardized = (test_col - mean_train) / std_train
    
    return train_standardized, test_standardized


# --- 5. Lưu dữ liệu đã xử lý ---
def save_processed_data(X_train, Y_train, X_test, path_train, path_test):
    """
    Lưu X_train, Y_train và X_test đã xử lý dưới dạng CSV CHỈ DÙNG NUMPY.
    """
    # Ghép X_train và Y_train
    # X_train là ma trận 2D, Y_train là mảng 1D, cần reshape Y_train thành 2D (Nx1)
    Train_data_to_save = np.hstack([X_train, Y_train[:, np.newaxis]])
    
    try:
        # Lưu tập TRAIN:
        np.savetxt(path_train, Train_data_to_save, delimiter=',', fmt='%.6f')
        # Lưu tập TEST:
        np.savetxt(path_test, X_test, delimiter=',', fmt='%.6f')
        
        print(f"LƯU TRỮ HOÀN TẤT: Đã lưu train data vào {path_train}")
        print(f"LƯU TRỮ HOÀN TẤT: Đã lưu test data vào {path_test}")
    except Exception as e:
        print(f"LỖI LƯU TRỮ: Không thể lưu. Chi tiết: {e}")