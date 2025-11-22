import numpy as np
import csv
import sys
# Tăng giới hạn đệ quy nếu cần thiết cho các thao tác mảng lớn
sys.setrecursionlimit(2000) 

# --- Load Dữ liệu ---
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


# --- Xử lý Missing Values (Imputation) ---
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

# --- Xử lý Outliers ---
def cap_outliers(data, field_map, col_name, lower_percentile=0.01, upper_percentile=0.99):
    """
    Xử lý ngoại lai bằng phương pháp Capping (Winsorizing).
    Thay thế các giá trị nằm ngoài khoảng [1%, 99%] bằng giá trị biên tương ứng.
    """
    field = field_map[col_name]
    column = data[field]
    
    # Tính các ngưỡng biên
    lower_bound = np.nanquantile(column, lower_percentile)
    upper_bound = np.nanquantile(column, upper_percentile)
    
    print(f"Capping '{col_name}': < {lower_bound:.2f} -> {lower_bound:.2f} | > {upper_bound:.2f} -> {upper_bound:.2f}")
    
    # Capping (Vectorized)
    # Dùng np.clip để giới hạn giá trị trong khoảng [lower, upper]
    data[field] = np.clip(column, lower_bound, upper_bound)
    
    return data

# --- Mã hóa (Encoding) ---
def ordinal_encode(data, field_name, mapping):
    """
    Mã hóa cột Ordinal (thứ tự) thành số.
    Sử dụng np.select để vector hóa việc mapping.
    """
    column = data[field_name]
    
    conditions = [(column == key) for key in mapping.keys()]
    choices = list(mapping.values())
    
    encoded_array = np.select(conditions, choices, default=-1) 
    return encoded_array

def one_hot_encode(data, field_map, col_name):
    """
    Tự triển khai One-Hot Encoding cho cột danh mục (Nominal).
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


# --- Chuẩn hóa (Standardization) ---
def standardize(train_col, test_col):
    """
    Thực hiện Z-score Standardization (Mean=0, Std=1).
    Tính Mean và Std CHỈ TỪ TẬP TRAIN và áp dụng lên cả TRAIN và TEST.
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


# --- Lưu dữ liệu đã xử lý ---
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


def remove_highly_correlated_features(X, threshold=0.95):
    """
    Loại bỏ các features có độ tương quan Pearson > threshold (Đa cộng tuyến).
    """
    print(f"-> Đang kiểm tra tương quan (Threshold > {threshold})...")
    
    # Tính ma trận tương quan (Correlation Matrix)
    try:
        corr_matrix = np.corrcoef(X, rowvar=False)
    except Exception as e:
        print(f"Lỗi tính ma trận tương quan: {e}")
        return X, list(range(X.shape[1]))
    
    upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    
    # Tìm index của các cột cần loại bỏ
    to_drop = [column for column in range(upper_tri.shape[1]) 
               if any(abs(corr_matrix[:, column][upper_tri[:, column]]) > threshold)]
    
    # Giữ lại các cột không bị loại
    keep_indices = [i for i in range(X.shape[1]) if i not in to_drop]
    X_reduced = X[:, keep_indices]
    
    if to_drop:
        print(f"   Đã loại bỏ {len(to_drop)} features tương quan cao.")
        # print(f"   Các index bị loại: {to_drop}")
    else:
        print("   Không có features nào bị loại.")
        
    return X_reduced, keep_indices

def binning_numerical(data, field_map, col_name, bins):
    """
    Phân nhóm (Binning) cột số liên tục thành các nhóm rời rạc.
    Sử dụng np.digitize để vector hóa.
    """
    field = field_map[col_name]
    column = data[field]
    
    # Ép kiểu dữ liệu sang float để đảm bảo tính toán đúng
    # Xử lý trường hợp dữ liệu có thể chứa chuỗi rỗng hoặc không phải số
    try:
        col_float = column.astype(float)
    except ValueError:
        col_float = np.array([float(x) if x != '' else 0 for x in column])
        
    # Thực hiện Binning bằng np.digitize
    binned_indices = np.digitize(col_float, bins)
    
    # Chuẩn hóa chỉ số về bắt đầu từ 0 (để giống encoded feature)
    binned_feature = binned_indices - 1
    
    # Đảm bảo không có index âm (nếu có giá trị nhỏ hơn bin đầu tiên)
    binned_feature = np.maximum(binned_feature, 0)
    
    return binned_feature

def smote_vectorized(X, y, k_neighbors=5, sampling_strategy='auto', random_state=42):
    """
    Triển khai SMOTE hoàn toàn bằng Vectorization (KHÔNG FOR LOOP).
    """
    np.random.seed(random_state)
    
    # 1. Tách lớp thiểu số
    unique_classes, class_counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]
    
    X_minority = X[y == minority_class]
    X_majority = X[y == majority_class]
    
    n_minority = len(X_minority)
    n_majority = len(X_majority)
    
    # Xác định số lượng cần sinh thêm
    if sampling_strategy == 'auto':
        n_synthetic = n_majority - n_minority
    else:
        n_synthetic = int(n_majority * sampling_strategy) - n_minority
        
    if n_synthetic <= 0: return X, y

    print(f"-> SMOTE Vectorized: Đang sinh {n_synthetic} mẫu...")

    X2 = np.sum(X_minority**2, axis=1).reshape(-1, 1)
    dist_matrix = X2 + X2.T - 2 * np.dot(X_minority, X_minority.T)

    knn_indices = np.argsort(dist_matrix, axis=1)[:, 1:k_neighbors+1]
    
    # SINH MẪU MỚI (Batch Processing) ---
    random_base_indices = np.random.randint(0, n_minority, n_synthetic)
    X_base = X_minority[random_base_indices]
    
    # Chọn ngẫu nhiên 1 láng giềng cho mỗi mẫu gốc
    random_neighbor_offsets = np.random.randint(0, k_neighbors, n_synthetic)
    
    chosen_neighbor_indices = knn_indices[random_base_indices, random_neighbor_offsets]
    X_neighbor = X_minority[chosen_neighbor_indices]
    
    # Công thức SMOTE: New = Base + rand * (Neighbor - Base)
    gaps = np.random.rand(n_synthetic, 1)
    
    X_synthetic = X_base + gaps * (X_neighbor - X_base)
    
    y_synthetic = np.full(n_synthetic, minority_class)
    
    X_resampled = np.vstack((X, X_synthetic))
    y_resampled = np.hstack((y, y_synthetic))
    
    print(f"-> Hoàn tất. Kích thước mới: {X_resampled.shape}")
    return X_resampled, y_resampled

def train_test_split_numpy(X, y, test_size=0.2, random_state=42):
    """
    Chia dữ liệu thành tập Train và Validation chỉ dùng NumPy.
    """
    np.random.seed(random_state)
    m = X.shape[0]
    
    # Tạo danh sách chỉ số và xáo trộn ngẫu nhiên
    indices = np.arange(m)
    np.random.shuffle(indices)
    
    # Tính điểm cắt
    split_idx = int(m * (1 - test_size))
    
    # Chia chỉ số
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Tách dữ liệu
    X_train_new = X[train_indices]
    y_train_new = y[train_indices]
    
    X_val = X[val_indices]
    y_val = y[val_indices]
    
    return X_train_new, y_train_new, X_val, y_val