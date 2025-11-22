# src/models.py

import numpy as np

# --- 1. Hàm Activation ---
def sigmoid(z):
    """
    Hàm kích hoạt Sigmoid: g(z) = 1 / (1 + e^-z)
    Sử dụng np.clip để tránh tràn số (overflow) khi e^-z quá lớn.
    """
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# --- 2. Hàm Mất mát (Loss Function) ---
def compute_loss(X, Y, theta):
    """
    Tính toán Binary Cross-Entropy Loss (Log Loss).
    Sử dụng Vectorization CHỈ DÙNG NUMPY.
    """
    m = X.shape[0]
    h = sigmoid(X @ theta)
    
    # Tránh lỗi log(0) bằng cách clip giá trị h
    h = np.clip(h, 1e-15, 1 - 1e-15) 
    
    # Công thức Cross-Entropy Loss (Vectorized)
    loss = (-1 / m) * np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h))
    return loss


# --- 3. Thuật toán Tối ưu (Optimization) ---
def gradient_descent(X, Y, learning_rate, n_iterations):
    """
    Triển khai thuật toán Gradient Descent để tìm bộ trọng số theta tối ưu.
    Sử dụng Vectorization CHỈ DÙNG NUMPY.
    """
    m, n = X.shape
    # Khởi tạo trọng số
    theta = np.zeros(n) 
    loss_history = []
    
    for i in range(n_iterations):
        # 1. Tính dự đoán (Hypothesis)
        h = sigmoid(X @ theta)
        # 2. Tính lỗi
        error = h - Y
        
        # 3. Tính Gradient (Vectorized)
        gradient = (1 / m) * (X.T @ error)
        
        # 4. Cập nhật Trọng số
        theta = theta - learning_rate * gradient
        
        if i % 100 == 0:
            loss = compute_loss(X, Y, theta)
            loss_history.append(loss)
            
    return theta, loss_history


# --- 4. Hàm Dự đoán ---
def predict(X, theta, threshold=0.5):
    """
    Dự đoán xác suất và lớp (0 hoặc 1) dựa trên trọng số theta.
    """
    probabilities = sigmoid(X @ theta)
    # Trả về 0 hoặc 1 dựa trên ngưỡng (threshold)
    return (probabilities >= threshold).astype(int), probabilities


# --- 5. Hàm Đánh giá Mô hình (Evaluation Metrics) ---
def evaluate_model_numpy(Y_true, Y_pred):
    """
    Tính toán các độ đo: Accuracy, Precision, Recall, F1-Score CHỈ DÙNG NUMPY.
    """
    Y_true = Y_true.astype(int)
    Y_pred = Y_pred.astype(int)
    
    # Tính toán các thành phần của Confusion Matrix (Vectorization)
    TP = np.sum((Y_true == 1) & (Y_pred == 1))
    TN = np.sum((Y_true == 0) & (Y_pred == 0))
    FP = np.sum((Y_true == 0) & (Y_pred == 1))
    FN = np.sum((Y_true == 1) & (Y_pred == 0))
    
    # Tính toán các Độ đo
    total = len(Y_true)
    accuracy = (TP + TN) / total
    
    # Tránh chia cho 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN
    }
    return metrics