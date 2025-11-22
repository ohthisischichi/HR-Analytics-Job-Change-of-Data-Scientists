import numpy as np

# --- Hàm Activation ---
def sigmoid(z):
    """
    Hàm kích hoạt Sigmoid: g(z) = 1 / (1 + e^-z)
    Sử dụng np.clip để tránh tràn số (overflow) khi e^-z quá lớn.
    """
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# --- Hàm Mất mát (Loss Function) ---
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


# --- Thuật toán Tối ưu (Optimization) ---
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


# --- Hàm Dự đoán cho Logistic Regression ---
def predict(X, theta, threshold=0.3):
    """
    Dự đoán xác suất và lớp (0 hoặc 1) dựa trên trọng số theta.
    """
    probabilities = sigmoid(X @ theta)
    # Trả về 0 hoặc 1 dựa trên ngưỡng (threshold)
    return (probabilities >= threshold).astype(int), probabilities
    
# --- Hàm Đánh giá Mô hình (Evaluation Metrics) ---
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

def find_best_threshold(Y_true, probabilities):
    best_thresh = 0
    best_f1 = 0
    
    # Thử chạy threshold từ 0.1 đến 0.9
    for thresh in np.arange(0.1, 0.9, 0.01):
        # Dự đoán tạm thời
        y_pred = (probabilities >= thresh).astype(int)
        
        # Tính F1 nhanh
        tp = np.sum((Y_true == 1) & (y_pred == 1))
        fp = np.sum((Y_true == 0) & (y_pred == 1))
        fn = np.sum((Y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    return best_thresh, best_f1


def k_fold_cross_validation(X, y, k=5, learning_rate=0.01, n_iterations=1000, threshold=0.5):
    """
    Thực hiện K-Fold Cross-Validation thủ công bằng NumPy.
    """
    m = X.shape[0]
    fold_size = m // k
    
    # 1. Xáo trộn dữ liệu (Shuffle)
    indices = np.arange(m)
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Danh sách lưu kết quả từng fold
    fold_metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
    
    print(f"-> Bắt đầu chạy {k}-Fold Cross-Validation...")
    
    for i in range(k):
        # 2. Xác định chỉ số cho Validation set và Train set
        start_val = i * fold_size
        end_val = start_val + fold_size if i < k-1 else m
        
        val_indices = np.arange(start_val, end_val)
        train_indices = np.delete(np.arange(m), val_indices)
        
        # 3. Chia dữ liệu
        X_val_fold = X_shuffled[start_val:end_val]
        y_val_fold = y_shuffled[start_val:end_val]
        
        X_train_fold = X_shuffled[train_indices]
        y_train_fold = y_shuffled[train_indices]
        
        # 4. Huấn luyện mô hình trên fold hiện tại
        theta, _ = gradient_descent(X_train_fold, y_train_fold, learning_rate, n_iterations)
        
        # 5. Đánh giá trên tập Validation
        y_pred_val, _ = predict(X_val_fold, theta, threshold)
        metrics = evaluate_model_numpy(y_val_fold, y_pred_val)
        
        # Lưu kết quả
        for key in fold_metrics:
            fold_metrics[key].append(metrics[key])
            
        print(f"   Fold {i+1}/{k}: F1-Score = {metrics['F1-Score']:.4f}")

    # 6. Tính trung bình
    avg_metrics = {key: np.mean(values) for key, values in fold_metrics.items()}
    
    print(f"-> Kết quả trung bình {k}-Fold: {avg_metrics}")
    return avg_metrics


# --- Random Forest & Decision Tree ---

class Node:
    """Class đại diện cho một nút trong cây quyết định."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature      # Chỉ số của đặc trưng dùng để chia (index cột)
        self.threshold = threshold  # Ngưỡng chia (giá trị < hay >=)
        self.left = left            # Cây con bên trái
        self.right = right          # Cây con bên phải
        self.value = value          # Giá trị dự đoán (nếu là nút lá)
        
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    """
    Thuật toán Cây quyết định (CART) chỉ dùng NumPy.
    """
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        # Xác định số lượng đặc trưng sẽ dùng để tìm điểm chia tốt nhất
        # (Random Forest sẽ dùng sqrt(n_features) thay vì tất cả)
        self.n_features = X.shape[1] if not self.n_features else self.n_features
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Điều kiện dừng (Stop criteria)
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Chọn ngẫu nhiên các đặc trưng để xét (đặc trưng của Random Forest)
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Tìm đặc trưng và ngưỡng chia tốt nhất
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # Nếu không tìm được cách chia tốt hơn (ví dụ: gain = 0), tạo nút lá
        if best_feat is None:
             return Node(value=self._most_common_label(y))

        # Chia dữ liệu
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        
        # Đệ quy để xây cây con
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            # Optimization: Nếu quá nhiều giá trị unique, chỉ lấy percentiles để check cho nhanh
            if len(thresholds) > 100:
                 thresholds = np.percentile(X_column, np.linspace(0, 100, 20))

            for thr in thresholds:
                # Tính Information Gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        # Entropy cha
        parent_entropy = self._entropy(y)

        # Tạo con
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Entropy con (Weighted Avg)
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # Information Gain
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        # Vectorization: Trả về indices
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        # Vectorization: Tính entropy -sum(p*log2(p))
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9)) # +epsilon để tránh log(0)

    def _most_common_label(self, y):
        # Voting: Lấy nhãn xuất hiện nhiều nhất
        if len(y) == 0: return 0
        u, counts = np.unique(y, return_counts=True)
        return u[np.argmax(counts)]

    def predict(self, X):
        # Duyệt cây cho từng mẫu
        # Lưu ý: Tree traversal khó vectorize hoàn toàn, dùng list comprehension
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForest:
    """
    Thuật toán Random Forest (Bagging) tự cài đặt.
    """
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        print(f"-> Đang huấn luyện Random Forest ({self.n_trees} cây)...")
        
        for i in range(self.n_trees):
            # 1. Bootstrap Sampling (Lấy mẫu có hoàn lại)
            n_samples = X.shape[0]
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            
            # 2. Tạo và huấn luyện cây
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
            # In tiến độ
            if (i+1) % 5 == 0 or (i+1) == self.n_trees:
                print(f"   Đã xong cây {i+1}/{self.n_trees}")

    def predict(self, X):
        # 1. Lấy dự đoán từ tất cả các cây (Matrix: n_samples x n_trees)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # 2. Transpose để mỗi hàng là các dự đoán cho 1 mẫu
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        # 3. Majority Voting (Bầu chọn đa số)
        y_pred = [self._most_common_label(preds) for preds in tree_preds]
        
        # Tính xác suất (tỷ lệ số cây vote 1)
        y_prob = np.mean(tree_preds, axis=1)
        
        return np.array(y_pred), np.array(y_prob)
    
    def _most_common_label(self, y):
        u, counts = np.unique(y, return_counts=True)
        return u[np.argmax(counts)]