import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # Công thức đạo hàm: -2/N * tổng( (Y_thực - Y_đoán) * X_j )
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        
        # Chỉ cần dùng 1 biến mảng NumPy, không cần bọc trong List []
        weights = np.array(initial_weights, dtype=np.float64)
        N = X.shape[0]
        num_weights = len(initial_weights)
        
        for i in range(num_iterations):
            # 1. Tính dự đoán
            y_predictions = self.get_model_prediction(X, weights)
            gradient = np.zeros(num_weights)
            
            # 2. Tính đạo hàm theo từng biến
            for j in range(num_weights):
                gradient[j] = self.get_derivative(y_predictions, Y, N, X, j)
                
            # 3. Cập nhật trọng số (Vector hóa cú pháp trừ)
            # Thay vì lặp for j, bạn có thể trừ thẳng 2 mảng NumPy cho nhau
            weights = weights - self.learning_rate * gradient
            
        return np.round(weights, 5)