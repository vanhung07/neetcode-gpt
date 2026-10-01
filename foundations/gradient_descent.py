class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        w = init
        for i in range(iterations): 
            w = w - learning_rate*2*w
            if abs(2*w) <= 1e-3:
                break
        return round(w, 5)
