class L2Loss:
    def __init__(self) -> None:
        self.name = "L2Loss"
    
    def __call__(self, y_pred, y_true):
        return (y_pred - y_true) ** 2

class L1Loss:
    def __init__(self) -> None:
        self.name = "L1Loss"
    
    def __call__(self, y_pred, y_true):
        return (y_pred - y_true).abs()