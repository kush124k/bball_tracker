import numpy as np

class ActionClassifier:
    def __init__(self, window_size=10):
        self.y_history = []
        self.window_size = window_size

    def classify(self, ball_detections, possessor_id):
        """
        Classifies 'dribbling' vs 'holding' based on the 
        standard deviation of the ball's Y-axis center.
        """
        if possessor_id is None or len(ball_detections) == 0:
            self.y_history = []
            return "none"

        # 1. Get current Ball Y-center
        ball_box = ball_detections.xyxy[0]
        center_y = (ball_box[1] + ball_box[3]) / 2
        self.y_history.append(center_y)

        # 2. Maintain the temporal window
        if len(self.y_history) > self.window_size:
            self.y_history.pop(0)

        if len(self.y_history) < self.window_size:
            return "holding"

        # 3. Calculate Variance (Low = holding, High = bouncing)
        y_variance = np.std(self.y_history)
        
        # Threshold: 25 pixels is a good start for 480p/720p
        if y_variance > 25:
            return "dribbling"
        
        return "holding"