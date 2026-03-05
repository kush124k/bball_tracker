import numpy as np
from collections import deque


class ActionClassifier:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.y_history = deque(maxlen=window_size)
        self.last_possessor = None
        self.prev_possessor = None  # Used to detect pass/shot transitions

    def classify(self, ball_detections, possessor_id):
        """
        Classifies the current ball action based on Y-axis variance and
        possession transitions.

        Returns one of: "dribbling", "holding", "pass", "shot", "none"
        """
        # --- Detect transition events BEFORE updating state ---
        # A possession -> None transition means the ball just left someone's hands
        if self.last_possessor is not None and possessor_id is None:
            if len(ball_detections) > 0:
                ball_box = ball_detections.xyxy[0]
                ball_center_y = (ball_box[1] + ball_box[3]) / 2

                # If the ball is in the upper portion of its recent range, it's likely a shot
                if len(self.y_history) >= 2:
                    recent_avg_y = np.mean(self.y_history)
                    # Ball moving upward (decreasing Y) = shot attempt
                    if ball_center_y < recent_avg_y:
                        self._update_state(possessor_id)
                        return "shot"

            self._update_state(possessor_id)
            return "pass"

        # No ball or no possessor — return none without wiping history
        if possessor_id is None or len(ball_detections) == 0:
            self._update_state(possessor_id)
            return "none"

        # Reset window if possession changed to a new player
        if possessor_id != self.last_possessor:
            self.y_history.clear()

        self._update_state(possessor_id)

        # Track ball Y position
        ball_box = ball_detections.xyxy[0]
        center_y = (ball_box[1] + ball_box[3]) / 2
        self.y_history.append(center_y)

        # Wait for a full window before classifying
        if len(self.y_history) < self.window_size:
            return "holding"

        # Low variance = holding, high variance = dribbling
        y_variance = np.std(self.y_history)
        if y_variance > 25:
            return "dribbling"

        return "holding"

    def _update_state(self, possessor_id):
        self.prev_possessor = self.last_possessor
        self.last_possessor = possessor_id