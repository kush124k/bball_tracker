import numpy as np
import supervision as sv

class PossessionEngine:
    def __init__(self, possession_threshold=100):
        """
        possession_threshold: The maximum distance (in pixels) the ball 
        can be from a player's center to be considered "in possession".
        You will likely need to tweak this number based on video resolution.
        """
        self.threshold = possession_threshold

    def get_possessor(self, players: sv.Detections, ball: sv.Detections) -> int:
        """
        Calculates distances and returns the Tracker ID of the player with the ball.
        Returns None if no one has the ball (e.g., during a pass or shot).
        """
        # If there's no ball or no players detected in this frame, no one has possession
        if len(ball) == 0 or len(players) == 0:
            return None

        # 1. Find the center coordinates of the ball
        # supervision boxes are [x_min, y_min, x_max, y_max]
        ball_box = ball.xyxy[0] 
        ball_center = np.array([
            (ball_box[0] + ball_box[2]) / 2, 
            (ball_box[1] + ball_box[3]) / 2
        ])

        closest_distance = float('inf')
        possessor_id = None

        # 2. Loop through all tracked players to find the closest one
        for i in range(len(players)):
            # Skip players that haven't been assigned an ID by the tracker yet
            tracker_id = players.tracker_id[i] if players.tracker_id is not None else None
            if tracker_id is None:
                continue

            player_box = players.xyxy[i]
            player_center = np.array([
                (player_box[0] + player_box[2]) / 2, 
                (player_box[1] + player_box[3]) / 2
            ])

            # Calculate the Euclidean distance between ball center and player center
            distance = np.linalg.norm(ball_center - player_center)

            if distance < closest_distance:
                closest_distance = distance
                possessor_id = tracker_id

        # 3. If the closest player is within our pixel threshold, they have the ball
        if closest_distance <= self.threshold:
            return possessor_id
            
        return None