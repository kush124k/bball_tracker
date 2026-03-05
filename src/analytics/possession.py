import numpy as np
import supervision as sv


class PossessionEngine:
    def __init__(self, possession_threshold=100):
        """
        possession_threshold: The maximum distance (in pixels) the ball
        can be from a player's foot to be considered "in possession".
        Tweak based on video resolution.
        """
        self.threshold = possession_threshold

    def get_possessor(self, players: sv.Detections, ball: sv.Detections) -> int:
        """
        Calculates distances from ball center to each player's foot position
        (bottom-center of bounding box) and returns the Tracker ID of the
        closest player within the threshold.
        Returns None if no one has the ball.
        """
        if len(ball) == 0 or len(players) == 0:
            return None

        # Guard: ByteTrack hasn't assigned IDs yet
        if players.tracker_id is None:
            return None

        # Ball center
        ball_box = ball.xyxy[0]
        ball_center = np.array([
            (ball_box[0] + ball_box[2]) / 2,
            (ball_box[1] + ball_box[3]) / 2
        ])

        closest_distance = float('inf')
        possessor_id = None

        for i in range(len(players)):
            tracker_id = players.tracker_id[i]
            if tracker_id is None:
                continue

            player_box = players.xyxy[i]

            # Use foot position (bottom-center) instead of body center —
            # more accurate for a grounded sport like basketball
            player_foot = np.array([
                (player_box[0] + player_box[2]) / 2,
                player_box[3]  # bottom edge
            ])

            distance = np.linalg.norm(ball_center - player_foot)

            if distance < closest_distance:
                closest_distance = distance
                possessor_id = tracker_id

        if closest_distance <= self.threshold:
            return possessor_id

        return None