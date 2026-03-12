import supervision as sv


class ObjectTracker:
    def __init__(self, profile: dict):
        """
        Accepts an angle-specific profile dict loaded from angle_profiles.yaml.
        Example profile:
            track_activation_threshold: 0.25
            lost_track_buffer: 90
            minimum_matching_threshold: 0.70
            minimum_consecutive_frames: 3
            frame_rate: 30
        """
        self.tracker = sv.ByteTrack(
            track_activation_threshold=profile.get('track_activation_threshold', 0.25),
            lost_track_buffer=profile.get('lost_track_buffer', 60),
            minimum_matching_threshold=profile.get('minimum_matching_threshold', 0.80),
            minimum_consecutive_frames=profile.get('minimum_consecutive_frames', 2),
            frame_rate=profile.get('frame_rate', 30)
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self.tracker.update_with_detections(detections)