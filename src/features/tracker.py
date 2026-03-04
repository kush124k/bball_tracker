import supervision as sv

class ObjectTracker:
    def __init__(self):
        """
        Initializes the ByteTrack algorithm. It handles assigning 
        persistent IDs across frames.
        """
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25, # Lower barrier to start tracking
            lost_track_buffer=60,            # Remember lost players for 60 frames (~2 seconds)
            minimum_matching_threshold=0.8   # Stricter matching to prevent swapping IDs
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Takes raw YOLO detections and assigns tracking IDs to them.
        """
        # Pass the detections through the tracker
        tracked_detections = self.tracker.update_with_detections(detections)
        return tracked_detections