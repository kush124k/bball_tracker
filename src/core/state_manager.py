import time

class StateManager:
    def __init__(self):
        # Tracking IDs and Timestamps
        self.last_known_possessor = None
        self.possession_start_time = None
        
        # Dictionary to store total seconds per Player ID
        self.stats = {} # Example: {1: {'hold_time': 10.5, 'dribble_time': 5.0}}

    def update_possession(self, visually_detected_possessor: int):
        current_time = time.time()
        
        # If the possessor changed, we need to log the duration for the previous guy
        if visually_detected_possessor is not None and visually_detected_possessor != self.last_known_possessor:
            self._log_time(self.last_known_possessor, current_time)
            self.last_known_possessor = visually_detected_possessor
            self.possession_start_time = current_time
            
        return self.last_known_possessor

    def _log_time(self, player_id, current_time):
        if player_id is None or self.possession_start_time is None:
            return
            
        duration = current_time - self.possession_start_time
        
        if player_id not in self.stats:
            self.stats[player_id] = {'hold_time': 0.0}
            
        self.stats[player_id]['hold_time'] += duration

    def get_summary(self):
        return self.stats