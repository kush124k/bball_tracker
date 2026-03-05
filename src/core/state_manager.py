import time


class StateManager:
    def __init__(self):
        # Tracking IDs and Timestamps
        self.last_known_possessor = None
        self.possession_start_time = None

        # Per-player stats
        # Example: {1: {'hold_time': 10.5, 'dribble_time': 5.0, 'passes': 2, 'shots': 0}}
        self.stats = {}

    def update_possession(self, visually_detected_possessor: int):
        current_time = time.time()

        if visually_detected_possessor is not None and visually_detected_possessor != self.last_known_possessor:
            self._log_time(self.last_known_possessor, current_time)
            self.last_known_possessor = visually_detected_possessor
            self.possession_start_time = current_time

        return self.last_known_possessor

    def log_action(self, player_id, action: str):
        """
        Called externally to record discrete action events (pass, shot)
        for a specific player.
        """
        if player_id is None:
            return

        self._ensure_player(player_id)

        if action == "pass":
            self.stats[player_id]['passes'] += 1
        elif action == "shot":
            self.stats[player_id]['shots'] += 1
        elif action == "dribbling":
            # Dribble time is accumulative — call this once per frame
            self.stats[player_id]['dribble_time'] += (1 / 30)  # assumes ~30fps

    def finalize(self):
        """
        Call this after the video finishes processing to flush the last
        possessor's time — otherwise their final stint is never logged.
        """
        self._log_time(self.last_known_possessor, time.time())

    def get_summary(self):
        return self.stats

    def _log_time(self, player_id, current_time):
        if player_id is None or self.possession_start_time is None:
            return

        duration = current_time - self.possession_start_time
        self._ensure_player(player_id)
        self.stats[player_id]['hold_time'] += duration

    def _ensure_player(self, player_id):
        if player_id not in self.stats:
            self.stats[player_id] = {
                'hold_time': 0.0,
                'dribble_time': 0.0,
                'passes': 0,
                'shots': 0
            }