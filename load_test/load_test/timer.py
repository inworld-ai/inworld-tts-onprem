"""Timer utilities for TTS load testing."""

import time


class PausableTimer:
    """A timer that can be paused to exclude client-side processing time from measurements."""

    def __init__(self):
        self._start_time = None
        self._elapsed_time = 0.0
        self._is_running = False
        self._pause_start = None

    def start(self):
        """Start the timer."""
        if self._is_running:
            return
        self._start_time = time.time()
        self._is_running = True
        self._pause_start = None

    def pause(self):
        """Pause the timer to exclude client-side processing time."""
        if not self._is_running or self._pause_start is not None:
            return
        self._pause_start = time.time()

    def resume(self):
        """Resume the timer after client-side processing."""
        if not self._is_running or self._pause_start is None:
            return
        # Add the paused time to elapsed time (this is time we want to exclude)
        pause_duration = time.time() - self._pause_start
        self._elapsed_time += pause_duration
        self._pause_start = None

    def stop(self):
        """Stop the timer and return total server time (excluding paused periods)."""
        if not self._is_running or self._start_time is None:
            return self._elapsed_time

        current_time = time.time()
        if self._pause_start is not None:
            # If we're currently paused, don't count time since pause started
            total_time = (self._pause_start - self._start_time) - self._elapsed_time
        else:
            # Normal case: total time minus any accumulated pause time
            total_time = (current_time - self._start_time) - self._elapsed_time

        self._is_running = False
        return max(0.0, total_time)

    def elapsed(self):
        """Get current elapsed server time (excluding paused periods)."""
        if not self._is_running or self._start_time is None:
            return 0.0

        current_time = time.time()
        if self._pause_start is not None:
            # If currently paused, don't count time since pause started
            return max(0.0, (self._pause_start - self._start_time) - self._elapsed_time)
        else:
            return max(0.0, (current_time - self._start_time) - self._elapsed_time)
