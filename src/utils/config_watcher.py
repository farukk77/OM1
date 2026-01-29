import os
import logging
from typing import Callable
from threading import Timer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class ConfigFileWatcher(FileSystemEventHandler):
    """
    Monitors a specific configuration file for changes using Watchdog.
    Implements debouncing to prevent multiple reloads during rapid saves.
    """
    def __init__(self, config_path: str, callback: Callable, debounce_interval: float = 0.5):
        self.config_path = os.path.abspath(config_path)
        self.callback = callback
        self.debounce_interval = debounce_interval
        self.observer = Observer()
        self._timer = None

    def start(self):
        """Starts the file monitoring."""
        directory = os.path.dirname(self.config_path)
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return

        self.observer.schedule(self, directory, recursive=False)
        self.observer.start()
        logger.info(f"Started watchdog for config file: {self.config_path}")

    def stop(self):
        """Stops the file monitoring."""
        self.observer.stop()
        self.observer.join()

    def on_modified(self, event):
        """Called when a file is modified."""
        if os.path.abspath(event.src_path) == self.config_path:
            self._debounce()

    def _debounce(self):
        """Prevents multiple callbacks firing for a single save event."""
        if self._timer is not None:
            self._timer.cancel()
        self._timer = Timer(self.debounce_interval, self.callback)
        self._timer.start()
