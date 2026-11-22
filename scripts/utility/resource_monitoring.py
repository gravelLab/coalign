import gc
import threading

import psutil

BYTES_IN_MEBIBYTE = 1024 ** 2


class PeakMemoryMonitor:
    """
    Tracks the peak resident set size (RSS) of the current process between start() and stop().

    A background thread samples the RSS every `sampling_interval_seconds`. The RSS at the start is stored as well,
    so that the memory used by the monitored code can be separated from the memory the process already held
    (e.g. a worker process of a pool that has already run other tasks).

    Note that very short allocation spikes (shorter than the sampling interval) can be missed.
    """

    def __init__(self, sampling_interval_seconds: float = 0.02):
        self._process = psutil.Process()
        self._sampling_interval_seconds = sampling_interval_seconds
        self._stop_event = threading.Event()
        self._sampling_thread: threading.Thread | None = None
        self.baseline_rss_bytes = 0
        self.peak_rss_bytes = 0

    def _sample(self):
        rss = self._process.memory_info().rss
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss

    def _sampling_loop(self):
        while not self._stop_event.wait(self._sampling_interval_seconds):
            self._sample()

    def start(self):
        # Release the garbage left by the previous work so that it doesn't inflate the baseline
        gc.collect()
        self.baseline_rss_bytes = self._process.memory_info().rss
        self.peak_rss_bytes = self.baseline_rss_bytes
        self._stop_event.clear()
        self._sampling_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._sampling_thread.start()

    def stop(self):
        if self._sampling_thread is None:
            return
        self._stop_event.set()
        self._sampling_thread.join()
        self._sampling_thread = None
        self._sample()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False

    @property
    def peak_rss_mib(self) -> float:
        return self.peak_rss_bytes / BYTES_IN_MEBIBYTE

    @property
    def peak_rss_increase_mib(self) -> float:
        return (self.peak_rss_bytes - self.baseline_rss_bytes) / BYTES_IN_MEBIBYTE
