import time
import csv
from contextlib import contextmanager

# Timer class to manage multiple log files
class Timer:
    def __init__(self, log_file):
        """
        Initialize the Timer class with a specific log file.
        """
        self.log_file = log_file
        # Create the log file and write the header
        with open(self.log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Section", "Time (s)"])

    def log_timing(self, name, elapsed_time):
        """
        Logs the timing result to the specified CSV file.
        """
        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([name, elapsed_time])

    @contextmanager
    def timer(self, name):
        """
        Context manager for timing a block of code.
        """
        start_time = time.perf_counter()
        yield
        elapsed_time = time.perf_counter() - start_time
        self.log_timing(name, elapsed_time)

    def timing_decorator(self, name):
        """
        Decorator for timing a function and logging the result.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed_time = time.perf_counter() - start_time
                self.log_timing(name, elapsed_time)
                return result
            return wrapper
        return decorator


# Wrapper for timing any function
def measure_function(func, timer_instance, section_name, *args, **kwargs):
    with timer_instance.timer(section_name):
        return func(*args, **kwargs)