import time


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        if elapsed_time < 60:
            print(f"{func.__name__} took {elapsed_time:.2f} seconds to run.")
        else:
            elapsed_time_minutes = elapsed_time / 60
            print(f"{func.__name__} took {elapsed_time_minutes:.2f} minutes to run.")

        return result

    return wrapper
