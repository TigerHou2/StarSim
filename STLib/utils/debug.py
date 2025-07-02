import time
import functools


# Module-level flag to enable/disable timing
ENABLE_TIMER = True

# function timing
def timer(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        if not ENABLE_TIMER:
            return func(*args, **kwargs)
        
        start_time = time.process_time()
        result = func(*args, **kwargs)
        end_time = time.process_time()
        elapsed = end_time - start_time

        print(f"Function '{func.__name__}' took {elapsed:.3f} seconds.")
        return result
    
    return wrapper