import time
import functools
import inspect
import numpy as np
from typing import get_type_hints, Union, get_origin, get_args
from collections.abc import Callable

Number = Union[int, float]


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


def type_checker(func):
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    def _check_type(value, expected_type):
        origin = get_origin(expected_type)
        args = get_args(expected_type)

        # Handle Union (including nested)
        if origin is Union:
            return any(_check_type(value, arg) for arg in args)

        # Handle Callable
        if origin is Callable:
            if not callable(value):
                return False
            expected_sig = args[0] if args else None
            expected_ret = args[1] if len(args) == 2 else None
            if expected_sig is Ellipsis:
                return True
            try:
                actual_sig = inspect.signature(value)
            except ValueError:
                return False
            if len(expected_sig) != len(actual_sig.parameters):
                return False
            for ((_, param), expected_type) in zip(actual_sig.parameters.items(), expected_sig):
                if param.annotation is inspect.Parameter.empty:
                    continue
                if not param.annotation == expected_type:
                    return False
            if expected_ret and value.__annotations__.get("return") is not None:
                return value.__annotations__["return"] == expected_ret
            return True

        # Handle numpy arrays
        if isinstance(expected_type, type) and issubclass(expected_type, np.ndarray):
            return isinstance(value, np.ndarray)

        # Handle regular type
        if isinstance(expected_type, type):
            return isinstance(value, expected_type)

        return False

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for name, value in bound_args.arguments.items():
            if name in hints:
                expected_type = hints[name]
                if not _check_type(value, expected_type):
                    raise TypeError(
                        f"Argument '{name}' = {value} is not of type {expected_type}"
                    )

        result = func(*args, **kwargs)

        # Check return type if annotated
        if 'return' in hints:
            if not _check_type(result, hints['return']):
                raise TypeError(
                    f"Return value {result} is not of type {hints['return']}"
                )

        return result

    return wrapper