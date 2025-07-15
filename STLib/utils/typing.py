import functools
import inspect
import numpy as np
from typing import get_type_hints, Union, get_origin, get_args
from collections.abc import Callable

Number = Union[int, float]


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
            
            actual_sig = inspect.signature(value)
            if len(expected_sig) != len(actual_sig.parameters):
                return False
            for ((_, param), expected_type) in zip(actual_sig.parameters.items(), expected_sig):
                if param.annotation is inspect.Parameter.empty:
                    continue
                if not param.annotation == expected_type:
                    return False
            if actual_sig.return_annotation is not inspect.Parameter.empty:
                return actual_sig.return_annotation == expected_ret
            return True

        # Handle numpy arrays
        if isinstance(expected_type, type) and issubclass(expected_type, np.ndarray):
            return isinstance(value, np.ndarray)
        
        # Handle tuples
        if origin is tuple:
            if not isinstance(value, tuple):
                return False
            if len(args) != len(value):
                return False
            return all(_check_type(v, arg) for v, arg in zip(value, args))

        # Handle regular type
        if isinstance(expected_type, type):
            return isinstance(value, expected_type)

        return False  # pragma: no cover  (should never reach this)

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

        return result

    return wrapper