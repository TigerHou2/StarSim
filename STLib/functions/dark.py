from typing import Union
Number = Union[int, float]

def exponentialDarkModel(I0, T0, dT):
    def func(T: Number) -> float:
        return I0 * 2**((T-T0)/dT)
    return func