def exponentialDarkModel(I0, T0, dT):
    def func(T):
        return I0 * 2**((T-T0)/dT)
    return func