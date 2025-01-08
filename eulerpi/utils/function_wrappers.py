from typing import Callable


class FunctionWithDimensions:
    def __init__(self, func: Callable, dim_in: int, dim_out: int):
        self.func = func
        self.dim_in = dim_in
        self.dim_out = dim_out

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
