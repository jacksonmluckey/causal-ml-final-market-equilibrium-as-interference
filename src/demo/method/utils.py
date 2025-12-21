from typing import Callable


def numerical_derivative(
    f: Callable[[float], float], x: float, dx: float = 1e-6
) -> float:
    r"""Compute numerical derivative using central difference."""
    return (f(x + dx) - f(x - dx)) / (2 * dx)
