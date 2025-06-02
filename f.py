import numpy as np
import numpy.typing as npt


def f1(x: npt.NDArray) -> float:
    x1, x2 = x[0], x[1]
    return 100 * (x2 - x1**2)**2 + 5 * (1 - x1)**2


def f2(x: npt.NDArray) -> float:
    x1, x2 = x[0], x[1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2


def grad_f1(x: npt.NDArray) -> npt.NDArray:
    x1, x2 = x[0], x[1]
    return np.array([
        -400*x1*(x2-x1**2) - 10*(1-x1),
        200*(x2-x1**2)
    ], dtype=float)


def grad_f2(x: npt.NDArray) -> npt.NDArray:
    x1, x2 = x[0], x[1]
    return np.array([
        4*x1*(x1**2+x2-11) + 2*(x1+x2**2-7), 
        2*(x1**2+x2-11) + 4*x2*(x1+x2**2-7)
    ], dtype=float)


def hessian_f1(x: npt.NDArray) -> npt.NDArray:
    x1, x2 = x[0], x[1]
    return np.array([
        [
            10 - 400*x2 + 1200*x1**2,
            -400 * x1
        ],
        [
            -400*x1,
            200
        ]
    ])


def hessian_f2(x: npt.NDArray) -> npt.NDArray:
    x1, x2 = x[0], x[1]
    return np.array([
        [
            2 + 12*x1**2,
            4*x2 + 4*x1
        ],
        [
            4*x1 + 4*x2,
            2 + 4*x1 + 12*x2**2 -28
        ]
    ])
