from typing import Callable

import time
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar


def gradient_method(f: Callable[[npt.NDArray], npt.NDArray],
                    grad_f: Callable[[npt.NDArray], npt.NDArray], 
                    start: npt.NDArray, 
                    alpha: float = 0.001, 
                    eps: float = 1e-6, 
                    max_iter: int = 1000
                    ) -> tuple[list[npt.NDArray], int]:
    path = [start]

    prev = start
    iterations = 0
    for _ in range(max_iter):
        alpha = exact_line_search(f, prev, -grad_f(prev))

        curr = prev - alpha * grad_f(prev)
        path.append(curr)
        iterations += 1
        if np.linalg.norm(curr - prev) < eps and np.linalg.norm(grad_f(curr)) < eps: 
            break
        prev = curr

    return path, iterations


def newton_method(grad_f: Callable[[npt.NDArray], npt.NDArray],
                  hessian_f: Callable[[npt.NDArray], npt.NDArray],
                  start: npt.NDArray,
                  eps: float = 1e-6,
                  max_iter: int = 1000 
                  ) -> tuple[list[npt.NDArray], int]:
    path = [start]

    prev = start
    iterations = 0
    for _ in range(max_iter):
        curr = prev - np.linalg.inv(hessian_f(prev)) @ grad_f(prev)
        path.append(curr)
        iterations += 1
        if np.linalg.norm(curr - prev) < eps:
            break
        prev = curr

    return path, iterations


def exact_line_search(f: Callable[[npt.NDArray], npt.NDArray],
                x: npt.NDArray,
                p: npt.NDArray) -> float:
    phi = lambda alpha: f(x + alpha * p)
    res = minimize_scalar(phi)
    return res.x


def conjugate_gradient_method(f: Callable[[npt.NDArray], npt.NDArray],
                              grad_f: Callable[[npt.NDArray], npt.NDArray],
                              start: npt.NDArray,
                              eps: float = 1e-6,
                              max_iter: int = 1000) -> tuple[list[npt.NDArray], int]:
    path = [start]
    iterations = 0
    
    p = -grad_f(start)
    x = start
    for k in range(max_iter):
        alpha = exact_line_search(f, x, p)

        x_new = x + alpha * p
        path.append(x_new)
        iterations += 1

        # check for convergence
        grad_new = grad_f(x_new)
        if np.linalg.norm(grad_new) < eps:
            break
        
        # restart
        if k % 3 == 0:
            p_new = -grad_new
        else:
            beta = np.linalg.norm(grad_new)**2 / np.linalg.norm(grad_f(x))**2
            p_new = -grad_new + beta * p

        x, p = x_new, p_new
    
    return path, iterations


def nelder_mid_method(f: Callable[[npt.NDArray], npt.NDArray],
                      v1: npt.NDArray,
                      eps: float = 1e-6,
                      max_iter: int = 1000,
                      alpha: float = 1,
                      beta: float = 0.5,
                      gamma: float = 2) -> tuple[list[npt.NDArray], int]:

    v1 = np.array([0, 0], dtype=float)
    v2 = np.array([1, 0], dtype=float)
    v3 = np.array([0, 1], dtype=float)

    path = []
    iterations = 0

    for _ in range(max_iter):
        a = [[v1, f(v1)], [v2, f(v2)], [v3, f(v3)]]
        a.sort(key = lambda x: x[1])
        
        b, g, w = a[0][0], a[1][0], a[2][0]
        mid = (b + g) / 2

        # reflection
        x_r = mid + alpha * (mid - w)
        if f(x_r) < f(g):
            w = x_r
        else:
            if f(x_r) < f(w):
                w = x_r
            c = (w + mid) / 2
            if f(c) < f(w):
                w = c
        
        if f(x_r) < f(b):
            x_e = mid + gamma * (x_r - mid)
            if f(x_e) < f(x_r):
                w = x_e
            else:
                w = x_r

        if f(x_r) > f(g):
            x_c = mid + beta * (w - mid)
            if f(x_c) < f(w):
                w = x_c
        
        v1, v2, v3 = w, g, b
        iterations += 1

        vertices = np.array([v1, v2, v3])
        f_values = np.array([f(v1), f(v2), f(v3)])

        max_dist = np.max([np.linalg.norm(vertices[i] - vertices[j]) 
                           for i in range(3) for j in range(i+1, 3)])
        f_range = np.max(f_values) - np.min(f_values)
        path.append(b)
        if max_dist < eps and f_range < eps:
            break

    return path, iterations
