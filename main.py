import numpy as np

from f import *
from methods import *


def gradient_method_output():
    res, iter = gradient_method(f1, grad_f1, np.array([100, 100], dtype=float), alpha=0.001, max_iter=10000)
    print(f"Gradient method result for f1 after {iter} iterations: {res[len(res)-1]}")

    res, iter = gradient_method(f2, grad_f2, np.array([100, 100], dtype=float), alpha=0.001, max_iter=10000)
    print(f"Gradient method result for f2 after {iter} iterations: {res[len(res)-1]}")


def newton_method_output():
    res, iter = newton_method(grad_f1, hessian_f1, np.array([0, 0], dtype=float))
    print(f"Newton method result for f1 after {iter} iterations: {res[len(res)-1]}")

    res, iter = newton_method(grad_f2, hessian_f2, np.array([0, 0], dtype=float))
    print(f"Newton method result for f2 after {iter} iterations: {res[len(res)-1]}")


def conjugate_gradient_method_output():
    res, iter = conjugate_gradient_method(f1, grad_f1, np.array([-1, -1], dtype=float))
    print(f"Conjugate gradient method result for f1 after {iter} iterations: {res[len(res)-1]}")

    res, iter = conjugate_gradient_method(f2, grad_f2, np.array([-1, -1], dtype=float))
    print(f"Conjugate gradient method result for f2 after {iter} iterations: {res[len(res)-1]}")


def nelder_mid_method_output():
    res, iter = nelder_mid_method(f1, np.array([0, 0], dtype=float))
    print(f"Conjugate gradient method result for f1 after {iter} iterations: {res[len(res)-1]}")
    
    res, iter = nelder_mid_method(f2, np.array([0, 0], dtype=float))
    print(f"Conjugate gradient method result for f2 after {iter} iterations: {res[len(res)-1]}")


def main():
    """main function"""
    conjugate_gradient_method_output()


if __name__ == "__main__":
    main()
