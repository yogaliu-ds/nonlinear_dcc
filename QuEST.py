import numpy as np
from scipy.integrate import quad
from scipy.optimize import newton


def stieltjes_transform(z, t, n, p):
    """
    Equation (2.14): Stieltjes Transformation
    :param z: Complex number 
    :param t: Sample Eigenvalues
    :param n: Sample size
    :param p: Number of Dimensions
    :return: Stieltjes transformed m
    """
    def func(m):
        return m - (1 / p) * np.sum(1 / (t*(1 - p/n - p/n*z*m) - z+ 1e-8)) 
    
    initial_guess = 0.5
    solution = newton(func, initial_guess, maxiter=200)

    return solution

def f_n_p_x(x, t, n, p):
    """
    Equation (2.15): F_{n,p}^t(x)
    :param x: Real number
    :param t: Sample Eigenvalues
    :param n: Sample size
    :param p: Number of Dimensions
    :return: F_{n,p}^t(x)
    """
    if x == 0:
        term1 = 1 - n / p
        term2 = (1 / p) * np.sum([1 if ti == 0 else 0 for ti in t])
        return max(term1, term2)
    else:
        def integrand(xi):
            m_t = stieltjes_transform(xi + 1e-10j, t, n, p) # Set a extreme small value to replace the lim problem
            return np.imag(m_t)
        
        integral, _ = quad(integrand, -10, x, limit=50) # Should be -inf but it's too time-consuming
        return (1 / np.pi) * integral

def inverse_f_n_p(u, t, n, p):
    """
    Equation (2.16): Inverse Integral Function (F_{n,p}^t)^{-1}(u)
    :param u: Cumulative Probability [0, 1]
    :param t: Sample Eigenvaluse
    :param n: Sample Size
    :param p: Number of Dimensions
    :return: x which makes F_{n,p}^t(x) <= u
    """
    x_values = np.linspace(-10, 10, 1000)
    f_values = [f_n_p_x(x, t, n, p) for x in x_values]
    eligible_x = [x for x, F in zip(x_values, f_values) if F <= u]
    # print(max(eligible_x))
    return max(eligible_x) if eligible_x else None

def quantized_eigenvalue(i, t, n, p):
    """
    Equation (2.17): q_{n,p}^i(t)
    :param i: ith Eigenvalue
    :param t: Sample Eigenvalue List
    :param n: Sample Size
    :param p: Number of Dimensions
    :return: q_{n,p}^i(t)
    """
    upper_bound = i / p
    lower_bound = (i - 1) / p
    def integrand(u):
        return inverse_f_n_p(u, t, n, p)
    
    integral, _ = quad(integrand, lower_bound, upper_bound)
    return p * integral
