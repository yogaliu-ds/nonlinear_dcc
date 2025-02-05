import numpy as np

def stieltjes_transform(z, eigenvalues):
    """
    Computes the Stieltjes transform m_F(z) for a given complex z and sample eigenvalues.
    :param z: Complex number z
    :param eigenvalues: List or array of sample eigenvalues (real numbers)
    :return: Stieltjes transform m_F(z)
    """
    eigenvalues = np.asarray(eigenvalues)  # 確保 eigenvalues 是 numpy array
    m_F = np.mean(1 / (eigenvalues - z))   # 向量化計算
    return m_F

def delta(lambda_val, gamma, eigenvalues, m_F_zero):
    """
    計算非線性收縮因子 δ(λ)
    :param lambda_val: 特徵值 λ
    :param gamma: 比例參數 γ (p / n)
    :param m_F: Stieltjes 變換的估計值
    :return: 非線性收縮因子 δ(λ)
    """

    m_F = stieltjes_transform(lambda_val, eigenvalues)
    # m_F_zero = stieltjes_transform(0, eigenvalues)
    if lambda_val > 0:
        return lambda_val / (np.abs(1 - gamma**-1 - gamma**-1 * lambda_val * m_F)**2)
    elif lambda_val == 0 and gamma < 1:
        return gamma / ((1 - gamma) * m_F_zero)
    else:
        return 0