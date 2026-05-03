import numpy as np


def kallen(asq, bsq, csq):
    """Kallen function."""
    return asq**2 + bsq**2 + csq**2 - 2 * (asq * bsq + asq * csq + bsq * csq)


def two_body_phase_space_factor(m_IS, m_1, m_2):
    factor_1 = 1 / (2 * np.pi * m_IS)
    factor_2 = np.sqrt(kallen(m_IS**2, m_1**2, m_2**2)) / (8 * m_IS**2)
    return factor_1 * factor_2


def three_body_phase_space_factor(m_IS, m_1, m_2, m_3, m_12, theta1):
    factor_1 = np.sin(theta1) / ((2 * np.pi)**3 * m_IS)
    factor_2 = np.sqrt(kallen(m_IS**2, m_12**2, m_3**2)) / (8 * m_IS**2)
    factor_3 = np.sqrt(kallen(m_12**2, m_1**2, m_2**2)) / (8 * m_12**2)
    return factor_1 * factor_2 * factor_3

def four_body_phase_space_factor(m_IS, m_1, m_2, m_3, m_4, m_12, m_34, width_34, theta1, theta2):
    factor_1 = np.pi * np.sin(theta1) * np.sin(theta2) / ((2 * np.pi)**6 * m_IS * m_34 * width_34)
    factor_2 = np.sqrt(kallen(m_IS**2, m_12**2, m_34**2)) / (8 * m_IS**2)
    factor_3 = np.sqrt(kallen(m_12**2, m_1**2, m_2**2)) / (8 * m_12**2)
    factor_4 = np.sqrt(kallen(m_34**2, m_3**2, m_4**2)) / (8 * m_34**2)
    return factor_1 * factor_2 * factor_3 * factor_4
