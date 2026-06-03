#!/usr/bin/python
r"""标准模型B介子衰变到强子+中微子的半单举过程。参考1111.6402。
共有4个过程, 分别是 $B->pi(rho)vv$ 和 $B->K(K*)vv$ 的带电过程.
"""

from hadron2np.Phase_space_factors import kallen
import numpy as np
from .Lb_Lvv import xi


def _Gamma_SM_Bpirho(GF, f_IS, f_FS, m_IS, r_tau, tau_tau) -> float:
    """一个固定的系数 For B+->pi+(rho+)vv"""
    return (
        m_IS**6
        * r_tau**5
        * f_IS**2
        * f_FS**2
        * abs(xi('u', 'bd')) ** 2
        * GF**4
        * tau_tau
        / (64 * np.pi**2)
    )


def _Gamma_SM_BK(GF, alpha, C_L_sbnunu, m_IS) -> float:
    """一个固定的系数 For B+->K(*)+vv"""
    return m_IS**5 * (GF * alpha * C_L_sbnunu * abs(xi('t', 'bs'))) ** 2 / (256 * np.pi**5)


def _dGamma_dz_piPlus(GF, f_IS, f_FS, m_IS, m_FS, m_tau, tau_tau, z):
    """dGamma/dz(B+ -> pi+ v v)"""
    r_tau = m_tau / m_IS
    r_pi = m_FS / m_IS
    gamma_SM = _Gamma_SM_Bpirho(GF, f_IS, f_FS, m_IS, r_tau, tau_tau)
    return gamma_SM * ((1 - r_tau**2) * (1 - r_pi**2 / r_tau**2) - z)


def _dGamma_dz_rhoPlus(GF, f_IS, f_FS, m_IS, m_FS, m_tau, tau_tau, z, polarization: str = 'L'):
    """dGamma/dz(B+ -> rho+ v v)"""
    r_tau = m_tau / m_IS
    r_rho = m_FS / m_IS
    lambda_rhoz = kallen(1, z, r_rho**2)
    gamma_SM = _Gamma_SM_Bpirho(GF, f_IS, f_FS, m_IS, r_tau, tau_tau)
    if polarization == 'L':
        return (
            gamma_SM
            * (1 - z - r_rho**2) ** 2
            * ((1 - r_tau**2) * (1 - r_rho**2 / r_tau**2) - z)
            / lambda_rhoz
        )
    elif polarization == 'T':
        return (
            gamma_SM
            * z
            * (lambda_rhoz - 2 * (1 - r_tau**2) * (1 - r_rho**2 / r_tau**2) + 2 * z)
            / lambda_rhoz
        )
    else:
        raise ValueError('polarization must be L or T')


def _dGamma_dz_KPlus(GF, alpha, C_L_sbnunu, ffs, m_IS, m_FS, z):
    """dGamma/dz(B+ -> K+ v v)"""
    r_K = m_FS / m_IS
    lambda_Kz = kallen(1, z, r_K**2)
    gamma_SM = _Gamma_SM_BK(GF, alpha, C_L_sbnunu, m_IS)
    ff_fplus = ffs['f+']

    return gamma_SM * np.sqrt(lambda_Kz) ** 3 * ff_fplus**2


def _dGamma_dz_KstarPlus(GF, alpha, C_L_sbnunu, ffs, m_IS, m_FS, z, polarization: str = 'L'):
    """dGamma/dz(B+ -> K*+ v v)"""
    r_FS = m_FS / m_IS
    lambda_Kz = kallen(1, z, r_FS**2)
    gamma_SM = _Gamma_SM_BK(GF, alpha, C_L_sbnunu, m_IS)
    ff_V = ffs['V']
    ff_A1 = ffs['A1']
    ff_L = ffs['A2'] - (1 + r_FS) ** 2 * (1 - r_FS**2 - z) * ff_A1 / lambda_Kz
    if polarization == 'T':
        return (
            gamma_SM
            * 2
            * z
            * np.sqrt(lambda_Kz)
            * (lambda_Kz * ff_V**2 / (1 + r_FS) ** 2 + (1 + r_FS) ** 2 * ff_A1**2)
        )
    elif polarization == 'L':
        return gamma_SM * np.power(lambda_Kz, 5.0 / 2.0) * ff_L**2 / (4 * r_FS**2 * (1 + r_FS) ** 2)
    else:
        raise ValueError('polarization must be L or T')


def partial_width_B2pi_plus(par, qsq):
    GF = par['GF']
    m_IS = par['m_B+']
    m_FS = par['m_pi+']
    f_IS = par['f_B+']
    f_FS = par['f_pi+']
    m_tau = par['m_tau']
    tau_tau = par['tau_tau']
    z = qsq / m_IS**2
    return _dGamma_dz_piPlus(GF, f_IS, f_FS, m_IS, m_FS, m_tau, tau_tau, z)


def partial_width_B2rho_plus(par, qsq, polarization='T'):
    GF = par['GF']
    m_IS = par['m_B+']
    m_FS = par['m_rho+']
    f_IS = par['f_B+']
    f_FS = par['f_rho+']
    m_tau = par['m_tau']
    tau_tau = par['tau_tau']
    z = qsq / m_IS**2
    return _dGamma_dz_rhoPlus(GF, f_IS, f_FS, m_IS, m_FS, m_tau, tau_tau, z, polarization)


def partial_width_B2K_plus(par, ffs, m_IS, m_FS, qsq):
    GF = par['GF']
    alpha = par['alpha_e']
    C_L_sbnunu = par['C_L_sbnunu']
    z = qsq / m_IS**2
    return _dGamma_dz_KPlus(GF, alpha, C_L_sbnunu, ffs, m_IS, m_FS, z)


def partial_width_B2Kstar_plus(par, ffs, m_IS, m_FS, qsq, polarization):
    GF = par['GF']
    alpha = par['alpha_e']
    C_L_sbnunu = par['C_L_sbnunu']
    z = qsq / m_IS**2
    return _dGamma_dz_KstarPlus(GF, alpha, C_L_sbnunu, ffs, m_IS, m_FS, z, polarization)
