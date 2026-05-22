#!/usr/bin/python
r"""标准模型 Lambda_b -> Lambda nu bar nu。
参考 Altmannshofer 2501.10652v2 中式(10).
"""

import hadron2np
import numpy as np
from ckmutil import ckm


def xi(a, bc):
    """CKM factor xi_a^bc = V_{ab} V_{ac}^*"""
    par = hadron2np.parameters_dict
    _q_dict_u = {'u': 0, 'c': 1, 't': 2}
    _q_dict_d = {'d': 0, 's': 1, 'b': 2}
    a_index = _q_dict_u[a]
    b_index = _q_dict_d[bc[0]]
    c_index = _q_dict_d[bc[1]]
    v = ckm.ckm_tree(par['Vus'], par['Vub'], par['Vcb'], par['delta'])
    return v[a_index, b_index] * v[a_index, c_index].conj()


def dGamma_dE(ffs, m_IS, m_FS, qsq):
    FFFp = ffs["f+"]
    FFFv = ffs["fp"]
    FFGp = ffs["g+"]
    FFGv = ffs["gp"]

    par = hadron2np.parameters_dict
    GF = par['GF']
    alpha = par['alpha_e']
    wc_L = par['C_L_sbnunu']
    wc_R = par['C_R_sbnunu']

    E_FS = (m_IS**2 + m_FS**2 - qsq) / (2 * m_IS)
    x_FS = m_FS / m_IS
    calF_V = (1 - m_FS / E_FS) * (
        (1 + x_FS) ** 2 * FFFp**2 + 2 * qsq * FFFv**2 / m_IS**2
    )
    calF_A = (1 + m_FS / E_FS) * (
        (1 - x_FS) ** 2 * FFGp**2 + 2 * qsq * FFGv**2 / m_IS**2
    )
    prefactor = (
        alpha**2
        * GF**2
        / (32 * np.pi**5)
        * m_IS**2
        * E_FS**2
        * abs(xi('t', 'bs')) ** 2
        * np.sqrt(1 - m_FS**2 / E_FS**2)
    )

    return prefactor * (abs(wc_L + wc_R) ** 2 * calF_V + abs(wc_L - wc_R) ** 2 * calF_A)


def dGamma_dqsq(ffs, m_IS, m_FS, qsq):
    partial_width = dGamma_dE(ffs, m_IS, m_FS, qsq)
    return partial_width / (2 * m_IS)
