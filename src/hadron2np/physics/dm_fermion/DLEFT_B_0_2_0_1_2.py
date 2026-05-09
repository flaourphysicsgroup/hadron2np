import numpy as np


def amp_square_2_0_1(wcs: dict, f_B, m_iq, m_fq, m_IS, m_dm1, m_dm2) -> float:
    """For Bs -> chibarchi"""
    wc_V = wcs["V"]
    wc_A = wcs["A"]

    ampSq1_1 = 0
    ampSq1_2 = 0

    ampSq2_1 = 0
    ampSq2_2 = wc_A*wc_A.conjugate()*np.power(f_B,2)*(-1.*np.power(m_dm1,4) -np.power(m_dm2,4) + np.power(m_dm2,2)*np.power(m_IS,2) + np.power(m_dm1,2)*(2.*np.power(m_dm2,2) + np.power(m_IS,2)))

    return (
        ampSq1_1
        + ampSq1_2
        + ampSq2_1
        + ampSq2_2
    ).real
