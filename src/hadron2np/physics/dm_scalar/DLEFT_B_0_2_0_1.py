import numpy as np


def amp_square_2_0_1(wcs: dict, f_B, m_iq, m_fq, m_IS, m_dm1, m_dm2) -> float:
    wc_P = wcs["P"]
    wc_A = wcs["A"]

    ampSq1_1 = 0
    ampSq1_2 = 0
    ampSq1_3 = 0
    ampSq1_4 = 0

    ampSq2_1 = 0
    ampSq2_2 = 4.*wc_P*wc_P.conjugate()*np.power(f_B,2)*np.power(m_fq + m_iq,-2)*np.power(m_IS,4)
    ampSq2_3 = 0
    ampSq2_4 = 4.*1j*wc_A*wc_P.conjugate()*np.power(f_B,2)*(np.power(m_dm1,2) -np.power(m_dm2,2))*np.power(m_fq + m_iq,-1)*np.power(m_IS,2)

    ampSq3_1 = 0
    ampSq3_2 = 0
    ampSq3_3 = 0
    ampSq3_4 = 0

    ampSq4_1 = 0
    ampSq4_2 = -4.*1j*wc_A.conjugate()*wc_P*np.power(f_B,2)*(np.power(m_dm1,2) -np.power(m_dm2,2))*np.power(m_fq + m_iq,-1)*np.power(m_IS,2)
    ampSq4_3 = 0
    ampSq4_4 = 4.*wc_A*wc_A.conjugate()*np.power(f_B,2)*np.power(np.power(m_dm1,2) -np.power(m_dm2,2),2)

    # 返回16个元素和的实部
    return (
        ampSq1_1
        + ampSq1_2
        + ampSq1_3
        + ampSq1_4
        + ampSq2_1
        + ampSq2_2
        + ampSq2_3
        + ampSq2_4
        + ampSq3_1
        + ampSq3_2
        + ampSq3_3
        + ampSq3_4
        + ampSq4_1
        + ampSq4_2
        + ampSq4_3
        + ampSq4_4
    ).real
