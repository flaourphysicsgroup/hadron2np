import numpy as np


def amp_square_3_1_1(
    wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1, m_dm2, qsq
) -> float:
    """For Bs -> chibarchi"""
    wc_V = wcs["V"]
    wc_A = wcs["A"]

    FFf0 = ffs["f0"]
    FFfp = ffs["f+"]
    FFft = ffs["fT"]

    ampSq1_1 = -0.6666666666666666*np.power(qsq,-2)*np.power(np.abs(wc_V),2)*(np.power(FFfp,2)*(-1.*m_FS -m_IS + np.power(qsq,0.5))*(m_FS -m_IS + np.power(qsq,0.5))*(-1.*m_FS + m_IS + np.power(qsq,0.5))*(m_FS + m_IS + np.power(qsq,0.5))*(np.power(m_dm1,4) + np.power(m_dm1,2)*(qsq - 2.*np.power(m_dm2,2)) + qsq*np.power(m_dm2,2) + np.power(m_dm2,4) - 2.*np.power(qsq,2)) + 3.*(np.power(m_dm1,4) -qsq*np.power(m_dm2,2) -np.power(m_dm1,2)*(qsq + 2.*np.power(m_dm2,2)) + np.power(m_dm2,4))*np.power(FFf0,2)*np.power(np.power(m_FS,2) -np.power(m_IS,2),2))
    ampSq1_2 = 0

    ampSq2_1 = 0
    ampSq2_2 = 0

    return (
        ampSq1_1
        + ampSq1_2
        + ampSq2_1
        + ampSq2_2
    ).real
