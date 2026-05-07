import numpy as np


def amp_square_3_1_1(
    wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1, m_dm2, qsq
) -> float:
    """For Bs -> chichi"""
    wc_S = wcs["S"]
    wc_P = wcs["P"]
    wc_T = wcs["T"]

    FFf0 = ffs["f0"]
    FFfp = ffs["f+"]
    FFft = ffs["fT"]

    ampSq1_1=-8.*wc_S*wc_S.conjugate()*(-1.*qsq + np.power(m_dm1,2) + np.power(m_dm2,2))*np.power(m_fq -m_iq,-2)*np.power(FFf0,2)*np.power(np.power(m_FS,2) -np.power(m_IS,2),2)
    ampSq1_2 = 0
    ampSq1_3 = 0

    ampSq2_1 = 0
    ampSq2_2 = 0
    ampSq2_3 = 0

    ampSq3_1 = 0
    ampSq3_2 = 0
    ampSq3_3=10.666666666666666*np.power(m_FS + m_IS,-2)*np.power(FFft,2)*np.power(qsq,-1)*(-1.*m_FS -m_IS + np.power(qsq,0.5))*(m_FS -m_IS + np.power(qsq,0.5))*(-1.*m_FS + m_IS + np.power(qsq,0.5))*(m_FS + m_IS + np.power(qsq,0.5))*(-2.*np.power(m_dm1,4) + qsq*np.power(m_dm2,2) + np.power(m_dm1,2)*(qsq + 4.*np.power(m_dm2,2)) - 2.*np.power(m_dm2,4) + np.power(qsq,2))*np.power(np.abs(wc_T),2)

    return (
        ampSq1_1
        + ampSq1_2
        + ampSq1_3
        + ampSq2_1
        + ampSq2_2
        + ampSq2_3
        + ampSq3_1
        + ampSq3_2
        + ampSq3_3
    ).real
