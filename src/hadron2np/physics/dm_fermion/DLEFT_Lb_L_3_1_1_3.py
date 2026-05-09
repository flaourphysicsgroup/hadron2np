import numpy as np


def amp_square_3_1_1(
    wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1, m_dm2, qsq
) -> float:
    """For Bs -> chibarchibar"""
    wc_S = wcs["S"]
    wc_P = wcs["P"]
    wc_T = wcs["T"]

    FFFp = ffs["f+"]
    FFF0 = ffs["f0"]
    FFFv = ffs["fp"]
    FFGp = ffs["g+"]
    FFG0 = ffs["g0"]
    FFGv = ffs["gp"]
    FFhp = ffs["h+"]
    FFhv = ffs["hp"]
    FFhtp = ffs["ht+"]
    FFhtv = ffs["htp"]

    ampSq1_1 = 8.*wc_S*wc_S.conjugate()*(-1.*qsq + np.power(m_dm1,2) + np.power(m_dm2,2))*np.power(m_fq -m_iq,-2)*np.power(m_FS -m_IS,2)*(qsq -np.power(m_FS + m_IS,2))*np.power(FFF0,2)
    ampSq1_2 = 0
    ampSq1_3 = 0

    ampSq2_1 = 0
    ampSq2_2 = 8.*wc_P*wc_P.conjugate()*(-1.*qsq + np.power(m_dm1,2) + np.power(m_dm2,2))*np.power(m_fq + m_iq,-2)*(qsq -np.power(m_FS -m_IS,2))*np.power(m_FS + m_IS,2)*np.power(FFG0,2)
    ampSq2_3 = 0

    ampSq3_1 = 0
    ampSq3_2 = 0
    ampSq3_3 = -10.666666666666666*wc_T*wc_T.conjugate()*(qsq*(qsq -np.power(m_FS -m_IS,2))*np.power(FFhp,2) + qsq*(qsq -np.power(m_FS + m_IS,2))*np.power(FFhtp,2) - 2.*np.power(m_FS -m_IS,2)*(-1.*qsq + np.power(m_FS + m_IS,2))*np.power(FFhtv,2) - 2.*(-1.*qsq + np.power(m_FS -m_IS,2))*np.power(m_FS + m_IS,2)*np.power(FFhv,2))*np.power(qsq,-2)*(-2.*np.power(m_dm1,4) + qsq*np.power(m_dm2,2) + np.power(m_dm1,2)*(qsq + 4.*np.power(m_dm2,2)) - 2.*np.power(m_dm2,4) + np.power(qsq,2))

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
