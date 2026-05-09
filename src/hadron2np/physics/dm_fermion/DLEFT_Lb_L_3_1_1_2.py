import numpy as np


def amp_square_3_1_1(
    wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1, m_dm2, qsq
) -> float:
    """For Bs -> chibarchi"""
    wc_V = wcs["V"]
    wc_A = wcs["A"]

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

    ampSq1_1 = wc_V*wc_V.conjugate()*np.power(qsq,-2)*(0.5*(4.*(np.power(m_dm1,4) -qsq*np.power(m_dm2,2) -np.power(m_dm1,2)*(qsq + 2.*np.power(m_dm2,2)) + np.power(m_dm2,4))*np.power(m_FS -m_IS,2)*(qsq -np.power(m_FS + m_IS,2))*np.power(FFF0,2) + qsq*(qsq -np.power(m_FS -m_IS,2))*np.power(FFFv,2)*(3.*np.power(m_dm1,4) + 2.*np.power(m_dm1,2)*(qsq - 3.*np.power(m_dm2,2)) + 2.*qsq*np.power(m_dm2,2) + 3.*np.power(m_dm2,4) - 5.*np.power(qsq,2)) + (qsq -np.power(m_FS -m_IS,2))*np.power(m_FS + m_IS,2)*np.power(FFFp,2)*(np.power(m_dm1,4) + 2.*np.power(m_dm1,2)*(qsq -np.power(m_dm2,2)) + 2.*qsq*np.power(m_dm2,2) + np.power(m_dm2,4) - 3.*np.power(qsq,2))) - 0.16666666666666666*(qsq -np.power(m_FS -m_IS,2))*(-1.*np.power(m_FS + m_IS,2)*np.power(FFFp,2) + qsq*np.power(FFFv,2))*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq + np.power(m_dm2,2)) + np.power(qsq -np.power(m_dm2,2),2)))
    ampSq1_2 = 0

    ampSq2_1 = 0
    ampSq2_2 = 0.6666666666666666*np.power(qsq,-2)*(3.*(np.power(m_dm1,4) -qsq*np.power(m_dm2,2) -np.power(m_dm1,2)*(qsq + 2.*np.power(m_dm2,2)) + np.power(m_dm2,4))*np.power(m_FS + m_IS,2)*np.power(FFG0,2)*(m_FS -m_IS + np.power(qsq,0.5))*(-1.*m_FS + m_IS + np.power(qsq,0.5)) + np.power(m_FS -m_IS,2)*(qsq -np.power(m_FS + m_IS,2))*np.power(FFGp,2)*(np.power(m_dm1,4) + np.power(m_dm1,2)*(qsq - 2.*np.power(m_dm2,2)) + qsq*np.power(m_dm2,2) + np.power(m_dm2,4) - 2.*np.power(qsq,2)) + 2.*qsq*(qsq -np.power(m_FS + m_IS,2))*np.power(FFGv,2)*(np.power(m_dm1,4) + np.power(m_dm1,2)*(qsq - 2.*np.power(m_dm2,2)) + qsq*np.power(m_dm2,2) + np.power(m_dm2,4) - 2.*np.power(qsq,2)))*np.power(np.abs(wc_A),2)

    return (
        ampSq1_1
        + ampSq1_2
        + ampSq2_1
        + ampSq2_2
    ).real
