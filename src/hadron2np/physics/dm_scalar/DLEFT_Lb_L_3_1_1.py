import numpy as np


def amp_square_3_1_1(
    wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1, m_dm2, qsq
) -> float:
    wc_S = wcs["S"]
    wc_P = wcs["P"]
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

    ampSq1_1 = 8.*wc_S*wc_S.conjugate()*np.power(m_fq -m_iq,-2)*np.power(m_FS -m_IS,2)*(-1.*qsq + np.power(m_FS + m_IS,2))*np.power(FFF0,2)
    ampSq1_2 = 0
    ampSq1_3 = -8.*(m_dm1 -m_dm2)*(m_dm1 + m_dm2)*wc_S.conjugate()*wc_V*np.power(m_fq -m_iq,-1)*np.power(m_FS -m_IS,2)*np.power(FFF0,2)*np.power(qsq,-1)*(m_FS + m_IS -np.power(qsq,0.5))*(m_FS + m_IS + np.power(qsq,0.5))
    ampSq1_4 = 0
    
    ampSq2_1 = 0
    ampSq2_2 = 8.*wc_P*wc_P.conjugate()*np.power(m_fq + m_iq,-2)*(-1.*qsq + np.power(m_FS -m_IS,2))*np.power(m_FS + m_IS,2)*np.power(FFG0,2)
    ampSq2_3 = 0
    ampSq2_4 = -8.*1j*(m_dm1 -m_dm2)*(m_dm1 + m_dm2)*wc_A*wc_P.conjugate()*np.power(m_fq + m_iq,-1)*np.power(m_FS + m_IS,2)*np.power(FFG0,2)*np.power(qsq,-1)*(m_FS -m_IS + np.power(qsq,0.5))*(-1.*m_FS + m_IS + np.power(qsq,0.5))
    
    ampSq3_1 = -8.*(m_dm1 -m_dm2)*(m_dm1 + m_dm2)*wc_S*wc_V.conjugate()*np.power(m_fq -m_iq,-1)*np.power(m_FS -m_IS,2)*np.power(FFF0,2)*np.power(qsq,-1)*(m_FS + m_IS -np.power(qsq,0.5))*(m_FS + m_IS + np.power(qsq,0.5))
    ampSq3_2 = 0
    ampSq3_3 = -2.6666666666666665*np.power(qsq,-2)*np.power(np.abs(wc_V),2)*(np.power(m_FS + m_IS,2)*np.power(FFFp,2)*(m_dm1 -m_dm2 -np.power(qsq,0.5))*(m_dm1 + m_dm2 -np.power(qsq,0.5))*(m_dm1 -m_dm2 + np.power(qsq,0.5))*(m_dm1 + m_dm2 + np.power(qsq,0.5))*(m_FS -m_IS + np.power(qsq,0.5))*(-1.*m_FS + m_IS + np.power(qsq,0.5)) + 2.*qsq*np.power(FFFv,2)*(-1.*m_dm1 -m_dm2 + np.power(qsq,0.5))*(m_dm1 -m_dm2 + np.power(qsq,0.5))*(-1.*m_dm1 + m_dm2 + np.power(qsq,0.5))*(m_dm1 + m_dm2 + np.power(qsq,0.5))*(m_FS -m_IS + np.power(qsq,0.5))*(-1.*m_FS + m_IS + np.power(qsq,0.5)) - 3.*np.power(m_FS -m_IS,2)*(-1.*qsq + np.power(m_FS + m_IS,2))*np.power(FFF0,2)*np.power(np.power(m_dm1,2) -np.power(m_dm2,2),2))
    ampSq3_4 = 0
    
    ampSq4_1 = 0
    ampSq4_2 = 8.*1j*(m_dm1 -m_dm2)*(m_dm1 + m_dm2)*wc_A.conjugate()*wc_P*np.power(m_fq + m_iq,-1)*np.power(m_FS + m_IS,2)*np.power(FFG0,2)*np.power(qsq,-1)*(m_FS -m_IS + np.power(qsq,0.5))*(-1.*m_FS + m_IS + np.power(qsq,0.5))
    ampSq4_3 = 0
    ampSq4_4 = -2.6666666666666665*np.power(qsq,-2)*np.power(np.abs(wc_A),2)*(np.power(m_FS -m_IS,2)*(qsq -np.power(m_FS + m_IS,2))*np.power(FFGp,2)*(m_dm1 -m_dm2 -np.power(qsq,0.5))*(m_dm1 + m_dm2 -np.power(qsq,0.5))*(m_dm1 -m_dm2 + np.power(qsq,0.5))*(m_dm1 + m_dm2 + np.power(qsq,0.5)) + 2.*qsq*(qsq -np.power(m_FS + m_IS,2))*np.power(FFGv,2)*(m_dm1 -m_dm2 -np.power(qsq,0.5))*(m_dm1 + m_dm2 -np.power(qsq,0.5))*(m_dm1 -m_dm2 + np.power(qsq,0.5))*(m_dm1 + m_dm2 + np.power(qsq,0.5)) - 3.*(-1.*qsq + np.power(m_FS -m_IS,2))*np.power(m_FS + m_IS,2)*np.power(FFG0,2)*np.power(np.power(m_dm1,2) -np.power(m_dm2,2),2))

    return (
        ampSq1_1 + ampSq1_2 + ampSq1_3 + ampSq1_4 +
        ampSq2_1 + ampSq2_2 + ampSq2_3 + ampSq2_4 +
        ampSq3_1 + ampSq3_2 + ampSq3_3 + ampSq3_4 +
        ampSq4_1 + ampSq4_2 + ampSq4_3 + ampSq4_4
    ).real
