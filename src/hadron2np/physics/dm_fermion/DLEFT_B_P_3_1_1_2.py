import numpy as np


def amp_square_3_1_1(
    wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1, m_dm2, qsq, theta1
) -> float:
    """For Bs -> chibarchi"""
    wc_V = wcs["V"]
    wc_A = wcs["A"]

    FFf0 = ffs["f0"]
    FFfp = ffs["f+"]
    FFft = ffs["fT"]

    ampSq1_1=-1.*wc_V*wc_V.conjugate()*np.power(qsq,-2)*((np.power(m_dm1,4) -qsq*np.power(m_dm2,2) -np.power(m_dm1,2)*(qsq + 2.*np.power(m_dm2,2)) + np.power(m_dm2,4))*np.power(FFf0,2)*np.power(np.power(m_FS,2) -np.power(m_IS,2),2) + qsq*(-1.*qsq + np.power(m_dm1,2) + np.power(m_dm2,2))*np.power(FFfp,2)*(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2)) + np.power(FFfp,2)*np.power(np.cos(theta1),2)*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq + np.power(m_dm2,2)) + np.power(qsq -np.power(m_dm2,2),2))*(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2)) + 2.*FFf0*FFfp*np.cos(theta1)*(np.power(m_dm1,2) -np.power(m_dm2,2))*(np.power(m_FS,2) -np.power(m_IS,2))*np.power(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq + np.power(m_dm2,2)) + np.power(qsq -np.power(m_dm2,2),2),0.5)*np.power(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2),0.5))
    ampSq1_2 = 0

    ampSq2_1 = 0
    ampSq2_2 = 0

    return (
        ampSq1_1
        + ampSq1_2
        + ampSq2_1
        + ampSq2_2
    ).real
