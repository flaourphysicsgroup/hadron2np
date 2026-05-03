import numpy as np


def amp_square_3_1_1(
    wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1, m_dm2, qsq, theta1
) -> float:
    """For Bs -> chibarchi"""
    wc_V = wcs["V"]
    wc_A = wcs["A"]

    FFV = ffs["V"]
    FFA0 = ffs["A0"]
    FFA1 = ffs["A1"]
    FFA2 = ffs["A2"]
    FFA3 = ffs["A3"]
    FFT1 = ffs["T1"]
    FFT2 = ffs["T2"]
    FFT3 = ffs["T3"]

    ampSq1_1=wc_V*wc_V.conjugate()*np.power(m_FS + m_IS,-2)*np.power(FFV,2)*np.power(qsq,-1)*(-1.*np.power(m_dm1,4) + 2.*np.power(m_dm1,2)*np.power(m_dm2,2) -np.power(m_dm2,4) + np.power(qsq,2) + np.power(np.cos(theta1),2)*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq + np.power(m_dm2,2)) + np.power(qsq -np.power(m_dm2,2),2)))*(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))
    ampSq1_2=2.*wc_A*wc_V.conjugate()*FFA1*FFV*np.cos(theta1)*np.power(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq + np.power(m_dm2,2)) + np.power(qsq -np.power(m_dm2,2),2),0.5)*np.power(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2),0.5)

    ampSq2_1=2.*wc_A.conjugate()*wc_V*FFA1*FFV*np.cos(theta1)*np.power(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq + np.power(m_dm2,2)) + np.power(qsq -np.power(m_dm2,2),2),0.5)*np.power(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2),0.5)
    ampSq2_2=-0.25*wc_A*wc_A.conjugate()*np.power(m_FS,-2)*np.power(m_FS + m_IS,-2)*np.power(qsq,-2)*(2.*FFA2*np.power(m_FS + m_IS,2)*(2.*m_FS*(m_FS -m_IS)*(FFA0 -FFA3)*(np.power(m_dm1,4) -qsq*np.power(m_dm2,2) -np.power(m_dm1,2)*(qsq + 2.*np.power(m_dm2,2)) + np.power(m_dm2,4)) + FFA1*(np.power(m_dm1,4)*(np.power(m_FS,2) -np.power(m_IS,2)) + np.power(m_dm2,4)*(np.power(m_FS,2) -np.power(m_IS,2)) + (np.power(m_dm2,2) -np.power(m_FS,2) + np.power(m_IS,2))*np.power(qsq,2) + np.power(m_dm1,2)*(2.*np.power(m_dm2,2)*(-1.*np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2)) -np.power(qsq,3)))*(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2)) + np.power(np.cos(theta1),2)*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq + np.power(m_dm2,2)) + np.power(qsq -np.power(m_dm2,2),2))*(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))*(2.*FFA1*FFA2*(qsq + np.power(m_FS,2) -np.power(m_IS,2))*np.power(m_FS + m_IS,2) + np.power(m_FS + m_IS,4)*np.power(FFA1,2) + np.power(FFA2,2)*(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))) + np.power(FFA2,2)*(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))*(np.power(m_dm1,4)*np.power(np.power(m_FS,2) -np.power(m_IS,2),2) + np.power(m_dm1,2)*(-2.*(np.power(m_FS,2) + np.power(m_IS,2))*np.power(qsq,2) + np.power(qsq,3) - 2.*np.power(m_dm2,2)*np.power(np.power(m_FS,2) -np.power(m_IS,2),2)) -(qsq -np.power(m_dm2,2))*(-2.*(np.power(m_FS,2) + np.power(m_IS,2))*np.power(qsq,2) + np.power(qsq,3) + qsq*np.power(np.power(m_FS,2) -np.power(m_IS,2),2) + np.power(m_dm2,2)*np.power(np.power(m_FS,2) -np.power(m_IS,2),2))) + np.power(m_FS + m_IS,2)*(-4.*m_FS*(m_FS + m_IS)*FFA1*FFA3*(np.power(m_dm1,4) -qsq*np.power(m_dm2,2) -np.power(m_dm1,2)*(qsq + 2.*np.power(m_dm2,2)) + np.power(m_dm2,4))*(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2)) + 4.*m_FS*FFA0*((m_FS + m_IS)*FFA1 - 2.*m_FS*FFA3)*(np.power(m_dm1,4) -qsq*np.power(m_dm2,2) -np.power(m_dm1,2)*(qsq + 2.*np.power(m_dm2,2)) + np.power(m_dm2,4))*(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2)) + 4.*(np.power(m_dm1,4) -qsq*np.power(m_dm2,2) -np.power(m_dm1,2)*(qsq + 2.*np.power(m_dm2,2)) + np.power(m_dm2,4))*np.power(m_FS,2)*np.power(FFA0,2)*(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2)) + 4.*(np.power(m_dm1,4) -qsq*np.power(m_dm2,2) -np.power(m_dm1,2)*(qsq + 2.*np.power(m_dm2,2)) + np.power(m_dm2,4))*np.power(m_FS,2)*np.power(FFA3,2)*(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2)) + np.power(m_FS + m_IS,2)*np.power(FFA1,2)*(-2.*np.power(m_dm1,2)*(2.*qsq*np.power(m_dm2,2)*(np.power(m_FS,2) -np.power(m_IS,2)) + (np.power(m_dm2,2) - 2.*np.power(m_FS,2))*np.power(qsq,2) + np.power(m_dm2,2)*np.power(np.power(m_FS,2) -np.power(m_IS,2),2)) -(qsq -np.power(m_dm2,2))*(qsq*(np.power(m_FS,2) -np.power(m_IS,2))*(2.*np.power(m_dm2,2) + np.power(m_FS,2) -np.power(m_IS,2)) + (np.power(m_dm2,2) + 6.*np.power(m_FS,2) - 2.*np.power(m_IS,2))*np.power(qsq,2) + np.power(qsq,3) + np.power(m_dm2,2)*np.power(np.power(m_FS,2) -np.power(m_IS,2),2)) + np.power(m_dm1,4)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2))) + 2.*(m_FS + m_IS)*(2.*m_FS*FFA0 + (m_FS + m_IS)*FFA1 + m_FS*FFA2 -m_IS*FFA2 - 2.*m_FS*FFA3)*np.cos(theta1)*(np.power(m_dm1,2) -np.power(m_dm2,2))*(FFA1*(qsq + np.power(m_FS,2) -np.power(m_IS,2))*np.power(m_FS + m_IS,2) + FFA2*(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2)))*np.power(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq + np.power(m_dm2,2)) + np.power(qsq -np.power(m_dm2,2),2),0.5)*np.power(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2),0.5))

    return (
        ampSq1_1
        + ampSq1_2
        + ampSq2_1
        + ampSq2_2
    ).real
