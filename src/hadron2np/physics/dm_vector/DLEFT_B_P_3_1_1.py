import numpy as np

def amp_square_3_1_1(
    wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1, m_dm2, qsq, theta1
) -> float:
    wc_S = wcs["S"]
    wc_P = wcs["P"]
    wc_V = wcs["V"]
    wc_A = wcs["A"]
    wc_V_tilde = wcs["V_tilde"]
    wc_A_tilde = wcs["A_tilde"]
    wc_T = wcs["T"]
    wc_T5 = wcs["T5"]
    wc_V_partial = wcs["V_partial"]
    wc_A_partial = wcs["A_partial"]
    wc_V_21 = wcs["V_21"]
    wc_A_21 = wcs["A_21"]
    wc_V_tilde_21 = wcs["V_tilde_21"]
    wc_A_tilde_21 = wcs["A_tilde_21"]

    # 提取 ffs for B -> P (Scalar/Pseudoscalar meson): f0, f+, fT
    FFf0 = ffs["f0"]
    FFfp = ffs["f+"]
    FFft = ffs["fT"]

    # 初始化 16 个振幅平方项 (4x4 matrix elements squared/interference)
    # 注意：此处公式需要根据具体物理过程填充，目前设为 0
    ampSq1_1=wc_S*wc_S.conjugate()*np.power(m_dm1,-2)*np.power(m_dm2,-2)*np.power(m_fq -m_iq,-2)*np.power(FFf0,2)*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq - 5.*np.power(m_dm2,2)) + np.power(qsq -np.power(m_dm2,2),2))*np.power(np.power(m_FS,2) -np.power(m_IS,2),2)
    ampSq1_2 = 0
    ampSq1_3=-0.5*1j*wc_S.conjugate()*wc_V*FFf0*np.power(m_dm2,-2)*np.power(m_fq -m_iq,-1)*(np.power(m_FS,2) -np.power(m_IS,2))*np.power(qsq,-1)*(FFf0*(np.power(m_FS,2) -np.power(m_IS,2))*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq - 2.*np.power(m_dm2,2)) + 4.*qsq*np.power(m_dm2,2) - 5.*np.power(m_dm2,4) + np.power(qsq,2)) + FFfp*np.cos(theta1)*(-1.*qsq + np.power(m_dm1,2) + 5.*np.power(m_dm2,2))*np.power(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq + np.power(m_dm2,2)) + np.power(qsq -np.power(m_dm2,2),2),0.5)*np.power(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2),0.5))
    ampSq1_4 = 0
    
    ampSq2_1 = 0
    ampSq2_2 = 0
    ampSq2_3 = 0
    ampSq2_4 = 0
    
    ampSq3_1=0.5*1j*wc_S*wc_V.conjugate()*FFf0*np.power(m_dm2,-2)*np.power(m_fq -m_iq,-1)*(np.power(m_FS,2) -np.power(m_IS,2))*np.power(qsq,-1)*(FFf0*(np.power(m_FS,2) -np.power(m_IS,2))*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq - 2.*np.power(m_dm2,2)) + 4.*qsq*np.power(m_dm2,2) - 5.*np.power(m_dm2,4) + np.power(qsq,2)) + FFfp*np.cos(theta1)*(-1.*qsq + np.power(m_dm1,2) + 5.*np.power(m_dm2,2))*np.power(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq + np.power(m_dm2,2)) + np.power(qsq -np.power(m_dm2,2),2),0.5)*np.power(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2),0.5))
    ampSq3_2 = 0
    ampSq3_3=0.25*wc_V*wc_V.conjugate()*np.power(m_dm2,-2)*np.power(qsq,-2)*(np.power(FFf0,2)*(-2.*qsq*np.power(m_dm1,4) + np.power(m_dm1,6) + np.power(m_dm1,2)*(6.*qsq*np.power(m_dm2,2) - 3.*np.power(m_dm2,4) + np.power(qsq,2)) + 2.*np.power(-1.*m_dm2*qsq + np.power(m_dm2,3),2))*np.power(np.power(m_FS,2) -np.power(m_IS,2),2) + qsq*np.power(FFfp,2)*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq + np.power(m_dm2,2)) + np.power(qsq -np.power(m_dm2,2),2))*(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2)) + np.power(FFfp,2)*np.power(np.cos(theta1),2)*(-3.*qsq*np.power(m_dm1,4) + np.power(m_dm1,6) + np.power(m_dm1,2)*(-4.*qsq*np.power(m_dm2,2) - 3.*np.power(m_dm2,4) + 3.*np.power(qsq,2)) -(qsq - 2.*np.power(m_dm2,2))*np.power(qsq -np.power(m_dm2,2),2))*(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2)) + 2.*FFf0*FFfp*np.cos(theta1)*(np.power(m_dm1,4) + 2.*(qsq -np.power(m_dm2,2))*np.power(m_dm2,2) + np.power(m_dm1,2)*(-1.*qsq + np.power(m_dm2,2)))*(np.power(m_FS,2) -np.power(m_IS,2))*np.power(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq + np.power(m_dm2,2)) + np.power(qsq -np.power(m_dm2,2),2),0.5)*np.power(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2),0.5))
    ampSq3_4 = 0
    
    ampSq4_1 = 0
    ampSq4_2 = 0
    ampSq4_3 = 0
    ampSq4_4 = 0

    return (
        ampSq1_1 + ampSq1_2 + ampSq1_3 + ampSq1_4 +
        ampSq2_1 + ampSq2_2 + ampSq2_3 + ampSq2_4 +
        ampSq3_1 + ampSq3_2 + ampSq3_3 + ampSq3_4 +
        ampSq4_1 + ampSq4_2 + ampSq4_3 + ampSq4_4
    ).real