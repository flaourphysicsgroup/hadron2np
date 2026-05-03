import numpy as np

def amp_square_3_1_1(
    wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1, m_dm2, qsq, theta1
) -> float:
    # 提取 wcs: 只有 S, P, V, A
    wc_S = wcs["S"]
    wc_P = wcs["P"]
    wc_V = wcs["V"]
    wc_A = wcs["A"]

    # 提取 ffs for B -> P (Scalar/Pseudoscalar meson): f0, f+, fT
    FFf0 = ffs["f0"]
    FFfp = ffs["f+"]
    FFft = ffs["fT"]

    # 初始化 16 个振幅平方项 (4x4 matrix elements squared/interference)
    # 注意：此处公式需要根据具体物理过程填充，目前设为 0
    ampSq1_1=4.*wc_S*wc_S.conjugate()*np.power(m_fq -m_iq,-2)*np.power(FFf0,2)*np.power(np.power(m_FS,2) -np.power(m_IS,2),2)
    ampSq1_2 = 0
    ampSq1_3=-4.*wc_S.conjugate()*wc_V*FFf0*np.power(m_fq -m_iq,-1)*(np.power(m_FS,2) -np.power(m_IS,2))*np.power(qsq,-1)*(FFf0*(np.power(m_dm1,2) -np.power(m_dm2,2))*(np.power(m_FS,2) -np.power(m_IS,2)) + FFfp*np.cos(theta1)*np.power(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq + np.power(m_dm2,2)) + np.power(qsq -np.power(m_dm2,2),2),0.5)*np.power(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2),0.5))
    ampSq1_4 = 0
    
    ampSq2_1 = 0
    ampSq2_2 = 0
    ampSq2_3 = 0
    ampSq2_4 = 0
    
    ampSq3_1=-4.*wc_S*wc_V.conjugate()*FFf0*np.power(m_fq -m_iq,-1)*(np.power(m_FS,2) -np.power(m_IS,2))*np.power(qsq,-1)*(FFf0*(np.power(m_dm1,2) -np.power(m_dm2,2))*(np.power(m_FS,2) -np.power(m_IS,2)) + FFfp*np.cos(theta1)*np.power(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq + np.power(m_dm2,2)) + np.power(qsq -np.power(m_dm2,2),2),0.5)*np.power(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2),0.5))
    ampSq3_2 = 0
    ampSq3_3=4.*wc_V*wc_V.conjugate()*np.power(qsq,-2)*np.power(FFf0*(np.power(m_dm1,2) -np.power(m_dm2,2))*(np.power(m_FS,2) -np.power(m_IS,2)) + FFfp*np.cos(theta1)*np.power(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq + np.power(m_dm2,2)) + np.power(qsq -np.power(m_dm2,2),2),0.5)*np.power(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2),0.5),2)
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