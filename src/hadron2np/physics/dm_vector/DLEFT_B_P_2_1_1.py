import numpy as np


def amp_square_2_1_1(wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1) -> float:
    wc_V = wcs["V"]
    wc_A = wcs["A"]
    wc_T = wcs["T"]
    wc_T5 = wcs["T5"]

    # 提取 ffs for B -> P (Scalar/Pseudoscalar meson): f0, f+, fT
    FFf0 = ffs["f0"]
    FFfp = ffs["f+"]
    FFft = ffs["fT"]

    # 初始化 4x4 = 16 个振幅平方项
    # 注意：此处公式需要根据具体物理过程填充，目前设为 0
    ampSq1_1 = wc_V*wc_V.conjugate()*np.power(m_dm1,-2)*np.power(FFfp,2)*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))
    ampSq1_2 = 0
    ampSq1_3 = 2.*wc_T*wc_V.conjugate()*FFfp*FFft*np.power(m_FS + m_IS,-1)*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))
    ampSq1_4 = 0

    ampSq2_1 = 0
    ampSq2_2 = 0
    ampSq2_3 = 0
    ampSq2_4 = 0

    ampSq3_1 = 2.*wc_T.conjugate()*wc_V*FFfp*FFft*np.power(m_FS + m_IS,-1)*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))
    ampSq3_2 = 0
    ampSq3_3 = 4.*wc_T*wc_T.conjugate()*np.power(m_dm1,2)*np.power(m_FS + m_IS,-2)*np.power(FFft,2)*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))
    ampSq3_4 = 0

    ampSq4_1 = 0
    ampSq4_2 = 0
    ampSq4_3 = 0
    ampSq4_4 = 0

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