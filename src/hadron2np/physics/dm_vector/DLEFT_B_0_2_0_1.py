import numpy as np


def amp_square_2_0_1(wcs: dict, f_B, m_iq, m_fq, m_IS, m_dm1, m_dm2) -> float:
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

    ampSq1_1 = 0
    ampSq1_2 = 0
    ampSq1_3 = 0
    ampSq1_4 = 0

    ampSq2_1 = 0
    ampSq2_2 = 0
    ampSq2_3 = 0
    ampSq2_4 = 0

    ampSq3_1 = 0
    ampSq3_2 = 0
    ampSq3_3 = 0
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
