import numpy as np


def amp_square_2_1_1(wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1) -> float:
    wc_V = wcs['V']
    wc_T = wcs['T']
    FFfp = ffs['f+']
    FFft = ffs['fT']
    ampSq11 = (
        wc_V
        * wc_V.conjugate()
        * np.power(m_dm1, -2)
        * np.power(FFfp, 2)
        * (
            np.power(m_dm1, 4)
            - 2 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
            + np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
        )
    )
    ampSq13 = (
        2
        * wc_T
        * wc_V.conjugate()
        * FFfp
        * FFft
        * np.power(m_FS + m_IS, -1)
        * (
            np.power(m_dm1, 4)
            - 2 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
            + np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
        )
    )
    ampSq31 = (
        2
        * wc_T.conjugate()
        * wc_V
        * FFfp
        * FFft
        * np.power(m_FS + m_IS, -1)
        * (
            np.power(m_dm1, 4)
            - 2 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
            + np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
        )
    )
    ampSq33 = (
        4
        * wc_T
        * wc_T.conjugate()
        * np.power(m_dm1, 2)
        * np.power(m_FS + m_IS, -2)
        * np.power(FFft, 2)
        * (
            np.power(m_dm1, 4)
            - 2 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
            + np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
        )
    )
    return ampSq11 + ampSq13 + ampSq31 + ampSq33


def amp_square_3_1_1(
    wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1, m_dm2, qsq, theta1
) -> float:
    wc_S = wcs['S']
    wc_V = wcs['V']
    wc_T = wcs['T']
    wc_T5 = wcs['T5']
    wc_V_tilde = wcs['V_tilde']
    wc_V_tilde_21 = wcs['V_tilde_21']
    wc_V_partial = wcs['V_partial']
    wc_V_21 = wcs['V_21']
    FFf0 = ffs['f0']
    FFfp = ffs['f+']
    FFft = ffs['fT']

    ampSq1_1 = (
        wc_S
        * wc_S.conjugate()
        * np.power(m_dm1, -2)
        * np.power(m_dm2, -2)
        * np.power(m_fq - 1.0 * m_iq, -2)
        * np.power(FFf0, 2)
        * (
            np.power(m_dm1, 4)
            - 2.0 * np.power(m_dm1, 2) * (qsq - 5.0 * np.power(m_dm2, 2))
            + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
        )
        * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
    )
    ampSq1_3 = (
        -0.5
        * 1j
        * wc_S.conjugate()
        * wc_V
        * FFf0
        * np.power(m_dm2, -2)
        * np.power(m_fq - 1.0 * m_iq, -1)
        * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
        * np.power(qsq, -1)
        * (
            FFf0
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq - 2.0 * np.power(m_dm2, 2))
                + 4.0 * qsq * np.power(m_dm2, 2)
                - 5.0 * np.power(m_dm2, 4)
                + np.power(qsq, 2)
            )
            + FFfp
            * np.cos(theta1)
            * (-1.0 * qsq + np.power(m_dm1, 2) + 5.0 * np.power(m_dm2, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq1_7 = (
        1.0
        * 1j
        * (m_FS - 1.0 * m_IS)
        * wc_S.conjugate()
        * wc_T
        * FFf0
        * FFft
        * np.cos(theta1)
        * np.power(m_dm1, -2)
        * np.power(m_dm2, -2)
        * (-1.0 * qsq + np.power(m_dm1, 2) + np.power(m_dm2, 2))
        * np.power(m_fq - 1.0 * m_iq, -1)
        * np.power(
            np.power(m_dm1, 4)
            - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
            + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
            0.5,
        )
        * np.power(
            -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
            + np.power(qsq, 2)
            + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
            0.5,
        )
    )
    ampSq1_9 = (
        0.5
        * wc_S.conjugate()
        * wc_V_partial
        * FFf0
        * np.power(m_dm1, -2)
        * np.power(m_dm2, -2)
        * np.power(m_fq - 1.0 * m_iq, -1)
        * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
        * np.power(qsq, -2)
        * (
            FFf0
            * (
                np.power(m_dm1, 6)
                - 1.0 * np.power(m_dm1, 4) * (3.0 * qsq + np.power(m_dm2, 2))
                + np.power(m_dm1, 2)
                * (
                    -2.0 * qsq * np.power(m_dm2, 2)
                    - 1.0 * np.power(m_dm2, 4)
                    + 3.0 * np.power(qsq, 2)
                )
                - 1.0 * np.power(qsq - 1.0 * np.power(m_dm2, 2), 3)
            )
            * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            + 8.0
            * FFfp
            * qsq
            * np.power(m_dm1, 2)
            * np.power(m_dm2, 2)
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + FFfp
            * np.power(np.cos(theta1), 2)
            * (
                np.power(m_dm1, 6)
                - 1.0 * np.power(m_dm1, 4) * (qsq + np.power(m_dm2, 2))
                - 1.0
                * np.power(m_dm1, 2)
                * (
                    6.0 * qsq * np.power(m_dm2, 2)
                    + np.power(m_dm2, 4)
                    + np.power(qsq, 2)
                )
                + (qsq + np.power(m_dm2, 2))
                * np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + (FFf0 + FFfp)
            * np.cos(theta1)
            * (np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            * (-1.0 * qsq + np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq1_11 = (
        0.5
        * 1j
        * wc_S.conjugate()
        * wc_V_21
        * FFf0
        * np.power(m_dm1, -2)
        * np.power(m_fq - 1.0 * m_iq, -1)
        * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
        * np.power(qsq, -1)
        * (
            FFf0
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * (
                5.0 * np.power(m_dm1, 4)
                - 4.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                - 1.0 * np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            + FFfp
            * np.cos(theta1)
            * (-1.0 * qsq + 5.0 * np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq1 = ampSq1_1 + ampSq1_3 + ampSq1_7 + ampSq1_9 + ampSq1_11

    ampSq3_1 = (
        0.5
        * 1j
        * wc_S
        * wc_V.conjugate()
        * FFf0
        * np.power(m_dm2, -2)
        * np.power(m_fq - 1.0 * m_iq, -1)
        * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
        * np.power(qsq, -1)
        * (
            FFf0
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq - 2.0 * np.power(m_dm2, 2))
                + 4.0 * qsq * np.power(m_dm2, 2)
                - 5.0 * np.power(m_dm2, 4)
                + np.power(qsq, 2)
            )
            + FFfp
            * np.cos(theta1)
            * (-1.0 * qsq + np.power(m_dm1, 2) + 5.0 * np.power(m_dm2, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq3_3 = (
        0.25
        * wc_V
        * wc_V.conjugate()
        * np.power(m_dm2, -2)
        * np.power(qsq, -2)
        * (
            np.power(FFf0, 2)
            * (
                -2.0 * qsq * np.power(m_dm1, 4)
                + np.power(m_dm1, 6)
                + np.power(m_dm1, 2)
                * (
                    6.0 * qsq * np.power(m_dm2, 2)
                    - 3.0 * np.power(m_dm2, 4)
                    + np.power(qsq, 2)
                )
                + 2.0 * np.power(-1.0 * m_dm2 * qsq + np.power(m_dm2, 3), 2)
            )
            * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            + qsq
            * np.power(FFfp, 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + np.power(FFfp, 2)
            * np.power(np.cos(theta1), 2)
            * (
                -3.0 * qsq * np.power(m_dm1, 4)
                + np.power(m_dm1, 6)
                + np.power(m_dm1, 2)
                * (
                    -4.0 * qsq * np.power(m_dm2, 2)
                    - 3.0 * np.power(m_dm2, 4)
                    + 3.0 * np.power(qsq, 2)
                )
                - 1.0
                * (qsq - 2.0 * np.power(m_dm2, 2))
                * np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + 2.0
            * FFf0
            * FFfp
            * np.cos(theta1)
            * (
                np.power(m_dm1, 4)
                + 2.0 * (qsq - 1.0 * np.power(m_dm2, 2)) * np.power(m_dm2, 2)
                + np.power(m_dm1, 2) * (-1.0 * qsq + np.power(m_dm2, 2))
            )
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq3_7 = (
        -0.5
        * wc_T
        * wc_V.conjugate()
        * FFft
        * np.power(m_dm2, -2)
        * np.power(m_FS + m_IS, -1)
        * np.power(qsq, -1)
        * (
            FFfp
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + FFf0
            * np.cos(theta1)
            * (-1.0 * qsq + np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq3_9 = (
        0.25
        * 1j
        * wc_V.conjugate()
        * wc_V_partial
        * np.power(m_dm2, -2)
        * np.power(qsq, -3)
        * (
            FFfp
            * (
                2.0 * FFf0 * (np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
                + FFfp * (-1.0 * qsq + np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            )
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(np.cos(theta1), 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * (
                np.power(FFf0, 2)
                * (
                    np.power(m_dm1, 6)
                    - 3.0 * np.power(m_dm1, 4) * (qsq + np.power(m_dm2, 2))
                    + np.power(m_dm1, 2)
                    * (
                        2.0 * qsq * np.power(m_dm2, 2)
                        + 3.0 * np.power(m_dm2, 4)
                        + 3.0 * np.power(qsq, 2)
                    )
                    - 1.0
                    * (qsq + np.power(m_dm2, 2))
                    * np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
                )
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + FFf0
                * FFfp
                * qsq
                * (
                    np.power(m_dm1, 4)
                    - 2.0 * np.power(m_dm1, 2) * (qsq - 1.0 * np.power(m_dm2, 2))
                    + 2.0 * qsq * np.power(m_dm2, 2)
                    - 3.0 * np.power(m_dm2, 4)
                    + np.power(qsq, 2)
                )
                * (
                    -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(qsq, 2)
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
                + qsq
                * np.power(FFfp, 2)
                * (
                    np.power(m_dm1, 4)
                    - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                    + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
                )
                * (
                    -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(qsq, 2)
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
            )
            + np.cos(theta1)
            * (
                (np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
                * (-1.0 * qsq + np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
                * np.power(FFf0, 2)
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + FFf0
                * FFfp
                * (
                    2.0 * np.power(m_dm1, 4)
                    - 1.0 * qsq * np.power(m_dm2, 2)
                    - 1.0 * np.power(m_dm1, 2) * (3.0 * qsq + 4.0 * np.power(m_dm2, 2))
                    + 2.0 * np.power(m_dm2, 4)
                    + np.power(qsq, 2)
                )
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + 2.0
                * qsq
                * (-1.0 * qsq + np.power(m_dm1, 2) + np.power(m_dm2, 2))
                * np.power(FFfp, 2)
                * (
                    -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(qsq, 2)
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
            )
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
            + np.power(FFfp, 2)
            * np.power(np.cos(theta1), 3)
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                1.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                1.5,
            )
        )
    )
    ampSq3_11 = (
        -0.75
        * wc_V_21
        * wc_V.conjugate()
        * np.power(qsq, -2)
        * (
            FFf0
            * (-1.0 * qsq + np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            + FFfp
            * np.cos(theta1)
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
        * (
            FFf0
            * (qsq + np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            + FFfp
            * np.cos(theta1)
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq3 = ampSq3_1 + ampSq3_3 + ampSq3_7 + ampSq3_9 + ampSq3_11

    ampSq5_5 = (
        0.25
        * wc_V_tilde
        * wc_V_tilde.conjugate()
        * np.power(m_dm2, -2)
        * np.power(qsq, -2)
        * (
            2.0
            * np.power(m_dm2, 2)
            * np.power(FFf0, 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            + qsq
            * np.power(FFfp, 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq - 3.0 * np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            - 1.0
            * (qsq - 2.0 * np.power(m_dm2, 2))
            * np.power(FFfp, 2)
            * np.power(np.cos(theta1), 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + 4.0
            * FFf0
            * FFfp
            * np.cos(theta1)
            * (qsq + np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            * np.power(m_dm2, 2)
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq5_8 = (
        0.5
        * 1j
        * wc_T5
        * wc_V_tilde.conjugate()
        * FFft
        * np.power(m_dm2, -2)
        * np.power(m_FS + m_IS, -1)
        * np.power(qsq, -1)
        * (
            -1.0
            * FFfp
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq - 1.0 * np.power(m_dm2, 2))
                + 2.0 * qsq * np.power(m_dm2, 2)
                - 3.0 * np.power(m_dm2, 4)
                + np.power(qsq, 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + FFfp
            * np.power(np.cos(theta1), 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + 4.0
            * FFf0
            * np.cos(theta1)
            * np.power(m_dm2, 2)
            * (-1.0 * np.power(m_FS, 2) + np.power(m_IS, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq5_13 = (
        0.5
        * wc_V_tilde_21
        * wc_V_tilde.conjugate()
        * np.power(qsq, -2)
        * (
            np.power(FFf0, 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            + 2.0
            * qsq
            * (-1.0 * qsq + np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * np.power(FFfp, 2)
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + np.power(FFfp, 2)
            * np.power(np.cos(theta1), 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + 2.0
            * FFf0
            * FFfp
            * np.cos(theta1)
            * (np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq5 = ampSq5_5 + ampSq5_8 + ampSq5_13

    ampSq7_1 = (
        -1.0
        * 1j
        * (m_FS - 1.0 * m_IS)
        * wc_S
        * wc_T.conjugate()
        * FFf0
        * FFft
        * np.cos(theta1)
        * np.power(m_dm1, -2)
        * np.power(m_dm2, -2)
        * (-1.0 * qsq + np.power(m_dm1, 2) + np.power(m_dm2, 2))
        * np.power(m_fq - 1.0 * m_iq, -1)
        * np.power(
            np.power(m_dm1, 4)
            - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
            + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
            0.5,
        )
        * np.power(
            -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
            + np.power(qsq, 2)
            + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
            0.5,
        )
    )
    ampSq7_3 = (
        -0.5
        * wc_T.conjugate()
        * wc_V
        * FFft
        * np.power(m_dm2, -2)
        * np.power(m_FS + m_IS, -1)
        * np.power(qsq, -1)
        * (
            FFfp
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + FFf0
            * np.cos(theta1)
            * (-1.0 * qsq + np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq7_7 = (
        -1.0
        * wc_T
        * wc_T.conjugate()
        * np.power(m_dm1, -2)
        * np.power(m_dm2, -2)
        * np.power(m_FS + m_IS, -2)
        * np.power(FFft, 2)
        * np.power(qsq, -1)
        * (
            -1.0 * np.power(m_dm1, 2)
            - 1.0 * np.power(m_dm2, 2)
            + (-1.0 * qsq + np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * np.power(np.cos(theta1), 2)
        )
        * (
            np.power(m_dm1, 4)
            - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
            + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
        )
        * (
            -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
            + np.power(qsq, 2)
            + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
        )
    )
    ampSq7_9 = (
        -0.5
        * 1j
        * wc_T.conjugate()
        * wc_V_partial
        * FFft
        * np.power(m_dm1, -2)
        * np.power(m_dm2, -2)
        * np.power(m_FS + m_IS, -1)
        * np.power(qsq, -2)
        * (
            (FFf0 + FFfp)
            * (np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + np.cos(theta1)
            * (
                FFf0
                * (
                    np.power(m_dm1, 4)
                    - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                    + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
                )
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + 2.0
                * FFfp
                * (
                    np.power(m_dm1, 4)
                    - 1.0 * qsq * np.power(m_dm2, 2)
                    - 1.0 * np.power(m_dm1, 2) * (qsq + 2.0 * np.power(m_dm2, 2))
                    + np.power(m_dm2, 4)
                )
                * (
                    -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(qsq, 2)
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
            )
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
            - 1.0
            * FFfp
            * np.power(np.cos(theta1), 3)
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                1.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                1.5,
            )
        )
    )
    ampSq7_11 = (
        0.5
        * wc_T.conjugate()
        * wc_V_21
        * FFft
        * np.power(m_dm1, -2)
        * np.power(m_FS + m_IS, -1)
        * np.power(qsq, -1)
        * (
            FFfp
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + FFf0
            * np.cos(theta1)
            * (qsq + np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq7 = ampSq7_1 + ampSq7_3 + ampSq7_7 + ampSq7_9 + ampSq7_11

    ampSq8_5 = (
        -0.5
        * 1j
        * wc_T5.conjugate()
        * wc_V_tilde
        * FFft
        * np.power(m_dm2, -2)
        * np.power(m_FS + m_IS, -1)
        * np.power(qsq, -1)
        * (
            -1.0
            * FFfp
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq - 1.0 * np.power(m_dm2, 2))
                + 2.0 * qsq * np.power(m_dm2, 2)
                - 3.0 * np.power(m_dm2, 4)
                + np.power(qsq, 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + FFfp
            * np.power(np.cos(theta1), 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + 4.0
            * FFf0
            * np.cos(theta1)
            * np.power(m_dm2, 2)
            * (-1.0 * np.power(m_FS, 2) + np.power(m_IS, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq8_8 = (
        -1.0
        * wc_T5
        * wc_T5.conjugate()
        * np.power(m_dm1, -2)
        * np.power(m_dm2, -2)
        * np.power(m_FS + m_IS, -2)
        * np.power(FFft, 2)
        * np.power(qsq, -1)
        * (
            -1.0 * np.power(m_dm1, 6)
            + np.power(m_dm1, 4) * (2.0 * qsq + np.power(m_dm2, 2))
            + np.power(m_dm1, 2)
            * (
                -4.0 * qsq * np.power(m_dm2, 2)
                + np.power(m_dm2, 4)
                - 1.0 * np.power(qsq, 2)
            )
            + (np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * np.power(np.cos(theta1), 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            - 1.0 * np.power(-1.0 * m_dm2 * qsq + np.power(m_dm2, 3), 2)
        )
        * (
            -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
            + np.power(qsq, 2)
            + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
        )
    )
    ampSq8_13 = (
        0.5
        * 1j
        * wc_T5.conjugate()
        * wc_V_tilde_21
        * FFft
        * np.power(m_dm1, -2)
        * np.power(m_FS + m_IS, -1)
        * np.power(qsq, -1)
        * (
            FFfp
            * (
                3.0 * np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                - 1.0 * np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + FFfp
            * np.power(np.cos(theta1), 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + 4.0
            * FFf0
            * np.cos(theta1)
            * np.power(m_dm1, 2)
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq8 = ampSq8_5 + ampSq8_8 + ampSq8_13

    ampSq9_1 = (
        0.5
        * wc_S
        * wc_V_partial.conjugate()
        * FFf0
        * np.power(m_dm1, -2)
        * np.power(m_dm2, -2)
        * np.power(m_fq - 1.0 * m_iq, -1)
        * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
        * np.power(qsq, -2)
        * (
            FFf0
            * (
                np.power(m_dm1, 6)
                - 1.0 * np.power(m_dm1, 4) * (3.0 * qsq + np.power(m_dm2, 2))
                + np.power(m_dm1, 2)
                * (
                    -2.0 * qsq * np.power(m_dm2, 2)
                    - 1.0 * np.power(m_dm2, 4)
                    + 3.0 * np.power(qsq, 2)
                )
                - 1.0 * np.power(qsq - 1.0 * np.power(m_dm2, 2), 3)
            )
            * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            + 8.0
            * FFfp
            * qsq
            * np.power(m_dm1, 2)
            * np.power(m_dm2, 2)
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + FFfp
            * np.power(np.cos(theta1), 2)
            * (
                np.power(m_dm1, 6)
                - 1.0 * np.power(m_dm1, 4) * (qsq + np.power(m_dm2, 2))
                - 1.0
                * np.power(m_dm1, 2)
                * (
                    6.0 * qsq * np.power(m_dm2, 2)
                    + np.power(m_dm2, 4)
                    + np.power(qsq, 2)
                )
                + (qsq + np.power(m_dm2, 2))
                * np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + (FFf0 + FFfp)
            * np.cos(theta1)
            * (np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            * (-1.0 * qsq + np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq9_3 = (
        -0.25
        * 1j
        * wc_V
        * wc_V_partial.conjugate()
        * np.power(m_dm2, -2)
        * np.power(qsq, -3)
        * (
            FFfp
            * (
                2.0 * FFf0 * (np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
                + FFfp * (-1.0 * qsq + np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            )
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(np.cos(theta1), 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * (
                np.power(FFf0, 2)
                * (
                    np.power(m_dm1, 6)
                    - 3.0 * np.power(m_dm1, 4) * (qsq + np.power(m_dm2, 2))
                    + np.power(m_dm1, 2)
                    * (
                        2.0 * qsq * np.power(m_dm2, 2)
                        + 3.0 * np.power(m_dm2, 4)
                        + 3.0 * np.power(qsq, 2)
                    )
                    - 1.0
                    * (qsq + np.power(m_dm2, 2))
                    * np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
                )
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + FFf0
                * FFfp
                * qsq
                * (
                    np.power(m_dm1, 4)
                    - 2.0 * np.power(m_dm1, 2) * (qsq - 1.0 * np.power(m_dm2, 2))
                    + 2.0 * qsq * np.power(m_dm2, 2)
                    - 3.0 * np.power(m_dm2, 4)
                    + np.power(qsq, 2)
                )
                * (
                    -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(qsq, 2)
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
                + qsq
                * np.power(FFfp, 2)
                * (
                    np.power(m_dm1, 4)
                    - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                    + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
                )
                * (
                    -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(qsq, 2)
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
            )
            + np.cos(theta1)
            * (
                (np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
                * (-1.0 * qsq + np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
                * np.power(FFf0, 2)
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + FFf0
                * FFfp
                * (
                    2.0 * np.power(m_dm1, 4)
                    - 1.0 * qsq * np.power(m_dm2, 2)
                    - 1.0 * np.power(m_dm1, 2) * (3.0 * qsq + 4.0 * np.power(m_dm2, 2))
                    + 2.0 * np.power(m_dm2, 4)
                    + np.power(qsq, 2)
                )
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + 2.0
                * qsq
                * (-1.0 * qsq + np.power(m_dm1, 2) + np.power(m_dm2, 2))
                * np.power(FFfp, 2)
                * (
                    -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(qsq, 2)
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
            )
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
            + np.power(FFfp, 2)
            * np.power(np.cos(theta1), 3)
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                1.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                1.5,
            )
        )
    )
    ampSq9_7 = (
        0.5
        * 1j
        * wc_T
        * wc_V_partial.conjugate()
        * FFft
        * np.power(m_dm1, -2)
        * np.power(m_dm2, -2)
        * np.power(m_FS + m_IS, -1)
        * np.power(qsq, -2)
        * (
            (FFf0 + FFfp)
            * (np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + np.cos(theta1)
            * (
                FFf0
                * (
                    np.power(m_dm1, 4)
                    - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                    + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
                )
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + 2.0
                * FFfp
                * (
                    np.power(m_dm1, 4)
                    - 1.0 * qsq * np.power(m_dm2, 2)
                    - 1.0 * np.power(m_dm1, 2) * (qsq + 2.0 * np.power(m_dm2, 2))
                    + np.power(m_dm2, 4)
                )
                * (
                    -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(qsq, 2)
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
            )
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
            - 1.0
            * FFfp
            * np.power(np.cos(theta1), 3)
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                1.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                1.5,
            )
        )
    )
    ampSq9_9 = (
        0.25
        * wc_V_partial
        * wc_V_partial.conjugate()
        * np.power(m_dm1, -2)
        * np.power(m_dm2, -2)
        * np.power(qsq, -4)
        * (
            2.0
            * FFf0
            * FFfp
            * qsq
            * (np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + np.power(FFf0, 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            * (
                np.power(m_dm2, 2) * np.power(qsq, 3)
                + np.power(m_dm1, 4)
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                - 1.0
                * qsq
                * np.power(m_dm2, 2)
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + np.power(m_dm2, 4)
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + np.power(qsq, 2)
                * (
                    -2.0 * np.power(m_dm2, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
                + np.power(m_dm1, 2)
                * (
                    -2.0 * (np.power(m_FS, 2) + np.power(m_IS, 2)) * np.power(qsq, 2)
                    + np.power(qsq, 3)
                    - 1.0
                    * qsq
                    * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                    - 2.0
                    * np.power(m_dm2, 2)
                    * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
            )
            + qsq
            * np.power(FFfp, 2)
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            * (
                np.power(m_dm1, 6)
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                - 1.0
                * np.power(m_dm1, 4)
                * (2.0 * qsq + np.power(m_dm2, 2))
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + np.power(m_dm2, 2)
                * np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + np.power(m_dm1, 2)
                * (
                    16.0 * np.power(m_dm2, 2) * np.power(qsq, 3)
                    + 12.0
                    * qsq
                    * np.power(m_dm2, 2)
                    * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                    - 1.0
                    * np.power(m_dm2, 4)
                    * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                    + np.power(qsq, 2)
                    * (
                        -32.0
                        * np.power(m_dm2, 2)
                        * (np.power(m_FS, 2) + np.power(m_IS, 2))
                        + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                    )
                )
            )
            + np.power(np.cos(theta1), 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            * (
                (
                    np.power(m_dm1, 4)
                    - 1.0 * qsq * np.power(m_dm2, 2)
                    - 1.0 * np.power(m_dm1, 2) * (qsq + 2.0 * np.power(m_dm2, 2))
                    + np.power(m_dm2, 4)
                )
                * np.power(FFf0, 2)
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + 2.0
                * FFf0
                * FFfp
                * (
                    2.0 * np.power(m_dm1, 4)
                    - 1.0 * qsq * np.power(m_dm2, 2)
                    - 1.0 * np.power(m_dm1, 2) * (qsq + 4.0 * np.power(m_dm2, 2))
                    + 2.0 * np.power(m_dm2, 4)
                    - 1.0 * np.power(qsq, 2)
                )
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + np.power(FFfp, 2)
                * (
                    np.power(m_dm1, 4)
                    * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                    + np.power(m_dm1, 2)
                    * (
                        -8.0
                        * (np.power(m_FS, 2) + np.power(m_IS, 2))
                        * np.power(qsq, 2)
                        + 4.0 * np.power(qsq, 3)
                        + 3.0
                        * qsq
                        * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                        - 2.0
                        * np.power(m_dm2, 2)
                        * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                    )
                    + np.power(m_dm2, 2)
                    * (
                        -8.0
                        * (np.power(m_FS, 2) + np.power(m_IS, 2))
                        * np.power(qsq, 2)
                        + 4.0 * np.power(qsq, 3)
                        + 3.0
                        * qsq
                        * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                        + np.power(m_dm2, 2)
                        * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                    )
                )
            )
            + 2.0
            * (FFf0 + FFfp)
            * np.cos(theta1)
            * (np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * (
                FFf0
                * (
                    np.power(m_dm1, 4)
                    - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                    + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
                )
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + 2.0
                * FFfp
                * qsq
                * (-1.0 * qsq + np.power(m_dm1, 2) + np.power(m_dm2, 2))
                * (
                    -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(qsq, 2)
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
            )
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
            + 2.0
            * FFfp
            * (FFf0 + FFfp)
            * (np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(np.cos(theta1), 3)
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                1.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                1.5,
            )
            + np.power(FFfp, 2)
            * np.power(np.cos(theta1), 4)
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                2,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                2,
            )
        )
    )
    ampSq9_11 = (
        0.25
        * 1j
        * wc_V_21
        * wc_V_partial.conjugate()
        * np.power(m_dm1, -2)
        * np.power(qsq, -3)
        * (
            FFfp
            * (
                2.0 * FFf0 * (np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
                + FFfp * (qsq + np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            )
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(np.cos(theta1), 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * (
                np.power(FFf0, 2)
                * (
                    np.power(m_dm1, 6)
                    - 1.0 * np.power(m_dm1, 4) * (qsq + 3.0 * np.power(m_dm2, 2))
                    - 1.0
                    * np.power(m_dm1, 2)
                    * (
                        2.0 * qsq * np.power(m_dm2, 2)
                        - 3.0 * np.power(m_dm2, 4)
                        + np.power(qsq, 2)
                    )
                    + np.power(qsq - 1.0 * np.power(m_dm2, 2), 3)
                )
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                - 1.0
                * qsq
                * np.power(FFfp, 2)
                * (
                    np.power(m_dm1, 4)
                    - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                    + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
                )
                * (
                    -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(qsq, 2)
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
                - 1.0
                * FFf0
                * FFfp
                * qsq
                * (
                    -3.0 * np.power(m_dm1, 4)
                    + 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                    + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
                )
                * (
                    -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(qsq, 2)
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
            )
            + np.cos(theta1)
            * (
                (np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
                * (qsq + np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
                * np.power(FFf0, 2)
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + FFf0
                * FFfp
                * (
                    2.0 * np.power(m_dm1, 4)
                    - 3.0 * qsq * np.power(m_dm2, 2)
                    - 1.0 * np.power(m_dm1, 2) * (qsq + 4.0 * np.power(m_dm2, 2))
                    + 2.0 * np.power(m_dm2, 4)
                    + np.power(qsq, 2)
                )
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + 2.0
                * qsq
                * (-1.0 * qsq + np.power(m_dm1, 2) + np.power(m_dm2, 2))
                * np.power(FFfp, 2)
                * (
                    -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(qsq, 2)
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
            )
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
            + np.power(FFfp, 2)
            * np.power(np.cos(theta1), 3)
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                1.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                1.5,
            )
        )
    )
    ampSq9 = ampSq9_1 + ampSq9_3 + ampSq9_7 + ampSq9_9 + ampSq9_11

    ampSq11_1 = (
        -0.5
        * 1j
        * wc_S
        * wc_V_21.conjugate()
        * FFf0
        * np.power(m_dm1, -2)
        * np.power(m_fq - 1.0 * m_iq, -1)
        * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
        * np.power(qsq, -1)
        * (
            FFf0
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * (
                5.0 * np.power(m_dm1, 4)
                - 4.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                - 1.0 * np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            + FFfp
            * np.cos(theta1)
            * (-1.0 * qsq + 5.0 * np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq11_3 = (
        -0.75
        * wc_V
        * wc_V_21.conjugate()
        * np.power(qsq, -2)
        * (
            FFf0
            * (-1.0 * qsq + np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            + FFfp
            * np.cos(theta1)
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
        * (
            FFf0
            * (qsq + np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            + FFfp
            * np.cos(theta1)
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq11_7 = (
        0.5
        * wc_T
        * wc_V_21.conjugate()
        * FFft
        * np.power(m_dm1, -2)
        * np.power(m_FS + m_IS, -1)
        * np.power(qsq, -1)
        * (
            FFfp
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + FFf0
            * np.cos(theta1)
            * (qsq + np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq11_9 = (
        -0.25
        * 1j
        * wc_V_21.conjugate()
        * wc_V_partial
        * np.power(m_dm1, -2)
        * np.power(qsq, -3)
        * (
            FFfp
            * (
                2.0 * FFf0 * (np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
                + FFfp * (qsq + np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            )
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(np.cos(theta1), 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * (
                np.power(FFf0, 2)
                * (
                    np.power(m_dm1, 6)
                    - 1.0 * np.power(m_dm1, 4) * (qsq + 3.0 * np.power(m_dm2, 2))
                    - 1.0
                    * np.power(m_dm1, 2)
                    * (
                        2.0 * qsq * np.power(m_dm2, 2)
                        - 3.0 * np.power(m_dm2, 4)
                        + np.power(qsq, 2)
                    )
                    + np.power(qsq - 1.0 * np.power(m_dm2, 2), 3)
                )
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                - 1.0
                * qsq
                * np.power(FFfp, 2)
                * (
                    np.power(m_dm1, 4)
                    - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                    + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
                )
                * (
                    -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(qsq, 2)
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
                - 1.0
                * FFf0
                * FFfp
                * qsq
                * (
                    -3.0 * np.power(m_dm1, 4)
                    + 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                    + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
                )
                * (
                    -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(qsq, 2)
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
            )
            + np.cos(theta1)
            * (
                (np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
                * (qsq + np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
                * np.power(FFf0, 2)
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + FFf0
                * FFfp
                * (
                    2.0 * np.power(m_dm1, 4)
                    - 3.0 * qsq * np.power(m_dm2, 2)
                    - 1.0 * np.power(m_dm1, 2) * (qsq + 4.0 * np.power(m_dm2, 2))
                    + 2.0 * np.power(m_dm2, 4)
                    + np.power(qsq, 2)
                )
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                + 2.0
                * qsq
                * (-1.0 * qsq + np.power(m_dm1, 2) + np.power(m_dm2, 2))
                * np.power(FFfp, 2)
                * (
                    -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(qsq, 2)
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
            )
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
            + np.power(FFfp, 2)
            * np.power(np.cos(theta1), 3)
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                1.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                1.5,
            )
        )
    )
    ampSq11_11 = (
        0.25
        * wc_V_21
        * wc_V_21.conjugate()
        * np.power(m_dm1, -2)
        * np.power(qsq, -2)
        * (
            np.power(FFf0, 2)
            * (
                2.0 * np.power(m_dm1, 6)
                - 1.0 * np.power(m_dm1, 4) * (4.0 * qsq + 3.0 * np.power(m_dm2, 2))
                + 2.0
                * np.power(m_dm1, 2)
                * (3.0 * qsq * np.power(m_dm2, 2) + np.power(qsq, 2))
                + np.power(-1.0 * m_dm2 * qsq + np.power(m_dm2, 3), 2)
            )
            * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            + qsq
            * np.power(FFfp, 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + np.power(FFfp, 2)
            * np.power(np.cos(theta1), 2)
            * (
                2.0 * np.power(m_dm1, 6)
                - 1.0 * np.power(m_dm1, 4) * (5.0 * qsq + 3.0 * np.power(m_dm2, 2))
                + 4.0
                * np.power(m_dm1, 2)
                * (-1.0 * qsq * np.power(m_dm2, 2) + np.power(qsq, 2))
                - 1.0 * np.power(qsq - 1.0 * np.power(m_dm2, 2), 3)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + 2.0
            * FFf0
            * FFfp
            * np.cos(theta1)
            * (
                2.0 * np.power(m_dm1, 4)
                + (qsq - 1.0 * np.power(m_dm2, 2)) * np.power(m_dm2, 2)
                - 1.0 * np.power(m_dm1, 2) * (2.0 * qsq + np.power(m_dm2, 2))
            )
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq11 = ampSq11_1 + ampSq11_3 + ampSq11_7 + ampSq11_9 + ampSq11_11

    ampSq13_5 = (
        0.5
        * wc_V_tilde
        * wc_V_tilde_21.conjugate()
        * np.power(qsq, -2)
        * (
            np.power(FFf0, 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            + 2.0
            * qsq
            * (-1.0 * qsq + np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * np.power(FFfp, 2)
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + np.power(FFfp, 2)
            * np.power(np.cos(theta1), 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + 2.0
            * FFf0
            * FFfp
            * np.cos(theta1)
            * (np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq13_8 = (
        -0.5
        * 1j
        * wc_T5
        * wc_V_tilde_21.conjugate()
        * FFft
        * np.power(m_dm1, -2)
        * np.power(m_FS + m_IS, -1)
        * np.power(qsq, -1)
        * (
            FFfp
            * (
                3.0 * np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                - 1.0 * np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + FFfp
            * np.power(np.cos(theta1), 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + 4.0
            * FFf0
            * np.cos(theta1)
            * np.power(m_dm1, 2)
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq13_13 = (
        0.25
        * wc_V_tilde_21
        * wc_V_tilde_21.conjugate()
        * np.power(m_dm1, -2)
        * np.power(qsq, -2)
        * (
            2.0
            * np.power(m_dm1, 2)
            * np.power(FFf0, 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            + qsq
            * np.power(FFfp, 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq - 3.0 * np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + (-1.0 * qsq + 2.0 * np.power(m_dm1, 2))
            * np.power(FFfp, 2)
            * np.power(np.cos(theta1), 2)
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2)
            )
            * (
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + 4.0
            * FFf0
            * FFfp
            * np.cos(theta1)
            * np.power(m_dm1, 2)
            * (-1.0 * qsq + np.power(m_dm1, 2) - 1.0 * np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - 1.0 * np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2.0 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq13 = ampSq13_5 + ampSq13_8 + ampSq13_13
    return ampSq1 + ampSq3 + ampSq5 + ampSq7 + ampSq8 + ampSq9 + ampSq11 + ampSq13
