import numpy as np


def amp_square_2_1_1(wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1) -> float:
    wc_V = wcs['V']
    wc_A = wcs['A']
    wc_T = wcs['T']
    wc_T5 = wcs['T5']
    FFV = ffs['V']
    FFA1 = ffs['A1']
    FFA2 = ffs['A2']
    FFT1 = ffs['T1']
    FFT2 = ffs['T2']
    FFT3 = ffs['T3']
    ampSq1_1 = (
        2.0
        * wc_V
        * wc_V.conjugate()
        * np.power(m_FS + m_IS, -2)
        * np.power(FFV, 2)
        * (
            np.power(m_dm1, 4)
            - 2.0 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
            + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
        )
    )
    ampSq1_3 = (
        -4.0
        * wc_T
        * wc_V.conjugate()
        * FFT1
        * FFV
        * np.power(m_FS + m_IS, -1)
        * (
            np.power(m_dm1, 4)
            - 2.0 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
            + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
        )
    )
    ampSq2_2 = (
        0.25
        * wc_A
        * wc_A.conjugate()
        * np.power(m_dm1, -2)
        * np.power(m_FS, -2)
        * np.power(m_FS + m_IS, -2)
        * (
            np.power(m_FS + m_IS, 4)
            * np.power(FFA1, 2)
            * (
                np.power(m_dm1, 4)
                + 2.0
                * np.power(m_dm1, 2)
                * (5.0 * np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + 2.0
            * FFA1
            * FFA2
            * np.power(m_FS + m_IS, 2)
            * (
                np.power(m_dm1, 6)
                - 1.0
                * np.power(m_dm1, 4)
                * (np.power(m_FS, 2) + 3.0 * np.power(m_IS, 2))
                - 1.0
                * np.power(m_dm1, 2)
                * (
                    np.power(m_FS, 4)
                    + 2.0 * np.power(m_FS, 2) * np.power(m_IS, 2)
                    - 3.0 * np.power(m_IS, 4)
                )
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 3)
            )
            + np.power(FFA2, 2)
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                2,
            )
        )
    )
    ampSq2_4 = (
        0.5
        * 1j
        * wc_A.conjugate()
        * wc_T5
        * np.power(m_FS, -2)
        * np.power(m_FS - 1.0 * m_IS, -1)
        * np.power(m_FS + m_IS, -2)
        * (
            FFA2
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            * (
                -1.0
                * FFT2
                * (
                    np.power(m_dm1, 2)
                    - 3.0 * np.power(m_FS, 2)
                    - 1.0 * np.power(m_IS, 2)
                )
                * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
                + FFT3
                * (
                    np.power(m_dm1, 4)
                    - 2.0 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
            )
            + FFA1
            * np.power(m_FS + m_IS, 2)
            * (
                -1.0
                * FFT2
                * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
                * (
                    np.power(m_dm1, 4)
                    - 11.0 * np.power(m_FS, 4)
                    + 10.0 * np.power(m_FS, 2) * np.power(m_IS, 2)
                    - 2.0 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(m_IS, 4)
                )
                + FFT3
                * (
                    np.power(m_dm1, 6)
                    - 1.0
                    * np.power(m_dm1, 4)
                    * (np.power(m_FS, 2) + 3.0 * np.power(m_IS, 2))
                    - 1.0
                    * np.power(m_dm1, 2)
                    * (
                        np.power(m_FS, 4)
                        + 2.0 * np.power(m_FS, 2) * np.power(m_IS, 2)
                        - 3.0 * np.power(m_IS, 4)
                    )
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 3)
                )
            )
        )
    )
    ampSq3_1 = (
        -4.0
        * wc_T.conjugate()
        * wc_V
        * FFT1
        * FFV
        * np.power(m_FS + m_IS, -1)
        * (
            np.power(m_dm1, 4)
            - 2.0 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
            + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
        )
    )
    ampSq3_3 = (
        8.0
        * wc_T
        * wc_T.conjugate()
        * np.power(FFT1, 2)
        * (
            np.power(m_dm1, 4)
            - 2.0 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
            + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
        )
    )
    ampSq4_2 = (
        -0.5
        * 1j
        * wc_A
        * wc_T5.conjugate()
        * np.power(m_FS, -2)
        * np.power(m_FS - 1.0 * m_IS, -1)
        * np.power(m_FS + m_IS, -2)
        * (
            FFA2
            * (
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            * (
                -1.0
                * FFT2
                * (
                    np.power(m_dm1, 2)
                    - 3.0 * np.power(m_FS, 2)
                    - 1.0 * np.power(m_IS, 2)
                )
                * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
                + FFT3
                * (
                    np.power(m_dm1, 4)
                    - 2.0 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
                )
            )
            + FFA1
            * np.power(m_FS + m_IS, 2)
            * (
                -1.0
                * FFT2
                * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
                * (
                    np.power(m_dm1, 4)
                    - 11.0 * np.power(m_FS, 4)
                    + 10.0 * np.power(m_FS, 2) * np.power(m_IS, 2)
                    - 2.0 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(m_IS, 4)
                )
                + FFT3
                * (
                    np.power(m_dm1, 6)
                    - 1.0
                    * np.power(m_dm1, 4)
                    * (np.power(m_FS, 2) + 3.0 * np.power(m_IS, 2))
                    - 1.0
                    * np.power(m_dm1, 2)
                    * (
                        np.power(m_FS, 4)
                        + 2.0 * np.power(m_FS, 2) * np.power(m_IS, 2)
                        - 3.0 * np.power(m_IS, 4)
                    )
                    + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 3)
                )
            )
        )
    )
    ampSq4_4 = (
        wc_T5
        * wc_T5.conjugate()
        * np.power(np.power(m_FS, 3) - 1.0 * m_FS * np.power(m_IS, 2), -2)
        * (
            -2.0
            * FFT2
            * FFT3
            * np.power(m_dm1, 2)
            * (np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2))
            * (
                np.power(m_dm1, 6)
                - 1.0
                * np.power(m_dm1, 4)
                * (5.0 * np.power(m_FS, 2) + 3.0 * np.power(m_IS, 2))
                + np.power(m_dm1, 2)
                * (
                    7.0 * np.power(m_FS, 4)
                    + 6.0 * np.power(m_FS, 2) * np.power(m_IS, 2)
                    + 3.0 * np.power(m_IS, 4)
                )
                - 1.0
                * (3.0 * np.power(m_FS, 2) + np.power(m_IS, 2))
                * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            )
            + np.power(FFT2, 2)
            * np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2)
            * (
                np.power(m_dm1, 6)
                - 2.0
                * np.power(m_dm1, 4)
                * (3.0 * np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(m_dm1, 2)
                * np.power(3.0 * np.power(m_FS, 2) + np.power(m_IS, 2), 2)
                + 8.0 * np.power(np.power(m_FS, 3) - 1.0 * m_FS * np.power(m_IS, 2), 2)
            )
            + np.power(m_dm1, 2)
            * np.power(FFT3, 2)
            * np.power(
                np.power(m_dm1, 4)
                - 2.0 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(np.power(m_FS, 2) - 1.0 * np.power(m_IS, 2), 2),
                2,
            )
        )
    )
    return (
        ampSq1_1
        + ampSq1_3
        + ampSq2_2
        + ampSq2_4
        + ampSq3_1
        + ampSq3_3
        + ampSq4_2
        + ampSq4_4
    )


def amp_square_3_1_1(
    wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1, m_dm2, qsq, theta1
) -> float:
    raise NotImplementedError()
