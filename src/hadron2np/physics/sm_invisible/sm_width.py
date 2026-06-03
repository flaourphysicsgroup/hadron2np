from .B_Hvv import (
    partial_width_B2pi_plus,
    partial_width_B2rho_plus,
    partial_width_B2K_plus,
    partial_width_B2Kstar_plus,
)
from .Lb_Lvv import dGamma_dqsq as dGamma_Lb_L
import hadron2np


def partial_width_3_1_1(ff_imp, m_sm, fcnc_hadron, qsq, polarization):
    m_IS, m_FS, m_iq, m_fq = m_sm
    ffs = ff_imp.get_central_values(qsq)
    par = hadron2np.parameters_dict

    match fcnc_hadron:
        case 'B+->pi+':
            res = partial_width_B2pi_plus(par, qsq)
        case 'B+->rho+':
            res = partial_width_B2rho_plus(par, qsq, polarization)
        case 'B0->pi0' | 'B0->K0' | 'B+->K+':
            res = partial_width_B2K_plus(par, ffs, m_IS, m_FS, qsq)
        case 'B0->rho0' | 'B0->K*0' | 'B+->K*+':
            res = partial_width_B2Kstar_plus(par, ffs, m_IS, m_FS, qsq, polarization)
        case "Lambda_b->Lambda":
            res = dGamma_Lb_L(ffs, m_IS, m_FS, qsq)
        case _:
            raise NotImplementedError(f'Decay process not implementd yet: {fcnc_hadron}')

    return res
