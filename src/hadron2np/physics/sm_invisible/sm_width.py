from .B_Hvv import _dGamma_dz_pi, _dGammaL_dz_rho, _dGammaT_dz_rho, _dGamma_dz_K
from .Lb_Lvv import dGamma_dqsq as dGamma_Lb_L
import hadron2np

def partial_width_3_1_1(ff_imp, m_sm, fcnc_hadron, qsq):
    m_IS, m_FS, m_iq, m_fq = m_sm
    ffs = ff_imp.get_central_values(qsq)

    match fcnc_hadron:
        case 'B0->pi0' | 'B0->rho0':
            raise NotImplementedError(f'TODO: Decay process not implementd yet: {fcnc_hadron}')
        case 'B+->pi+' | 'B+->K+' | 'B0->K0':
            res = _dGamma_dz_pi(qsq / m_IS ** 2, hadron2np.parameters_dict)
            raise NotImplementedError(f'TODO: Decay process not implementd yet: {fcnc_hadron}')
        case 'B+->rho+' | 'B+->K*+' | 'B0->K*0':
            res = _dGammaL_dz_rho(qsq / m_IS ** 2, hadron2np.parameters_dict)
            raise NotImplementedError(f'TODO: Decay process not implementd yet: {fcnc_hadron}')
        case "Lambda_b->Lambda":
            res = dGamma_Lb_L(ffs, m_IS, m_FS, qsq)
        case _:
            raise NotImplementedError(f'Decay process not implementd yet: {fcnc_hadron}')

    return res