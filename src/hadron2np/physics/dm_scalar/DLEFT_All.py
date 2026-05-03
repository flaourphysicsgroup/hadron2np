"""分发 wilson 系数的味指标和 ampSq 函数"""

from . import DLEFT_B_0, DLEFT_B_P, DLEFT_B_V
from . import DLEFT_K_0, DLEFT_K_P
from . import DLEFT_Quarkonium_0
from . import DLEFT_Lambdab_Lambda
from hadron2np import Phase_space_factors as ps
import numpy as np
from hmff.classes import Impl


def Gamma_IS_XX(fcnc_hadron, index, wcs, m_dm, m_sm, decay_constant):
    L_S = wcs['L_S_dphi2'][*index]
    L_P = wcs['L_P_dphi2'][*index]
    L_V = wcs['L_V_dphi2'][*index]
    L_A = wcs['L_A_dphi2'][*index]
    m_dm_1, m_dm_2 = m_dm
    m_IS, _, m_iq, m_fq = m_sm

    match fcnc_hadron:
        case 'B->0':
            gamma_func = DLEFT_B_0.Gamma
        case 'K->0':
            gamma_func = DLEFT_K_0.Gamma
        case 'Upsilon->0' | 'Jpsi->0':
            gamma_func = DLEFT_Quarkonium_0.Gamma
        case _:
            raise NotImplementedError(f'Decay process not implementd yet: {fcnc_hadron}')

    return gamma_func(L_S, L_P, L_V, L_A, m_dm_1, m_dm_2, m_IS, m_iq, m_fq, decay_constant)


def Gamma_IS_FSX(fcnc_hadron, index, wcs, m_dm, m_sm, ff_imp):
    L_S = wcs['L_S_dphi'][*index]
    L_P = wcs['L_P_dphi'][*index]
    L_V = wcs['L_V_dphi'][*index]
    L_A = wcs['L_A_dphi'][*index]
    m_dm_1 = m_dm[0]
    m_IS, m_FS, m_iq, m_fq = m_sm
    ffs = ff_imp.get_values(wc_obj=None, par_dict=None, q2=m_dm_1**2)

    match fcnc_hadron:
        case 'B->P':
            gamma_func = DLEFT_B_P.Gamma_PX
        case 'B->V':
            gamma_func = DLEFT_B_V.Gamma_VX
        case 'K->P':
            gamma_func = DLEFT_K_P.Gamma_PX
        case 'Lambdab->Lambda':
            gamma_func = DLEFT_Lambdab_Lambda.Gamma_LambdaX
        case _:
            raise NotImplementedError(f'Decay process not implementd yet: {fcnc_hadron}')

    return gamma_func(L_S, L_P, L_V, L_A, m_dm_1, m_IS, m_FS, m_iq, m_fq, ffs)


def dGamma_IS_FSXX(fcnc_hadron, index, wcs, m_dm, m_sm, ff_imp, qsq):
    L_S = wcs['L_S_dphi2'][*index]
    L_P = wcs['L_P_dphi2'][*index]
    L_V = wcs['L_V_dphi2'][*index]
    L_A = wcs['L_A_dphi2'][*index]
    m_dm_1, m_dm_2 = m_dm
    m_IS, m_FS, m_iq, m_fq = m_sm
    ffs = ff_imp.get_values(wc_obj=None, par_dict=None, q2=qsq)

    match fcnc_hadron:
        case 'B->P':
            gamma_func = DLEFT_B_P.dGamma_PXX
        case 'B->V':
            gamma_func = DLEFT_B_V.dGamma_VXX
        case 'K->P':
            gamma_func = DLEFT_K_P.dGamma_PXX
        case 'Lambdab->Lambda':
            gamma_func = DLEFT_Lambdab_Lambda.dGamma_LambdaXX
        case _:
            raise NotImplementedError(f'Decay process not implementd yet: {fcnc_hadron}')

    return gamma_func(L_S, L_P, L_V, L_A, m_dm_1, m_dm_2, m_IS, m_FS, m_iq, m_fq, ffs, qsq)


def width_2_0_1(wcs: dict, decay_constants: dict, m_sm: list, m_dm: list, fcnc_hadron: str, flavor_index: list):
    L_S = wcs['L_S_dphi2'][*flavor_index]
    L_P = wcs['L_P_dphi2'][*flavor_index]
    L_V = wcs['L_V_dphi2'][*flavor_index]
    L_A = wcs['L_A_dphi2'][*flavor_index]
    wcs_dict = {'S': L_S, 'P': L_P, 'V': L_V, 'A': L_A}
    m_dm_1, m_dm_2 = m_dm
    m_IS, _, m_iq, m_fq = m_sm

    match fcnc_hadron:
        case 'B->0':
            f_B = decay_constants['Bs']
            ampSq = DLEFT_B_0.amp_square_2_0_1
        case 'K->0':
            f_B = decay_constants['KL']
            ampSq = DLEFT_K_0.amp_square_2_0_1
        case 'Upsilon->0':
            f_B = decay_constants['Upsilon(1S)']
            ampSq = DLEFT_Quarkonium_0.amp_square_2_0_1
        case 'Jpsi->0':
            f_B = decay_constants['J/psi']
            ampSq = DLEFT_Quarkonium_0.amp_square_2_0_1
        case _:
            raise NotImplementedError(f'Decay process not implementd yet: {fcnc_hadron}')
    f_phase_space = ps.two_body_phase_space_factor(m_IS, m_dm_1, m_dm_2)
    return f_phase_space * ampSq(wcs_dict, f_B, m_iq, m_fq, m_IS, m_dm_1, m_dm_2)


def width_2_1_1(
    wcs: dict, ff_imp: Impl, m_sm: list, m_dm: list, fcnc_hadron: str, flavor_index: list
):
    L_S = wcs['L_S_dphi'][*flavor_index]
    L_P = wcs['L_P_dphi'][*flavor_index]
    L_V = wcs['L_V_dphi'][*flavor_index]
    L_A = wcs['L_A_dphi'][*flavor_index]
    wcs_dict = {'S': L_S, 'P': L_P, 'V': L_V, 'A': L_A}
    m_dm_1 = m_dm[0]
    m_IS, m_FS, m_iq, m_fq = m_sm
    ffs = ff_imp.get_central_values(qsq=m_dm_1**2)

    match fcnc_hadron:
        case 'B->P':
            ampSq = DLEFT_B_P.amp_square_2_1_1
        case 'B->V':
            ampSq = DLEFT_B_V.amp_square_2_1_1
        case 'K->P':
            ampSq = DLEFT_K_P.amp_square_2_1_1
        case 'Lambdab->Lambda':
            ampSq = DLEFT_Lambdab_Lambda.amp_square_2_1_1
        case _:
            raise NotImplementedError(f'Decay process not implementd yet: {fcnc_hadron}')

    f_phase_space = ps.two_body_phase_space_factor(m_IS, m_FS, m_dm_1)
    return f_phase_space * ampSq(wcs_dict, ffs, m_iq, m_fq, m_IS, m_FS, m_dm_1)

def partial_width_3_1_1(
    wcs: dict, ff_imp: Impl, m_sm: list, m_dm: list, 
    fcnc_hadron: str, flavor_index: list, qsq, theta1
):
    L_S = wcs['L_S_dphi2'][*flavor_index]
    L_P = wcs['L_P_dphi2'][*flavor_index]
    L_V = wcs['L_V_dphi2'][*flavor_index]
    L_A = wcs['L_A_dphi2'][*flavor_index]
    wcs_dict = {'S': L_S, 'P': L_P, 'V': L_V, 'A': L_A}
    m_dm_1, m_dm_2 = m_dm
    m_IS, m_FS, m_iq, m_fq = m_sm
    ffs = ff_imp.get_central_values(qsq)

    match fcnc_hadron:
        case 'B->P':
            ampSq = DLEFT_B_P.amp_square_3_1_1
        case 'B->V':
            ampSq = DLEFT_B_V.amp_square_3_1_1
        case 'K->P':
            ampSq = DLEFT_K_P.amp_square_3_1_1
        case 'Lambdab->Lambda':
            ampSq = DLEFT_Lambdab_Lambda.amp_square_3_1_1
        case _:
            raise NotImplementedError(f'Decay process not implementd yet: {fcnc_hadron}')
    f_phase_space = ps.three_body_phase_space_factor(m_IS, m_dm_1, m_dm_2, m_FS, np.sqrt(qsq), theta1)
    return f_phase_space * ampSq(wcs_dict, ffs, m_iq, m_fq, m_IS, m_FS, m_dm_1, m_dm_2, qsq, theta1)