import yaml
from pathlib import Path
import numpy as np

with open(Path(__file__).parent / 'conventions.yaml', 'r') as file:
    par_DLEFT = yaml.safe_load(file)


def get_zero_wcs(basis):
    """初始化一个完整的 wcs 字典"""
    if basis == 'DLEFT(S/P)':
        ops = par_DLEFT['wcs in S/P basis']
        ret = {}
        for op in ops:
            if op[1] == 3:  # 只有三个粒子时, DM 的味指标没有意义, 所以只取了两维
                ret.update({op[0]: np.zeros((3, 3), dtype=complex)})
            elif op[1] == 4:
                ret.update({op[0]: np.zeros((3, 3, 2, 2), dtype=complex)})
            else:
                raise ValueError(f'operator initialization error: {op[0]}')
        return ret
    elif basis == 'DLEFT(L/R)':
        ops = par_DLEFT['wcs in L/R basis']
        ret = {}
        for op in ops:
            if op[1] == 3:
                ret.update({op[0]: np.zeros((3, 3), dtype=complex)})
            elif op[1] == 4:
                ret.update({op[0]: np.zeros((3, 3, 2, 2), dtype=complex)})
            else:
                raise ValueError(f'operator initialization error: {op[0]}')
        return ret
    else:
        raise ValueError(f'算符基矢命名错误: {basis}')


def convert_to_SP_basis(wcs_LR):
    wcs_SP = get_zero_wcs('DLEFT(S/P)')
    # 三粒子顶点的Wilson系数取的是二维矩阵, 可以直接转置复共轭
    wcs_SP['L_S_dphi'] = 1 / 2 * (np.swapaxes(wcs_LR['L_SR_dphi'], 0, 1).conj() + wcs_LR['L_SR_dphi'])
    wcs_SP['L_P_dphi'] = 1j / 2 * (np.swapaxes(wcs_LR['L_SR_dphi'], 0, 1).conj() - wcs_LR['L_SR_dphi'])
    wcs_SP['L_V_dphi'] = 1 / 2 * (wcs_LR['L_VR_dphi'] + wcs_LR['L_VL_dphi'])
    wcs_SP['L_A_dphi'] = 1 / 2 * (wcs_LR['L_VR_dphi'] - wcs_LR['L_VL_dphi'])

    # 四粒子顶点的Wilson系数是四维矩阵
    wcs_SP['L_S_dphi2'] = 1 / 2 * (np.swapaxes(wcs_LR['L_SR_dphi2'], 0, 1).conj() + wcs_LR['L_SR_dphi2'])
    wcs_SP['L_P_dphi2'] = 1j / 2 * (np.swapaxes(wcs_LR['L_SR_dphi2'], 0, 1).conj() - wcs_LR['L_SR_dphi2'])
    wcs_SP['L_V_dphi2'] = 1 / 2 * (wcs_LR['L_VR_dphi2'] + wcs_LR['L_VL_dphi2'])
    wcs_SP['L_A_dphi2'] = 1 / 2 * (wcs_LR['L_VR_dphi2'] - wcs_LR['L_VL_dphi2'])

    wcs_SP['L_V_dX'] = 1 / 2 * (wcs_LR['L_VR_dX'] + wcs_LR['L_VL_dX'])
    wcs_SP['L_A_dX'] = 1 / 2 * (wcs_LR['L_VR_dX'] - wcs_LR['L_VL_dX'])
    wcs_SP['L_T_dX'] = 1 / 2 * (np.swapaxes(wcs_LR['L_TR_dX'], 0, 1).conj() + wcs_LR['L_TR_dX'])
    wcs_SP['L_T5_dX'] = 1j / 2 * (np.swapaxes(wcs_LR['L_TR_dX'], 0, 1).conj() - wcs_LR['L_TR_dX'])

    wcs_SP['L_S_dX2'] = 1 / 2 * (np.swapaxes(wcs_LR['L_SR_dX2'], 0, 1).conj() + wcs_LR['L_SR_dX2'])
    wcs_SP['L_P_dX2'] = 1j / 2 * (np.swapaxes(wcs_LR['L_SR_dX2'], 0, 1).conj() - wcs_LR['L_SR_dX2'])
    wcs_SP['L_V_dX2'] = 1 / 2 * (wcs_LR['L_VR_dX2'] + wcs_LR['L_VL_dX2'])
    wcs_SP['L_A_dX2'] = 1 / 2 * (wcs_LR['L_VR_dX2'] - wcs_LR['L_VL_dX2'])
    wcs_SP['L_Vtilde_dX2'] = 1 / 2 * (wcs_LR['L_VRtilde_dX2'] + wcs_LR['L_VLtilde_dX2'])
    wcs_SP['L_Atilde_dX2'] = 1 / 2 * (wcs_LR['L_VRtilde_dX2'] - wcs_LR['L_VLtilde_dX2'])
    wcs_SP['L_T_dX2'] = 1 / 2 * (np.swapaxes(wcs_LR['L_TR_dX2'], 0, 1).conj() + wcs_LR['L_TR_dX2'])
    wcs_SP['L_T5_dX2'] = 1j / 2 * (np.swapaxes(wcs_LR['L_TR_dX2'], 0, 1).conj() - wcs_LR['L_TR_dX2'])
    wcs_SP['L_Vpartial_dX2'] = 1 / 2 * (wcs_LR['L_VRpartial_dX2'] + wcs_LR['L_VLpartial_dX2'])
    wcs_SP['L_Apartial_dX2'] = 1 / 2 * (wcs_LR['L_VRpartial_dX2'] - wcs_LR['L_VLpartial_dX2'])

    wcs_SP['L_S_dchi2'] = 1 / 2 * (wcs_LR['L_SLR_dchi2'] + wcs_LR['L_SRR_dchi2'])
    wcs_SP['L_P_dchi2'] = 1j / 2 * (wcs_LR['L_SLR_dchi2'] - wcs_LR['L_SRR_dchi2'])
    wcs_SP['L_V_dchi2'] = 1 / 2 * (wcs_LR['L_VRR_dchi2'] + wcs_LR['L_VLR_dchi2'])
    wcs_SP['L_A_dchi2'] = 1 / 2 * (wcs_LR['L_VRR_dchi2'] - wcs_LR['L_VLR_dchi2'])
    wcs_SP['L_T_dchi2'] = wcs_LR['L_TRR_dchi2']
    return wcs_SP


def get_wc_dimension(wc_name) -> int:
    """得到 wc 的量纲. 如 6 维算符返回 -2"""
    names = {name[0]: name[1] for name in par_DLEFT['wc names in S/P basis']}
    names.update({name[0]: name[1] for name in par_DLEFT['wc names in L/R basis']})
    return 4 - names[wc_name]
