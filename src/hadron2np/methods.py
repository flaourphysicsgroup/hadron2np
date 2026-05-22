from particle import Particle
import hadron2np
from numpy import sqrt
import re


def parse_errors(value) -> dict:
    """
    解析带有括号误差的字符串，如 '3.26(3)e-2', '2.00', '3.22(18)e3'。
    返回包含中心值、上限和下限的字典。
    """
    # 1. 优先判断：如果不是字符串（比如已经是 float, int 等），直接返回结果
    if not isinstance(value, str):
        try:
            central = float(value)
            return {'central': central, 'upper': central, 'lower': central}
        except (ValueError, TypeError):
            # 如果连转成 float 都失败，返回全 None
            return {'central': None, 'upper': None, 'lower': None}

    # 2. 走到这里说明 value 肯定是字符串，继续执行原有的解析逻辑
    value = value.strip()
    
    # 正则表达式解析：匹配 [数值](误差)[科学计数法] 的格式
    pattern = r'^([+-]?(?:\d+\.?\d*|\.\d+))(?:\((\d+)\))?(?:[eE]([+-]?\d+))?$'
    
    match = re.match(pattern, value)
    
    if not match:
        # 如果格式完全不匹配，尝试直接转为 float（兼容普通字符串如 '2.00'）
        try:
            central = float(value)
            return {'central': central, 'upper': central, 'lower': central}
        except ValueError:
            return {'central': None, 'upper': None, 'lower': None}

    # 提取匹配到的各组数据
    base_value_str = match.group(1)   # 基础数值部分，如 '3.26'
    error_digits_str = match.group(2) # 括号里的误差数字，如 '3' 或 '18'
    exponent_str = match.group(3)     # 科学计数法的指数，如 '-2' 或 '3'

    # 计算缩放比例（处理 e-2, e3 等科学计数法）
    scale = 10 ** int(exponent_str) if exponent_str else 1.0
    
    # 计算中心值 (base_value * scale)
    central = float(base_value_str) * scale
    
    # 如果没有括号误差，直接返回相等的上下限
    if error_digits_str is None:
        return {'central': central, 'upper': central, 'lower': central}

    # 计算绝对误差值
    # 找出中心值（不带指数时）的小数点后有几位
    decimal_places = len(base_value_str.split('.')[1]) if '.' in base_value_str else 0
    # 误差的绝对值 = 括号内的数字 * 10^(-小数位数) * 缩放比例
    error_magnitude = float(error_digits_str) * (10 ** -decimal_places) * scale

    return {
        'central': central,
        'upper': central + error_magnitude,
        'lower': central - error_magnitude
    }


def get_particle(name) -> Particle:
    return Particle.from_pdgid(hadron2np.config['pdgid'][name])


def lambda_f(asq, bsq, csq):
    """Kallen function.
    两体相空间中的末态粒子动量 $\vec{p}_1 = lambda^{1/2} / 2M"""
    return asq**2 + bsq**2 + csq**2 - 2 * (asq * bsq + asq * csq + bsq * csq)


class phase_space_3_body():
    def __init__(self, m_I, m_F, m_1, m_2, qsq):
        self.E2_star = self.get_E2_star(qsq, m_1, m_2)
        self.E3_star = self.get_E3_star(qsq, m_I, m_F)
        self.m_plus = self.get_m_plus(m_I, m_F, m_1, m_2, qsq)
        self.m_minus = self.get_m_minus(m_I, m_F, m_1, m_2, qsq)

    @staticmethod
    def get_E2_star(qsq, m_1, m_2):
        return (qsq - m_1**2 + m_2**2) / (2 * sqrt(qsq))

    @staticmethod
    def get_E3_star(qsq, m_I, m_F):
        return (m_I**2 - m_F**2 - qsq) / (2 * sqrt(qsq))

    def get_m_plus(self, m_I, m_F, m_1, m_2, qsq):
        E2 = self.E2_star
        E3 = self.E3_star
        return (E2 + E3)**2 - (sqrt(E2**2 - m_2**2) - sqrt(E3**2 - m_F**2))**2

    def get_m_minus(self, m_I, m_F, m_1, m_2, qsq):
        E2 = self.E2_star
        E3 = self.E3_star
        return (E2 + E3)**2 - (sqrt(E2**2 - m_2**2) + sqrt(E3**2 - m_F**2))**2

    def m23sq(self):
        return self.m_plus - self.m_minus

    def m23sq_sq(self):
        return self.m23sq() * (self.m_plus + self.m_minus)

    def m23sq_cube(self):
        return self.m23sq() * (self.m_plus**2 + self.m_minus**2 + self.m_plus * self.m_minus)
