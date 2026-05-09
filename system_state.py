"""系统状态：UD 位置/速度、每 UD 的 f_u、p_u^tra、ζ_u、阴影 χ_{u,m}、各 MEC 队列。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .params import SimulationParams


@dataclass
class SystemState:
    params: SimulationParams
    rng: np.random.Generator

    bs_pos: np.ndarray = field(init=False)
    user_pos: np.ndarray = field(init=False)
    user_vel: np.ndarray = field(init=False)
    user_mec: np.ndarray = field(init=False)
    f_user_hz: np.ndarray = field(init=False)
    # Table I：p_u^tra ∈ [0.1, 0.5] W，每 UD 初始化一次
    p_transmit_w: np.ndarray = field(init=False)
    zeta_u: np.ndarray = field(init=False)
    # AR(1) 阴影 χ_{u,m}（dB）；shadowing_ar1_rho=0 时不用于式 (5)
    chi_um_db: np.ndarray = field(init=False)

    QE: np.ndarray = field(init=False)
    QC: np.ndarray = field(init=False)
    ZE: np.ndarray = field(init=False)
    ZC: np.ndarray = field(init=False)

    sum_AE_hist: np.ndarray = field(init=False)
    sum_AC_hist: np.ndarray = field(init=False)
    # 本地计算积压（bits），用于 LC/RO 下与文献类似的排队时延量级（Little 型）
    LQ: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        p = self.params
        m = p.num_mec
        assert m == 4
        u = p.num_ud
        per = p.users_per_cluster

        half = p.area_m / 2.0
        centers = np.array(
            [
                [half / 2.0, half / 2.0],
                [3.0 * half / 2.0, half / 2.0],
                [half / 2.0, 3.0 * half / 2.0],
                [3.0 * half / 2.0, 3.0 * half / 2.0],
            ]
        )
        self.bs_pos = centers

        self.user_pos = np.zeros((u, 2), dtype=float)
        mean_scalar = float(p.velocity_mean_speed_m_s) / math.sqrt(2.0)
        self.user_vel = self.rng.normal(mean_scalar, float(p.velocity_std_asymptotic), size=(u, 2))
        for mi in range(m):
            idx = slice(mi * per, (mi + 1) * per)
            spread = min(half * 0.45, 200.0)
            self.user_pos[idx, 0] = centers[mi, 0] + self.rng.uniform(-spread, spread, per)
            self.user_pos[idx, 1] = centers[mi, 1] + self.rng.uniform(-spread, spread, per)
        self.user_pos = np.clip(self.user_pos, 0.0, p.area_m)

        self.f_user_hz = self.rng.uniform(p.f_user_hz_min, p.f_user_hz_max, size=u)
        lo, hi = float(p.transmit_power_min_w), float(p.transmit_power_max_w)
        assert lo <= hi, "transmit_power_min_w 必须 ≤ transmit_power_max_w"
        self.p_transmit_w = self.rng.uniform(lo, hi, size=u).astype(float)
        if p.zeta_per_ud is None:
            self.zeta_u = np.full(u, float(p.zeta), dtype=float)
        else:
            self.zeta_u = np.array(p.zeta_per_ud, dtype=float)
        self.chi_um_db = np.zeros((u, m), dtype=float)

        self.user_mec = np.zeros(u, dtype=int)
        self._reassign_mec()

        self.QE = np.zeros(m, dtype=float)
        self.QC = np.zeros(m, dtype=float)
        self.ZE = np.zeros(m, dtype=float)
        self.ZC = np.zeros(m, dtype=float)
        self.sum_AE_hist = np.zeros(m, dtype=float)
        self.sum_AC_hist = np.zeros(m, dtype=float)
        self.LQ = np.zeros(u, dtype=float)

    def _reassign_mec(self) -> None:
        d2 = np.sum((self.user_pos[:, None, :] - self.bs_pos[None, :, :]) ** 2, axis=2)
        self.user_mec = np.argmin(d2, axis=1)

    def users_on_mec(self, m: int) -> np.ndarray:
        return np.nonzero(self.user_mec == m)[0]

    def distances_to_mec(self, m: int, users: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(self.user_pos[users] - self.bs_pos[m], axis=1)
        return np.maximum(d, self.params.d_min_m)
