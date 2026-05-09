"""
仿真参数。

默认与论文 **Table I** 及 **Section III** 记号一致；支持 **每 MEC 异质**
``f_mec_hz_per_mec``、``bandwidth_hz_per_mec``、``connection_capacity_per_mec``、
``rate_mec_to_cloud_bps_per_mec``（为 None 时用标量广播）。

阴影：``shadowing_ar1_rho=0`` 时式 (5) 中 χ 每链路每调用为独立 N(0,σ²)；
``0<rho<1`` 时为 **(u,m) 上 AR(1) 慢相关**，与独立快衰常见扩展一致（正文若写 i.i.d. 则保持 rho=0）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator


def _tuple_len_ok(t: tuple[float, ...] | tuple[int, ...] | None, n: int) -> bool:
    return t is None or len(t) == n


@dataclass
class SimulationParams:
    area_m: float = 1000.0
    num_mec: int = 4
    num_ud: int = 200
    connection_capacity: int = 30
    fast_alg3: bool = False

    slot_duration: float = 1.0
    num_slots: int = 200

    omega: float = 0.9
    velocity_mean_speed_m_s: float = 1.0
    velocity_std_asymptotic: float = 2.0

    bandwidth_hz: float = 20e6
    noise_power_dbm: float = -98.0
    use_noise_psd: bool = False
    noise_psd_dbm_per_hz: float = -174.0

    transmit_power_min_w: float = 0.1
    transmit_power_max_w: float = 0.5

    f_user_hz_min: float = 1.0e9
    f_user_hz_max: float = 2.0e9
    f_mec_hz: float = 5.0e9
    f_cloud_hz: float = 10e9
    rate_mec_to_cloud_bps: float = 8e6

    zeta: float = 1e-28
    # 每 UD ζ_u；None 时全为 ``zeta``（与 Table I 单行一致）
    zeta_per_ud: tuple[float, ...] | None = None

    cpu_cycles_per_bit: float = 1000.0

    task_bits_min: float = 1e4
    task_bits_max: float = 1e6
    task_arrival_lambda: float = 1.0

    carrier_freq_hz: float = 2.0e9
    d_ref_m: float = 1.0
    path_loss_exponent: float = 2.42
    shadowing_std_db: float = 4.0
    # χ_{u,m} 的 AR(1) 系数；0 表示每时隙独立 χ（与式 (5) 文字 i.i.d. 一致）
    shadowing_ar1_rho: float = 0.0
    use_empirical_pathloss_db: bool = False

    pathloss_a_db: float = 128.1
    pathloss_b_db_per_dec: float = 37.6
    d_min_m: float = 1.0
    rayleigh_alpha: float = 4.0

    # 每 MEC 异质；None 时用上面标量
    f_mec_hz_per_mec: tuple[float, ...] | None = None
    bandwidth_hz_per_mec: tuple[float, ...] | None = None
    connection_capacity_per_mec: tuple[int, ...] | None = None
    rate_mec_to_cloud_bps_per_mec: tuple[float, ...] | None = None

    V: float = 35.0
    D_bar_E: float = 1.5
    D_bar_C: float = 1.0

    rng_seed: int = 42

    def __post_init__(self) -> None:
        n = int(self.num_mec)
        assert _tuple_len_ok(self.f_mec_hz_per_mec, n), "f_mec_hz_per_mec 长度须为 num_mec"
        assert _tuple_len_ok(self.bandwidth_hz_per_mec, n), "bandwidth_hz_per_mec 长度须为 num_mec"
        assert _tuple_len_ok(self.connection_capacity_per_mec, n), "connection_capacity_per_mec 长度须为 num_mec"
        assert _tuple_len_ok(self.rate_mec_to_cloud_bps_per_mec, n), "rate_mec_to_cloud_bps_per_mec 长度须为 num_mec"
        u = int(self.num_ud)
        assert self.zeta_per_ud is None or len(self.zeta_per_ud) == u, "zeta_per_ud 长度须为 num_ud"

    def f_mec_hz_at(self, m: int) -> float:
        if self.f_mec_hz_per_mec is not None:
            return float(self.f_mec_hz_per_mec[int(m)])
        return float(self.f_mec_hz)

    def bandwidth_hz_at(self, m: int) -> float:
        if self.bandwidth_hz_per_mec is not None:
            return float(self.bandwidth_hz_per_mec[int(m)])
        return float(self.bandwidth_hz)

    def connection_capacity_at(self, m: int) -> int:
        if self.connection_capacity_per_mec is not None:
            return int(self.connection_capacity_per_mec[int(m)])
        return int(self.connection_capacity)

    def rate_mec_to_cloud_at(self, m: int) -> float:
        if self.rate_mec_to_cloud_bps_per_mec is not None:
            return float(self.rate_mec_to_cloud_bps_per_mec[int(m)])
        return float(self.rate_mec_to_cloud_bps)

    @property
    def users_per_cluster(self) -> int:
        assert self.num_ud % self.num_mec == 0, "num_ud 必须能被 num_mec 整除"
        return self.num_ud // self.num_mec

    def sample_task_bits(self, rng: Generator) -> np.ndarray:
        u = self.num_ud
        if self.task_arrival_lambda >= 1.0 - 1e-15:
            return rng.uniform(self.task_bits_min, self.task_bits_max, size=u)
        m = rng.random(u) < float(self.task_arrival_lambda)
        bits = np.zeros(u, dtype=float)
        n = int(np.sum(m))
        if n > 0:
            bits[m] = rng.uniform(self.task_bits_min, self.task_bits_max, size=n)
        return bits


def params_table1() -> SimulationParams:
    """Table I + Section VI 短时隙：T=10 ms、U=40、¯D^E/¯D^C 与文中 τ 设定一致。"""
    return SimulationParams(
        num_ud=40,
        slot_duration=0.01,
        num_slots=5000,
        D_bar_E=0.05,
        D_bar_C=0.2,
        fast_alg3=False,
    )


def params_paper_figures() -> SimulationParams:
    """Fig.3–6：U=200、δ=1 s、¯D^E/¯D^C；其余继承 Table I 默认。"""
    return SimulationParams(
        num_ud=200,
        num_slots=200,
        slot_duration=1.0,
        D_bar_E=1.5,
        D_bar_C=1.0,
        fast_alg3=False,
    )


def noise_power_w(params: SimulationParams) -> float:
    if params.use_noise_psd:
        bw = float(params.bandwidth_hz)
        n_dbm = params.noise_psd_dbm_per_hz + 10.0 * math.log10(max(bw, 1.0))
        return 10.0 ** ((n_dbm - 30.0) / 10.0)
    return 10.0 ** ((params.noise_power_dbm - 30.0) / 10.0)
