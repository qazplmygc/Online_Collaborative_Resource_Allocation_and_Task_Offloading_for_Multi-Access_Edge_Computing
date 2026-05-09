"""无线信道：与论文 Section III-B 式 (3)–(6) 一致。"""
from __future__ import annotations

import math

import numpy as np

from .params import SimulationParams, noise_power_w
from .system_state import SystemState

SPEED_OF_LIGHT_M_S: float = 299792458.0


def path_loss_mean_db(params: SimulationParams, dist_m: np.ndarray) -> np.ndarray:
    d_km = np.maximum(dist_m / 1000.0, params.d_min_m / 1000.0)
    return params.pathloss_a_db + params.pathloss_b_db_per_dec * np.log10(d_km)


def path_loss_linear_db(params: SimulationParams, dist_m: np.ndarray) -> np.ndarray:
    pl_db = path_loss_mean_db(params, dist_m)
    return np.power(10.0, pl_db / 10.0)


def _large_scale_paper_eq5(
    params: SimulationParams,
    dist_m: np.ndarray,
    rng: np.random.Generator,
    chi_db_override: np.ndarray | float | None = None,
) -> np.ndarray:
    """论文式 (5)；chi_db_override 为 (d) 同形状或标量时不再内部采样 χ。"""
    c = SPEED_OF_LIGHT_M_S
    d0 = max(float(params.d_ref_m), 1e-9)
    fc = float(params.carrier_freq_hz)
    beta = float(params.path_loss_exponent)
    d = np.maximum(np.asarray(dist_m, dtype=float), float(params.d_min_m))

    fspl0 = (4.0 * math.pi * d0 * fc / c) ** 2
    geo = np.power(d / d0, beta)
    if chi_db_override is not None:
        chi = np.asarray(chi_db_override, dtype=float)
        if chi.shape == ():
            sh = np.power(10.0, float(chi) / 10.0)
            sh = np.broadcast_to(sh, d.shape).astype(float)
        else:
            sh = np.power(10.0, chi / 10.0)
    elif params.shadowing_std_db <= 1e-15:
        sh = np.ones_like(d, dtype=float)
    else:
        chi_db = rng.normal(0.0, float(params.shadowing_std_db), size=np.shape(d))
        sh = np.power(10.0, chi_db / 10.0)
    L = fspl0 * geo * sh
    return np.maximum(L, 1e-30)


def _large_scale_3gpp_db(
    params: SimulationParams,
    dist_m: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    pl_db = path_loss_mean_db(params, dist_m)
    if params.shadowing_std_db <= 1e-15:
        chi_db = np.zeros_like(pl_db, dtype=float)
    else:
        chi_db = rng.normal(0.0, float(params.shadowing_std_db), size=np.shape(dist_m))
    return np.power(10.0, (pl_db + chi_db) / 10.0)


def large_scale_loss_linear(
    params: SimulationParams,
    dist_m: np.ndarray,
    rng: np.random.Generator,
    chi_db_override: np.ndarray | float | None = None,
) -> np.ndarray:
    if params.use_empirical_pathloss_db:
        return _large_scale_3gpp_db(params, dist_m, rng)
    return _large_scale_paper_eq5(params, dist_m, rng, chi_db_override=chi_db_override)


def rayleigh_channel_gain(
    params: SimulationParams,
    dist_m: np.ndarray,
    rng: np.random.Generator,
    chi_db_link: float | None = None,
) -> np.ndarray:
    """式 (4)(6)；chi_db_link 为 AR(1) 阴影时 (u,m) 的 χ（dB），否则独立采样。"""
    L = large_scale_loss_linear(params, dist_m, rng, chi_db_override=chi_db_link)
    h = rng.rayleigh(scale=float(params.rayleigh_alpha), size=dist_m.shape)
    h2 = h * h
    g = h2 / np.maximum(L, 1e-12)
    return np.maximum(g, 1e-18)


def advance_slot_shadowing_chi_um(state: SystemState, p: SimulationParams) -> None:
    """每时隙更新 χ_{u,m}（dB）；``shadowing_ar1_rho>0`` 时 AR(1)，否则不修改（由 rayleigh 内独立采样）。"""
    rho = float(p.shadowing_ar1_rho)
    if rho <= 1e-12:
        return
    sigma = float(p.shadowing_std_db)
    u, m = int(p.num_ud), int(p.num_mec)
    eps = state.rng.normal(0.0, 1.0, size=(u, m)) * sigma * math.sqrt(max(1e-18, 1.0 - rho * rho))
    state.chi_um_db = rho * state.chi_um_db + eps


def shannon_rate_bps(
    a_frac: float,
    g: float,
    params: SimulationParams,
    p_tx_w: float,
    bandwidth_hz: float,
) -> float:
    """式 (3)：r = a B_m log2(1 + p_u^tra g / N0)。"""
    n0 = noise_power_w(params)
    snr = float(p_tx_w) * g / max(n0, 1e-30)
    return max(a_frac * float(bandwidth_hz) * math.log2(1.0 + snr), 1e3)


def beta_u(
    params: SimulationParams,
    s_bits: float,
    g: float,
    p_tx_w: float,
    bandwidth_hz: float,
) -> float:
    if s_bits < 1e-9:
        return 1e-15
    n0 = noise_power_w(params)
    pw = float(p_tx_w)
    denom = math.log2(1.0 + pw * g / max(n0, 1e-30))
    return pw * s_bits / max(denom, 1e-12)


def optimal_bandwidth_fractions(
    offload_mask: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    u = offload_mask.astype(bool)
    a = np.zeros_like(betas, dtype=float)
    if not np.any(u):
        return a
    s = np.sum(np.sqrt(np.maximum(betas[u], 0.0)))
    if s <= 0:
        a[u] = 1.0 / np.sum(u)
        return a
    a[u] = np.sqrt(np.maximum(betas[u], 0.0)) / s
    return a
