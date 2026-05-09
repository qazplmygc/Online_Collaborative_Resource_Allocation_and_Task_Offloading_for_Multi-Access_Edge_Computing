"""Gauss-Markov 移动模型：论文式 (1)(2)；Table I：ω=0.8，v̄=1 m/s。"""
from __future__ import annotations

import math

import numpy as np

from .params import SimulationParams


def gauss_markov_velocity_step(
    v: np.ndarray,
    params: SimulationParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """二维：各分量独立按式 (1)；渐近均值速度取 v̄/√2 m/s 每轴。"""
    w = rng.normal(0.0, params.velocity_std_asymptotic, size=v.shape)
    mean_scalar = params.velocity_mean_speed_m_s / math.sqrt(2.0)
    mean = np.full_like(v, mean_scalar, dtype=float)
    om = params.omega
    return om * v + (1.0 - om) * mean + float(np.sqrt(max(0.0, 1.0 - om * om))) * w


def update_positions(
    pos: np.ndarray,
    v: np.ndarray,
    params: SimulationParams,
) -> tuple[np.ndarray, np.ndarray]:
    """式 (2) + 区域边界反弹。"""
    pos_new = pos + v * params.slot_duration
    v_new = v.copy()
    for axis in (0, 1):
        lo = 0.0
        hi = params.area_m
        mask_lo = pos_new[:, axis] < lo
        mask_hi = pos_new[:, axis] > hi
        pos_new[mask_lo, axis] = lo
        pos_new[mask_hi, axis] = hi
        v_new[mask_lo, axis] = np.abs(v_new[mask_lo, axis])
        v_new[mask_hi, axis] = -np.abs(v_new[mask_hi, axis])
    return pos_new, v_new
