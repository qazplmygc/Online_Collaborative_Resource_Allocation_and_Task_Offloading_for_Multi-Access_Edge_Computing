"""队列演化式 (10)(11)、虚拟队列式 (15)、Little 指标、本地积压与移动性。"""
from __future__ import annotations

import numpy as np

from .mobility import gauss_markov_velocity_step, update_positions
from .params import SimulationParams
from .system_state import SystemState


def advance_queues_virtual_and_mobility(
    state: SystemState,
    p: SimulationParams,
    t_slot: int,
    task_bits: np.ndarray,
    x_mec: np.ndarray,
    x_cld: np.ndarray,
) -> tuple[float, float, float]:
    """
    返回 (Q^E 型边缘排队时延, Q^C 型云排队时延, 本地积压等效时延)。
    本地：LQ 按每 UD 最大处理能力 serv = f_u·δ/c_u 更新；等效时延 ≈ mean(LQ / serv_bps)。
    """
    for m in range(p.num_mec):
        users = state.users_on_mec(m)
        if len(users) == 0:
            continue
        s_vec = task_bits[users]
        xm = x_mec[users].astype(float)
        xc = x_cld[users]
        AE = float(np.sum((xm - xc) * s_vec))
        AC = float(np.sum(xc * s_vec))
        leave_e = p.f_mec_hz_at(m) * p.slot_duration / p.cpu_cycles_per_bit
        leave_c = p.rate_mec_to_cloud_at(m) * p.slot_duration
        state.QE[m] = max(state.QE[m] - leave_e, 0.0) + AE
        state.QC[m] = max(state.QC[m] - leave_c, 0.0) + AC
        state.sum_AE_hist[m] += AE
        state.sum_AC_hist[m] += AC

    U = p.num_ud
    xm_all = x_mec.astype(float)
    for u in range(U):
        arr = float((1.0 - xm_all[u]) * float(task_bits[u]))
        cap = float(state.f_user_hz[u] * p.slot_duration / p.cpu_cycles_per_bit)
        w = float(state.LQ[u]) + arr
        serv = min(w, cap)
        state.LQ[u] = w - serv

    serv_bps = state.f_user_hz / max(p.cpu_cycles_per_bit, 1e-12)
    delay_local = float(np.mean(state.LQ / np.maximum(serv_bps, 1.0)))

    t = max(t_slot, 1)
    delay_edge = 0.0
    delay_cloud = 0.0
    # 平均到达率过小时避免 QE/ÃE 数值爆炸（LC 无边缘到达时 Little 分母退化）
    ae_floor = max(p.task_bits_min * p.task_arrival_lambda * 0.1, 1e3)
    ac_floor = max(p.task_bits_min * p.task_arrival_lambda * 0.05, 1e3)

    for m in range(p.num_mec):
        ae_bar = max(state.sum_AE_hist[m] / t, ae_floor)
        ac_bar = max(state.sum_AC_hist[m] / t, ac_floor)
        delay_edge += state.QE[m] / ae_bar
        delay_cloud += state.QC[m] / ac_bar
        state.ZE[m] = max(state.ZE[m] + state.QE[m] / ae_bar - p.D_bar_E, 0.0)
        state.ZC[m] = max(state.ZC[m] + state.QC[m] / ac_bar - p.D_bar_C, 0.0)

    delay_edge /= p.num_mec
    delay_cloud /= p.num_mec

    state.user_vel = gauss_markov_velocity_step(state.user_vel, p, state.rng)
    state.user_pos, state.user_vel = update_positions(state.user_pos, state.user_vel, p)
    state._reassign_mec()
    return delay_edge, delay_cloud, delay_local
