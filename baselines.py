"""对比基线：LC、RO、GJTORA（论文 Section VI-A-3）。"""
from __future__ import annotations

import numpy as np

from .channel import advance_slot_shadowing_chi_um, rayleigh_channel_gain
from .dynamics import advance_queues_virtual_and_mobility
from .ojcta import algorithm2_cloud_decision, Q_function, total_ud_energy_um
from .params import SimulationParams
from .system_state import SystemState


def _channel_gains_ud_to_serving_mec(state: SystemState, p: SimulationParams) -> np.ndarray:
    advance_slot_shadowing_chi_um(state, p)
    u_tot = int(p.num_ud)
    g_all = np.zeros(u_tot, dtype=float)
    for u in range(u_tot):
        m = int(state.user_mec[u])
        d = float(np.linalg.norm(state.user_pos[u] - state.bs_pos[m]))
        chi_link: float | None = None
        if p.shadowing_ar1_rho > 1e-12:
            chi_link = float(state.chi_um_db[u, m])
        g_all[u] = float(
            rayleigh_channel_gain(p, np.array([max(d, p.d_min_m)]), state.rng, chi_db_link=chi_link)[0]
        )
    return g_all


def run_timeslot_lc(state: SystemState, t_slot: int, task_bits: np.ndarray) -> tuple[float, float, float]:
    """Local computing：全部本地。"""
    p = state.params
    U = task_bits.shape[0]
    x_mec = np.zeros(U, dtype=int)
    x_cld = np.zeros(U, dtype=float)

    g_all = _channel_gains_ud_to_serving_mec(state, p)

    total_e = 0.0
    for m in range(p.num_mec):
        users = state.users_on_mec(m)
        if len(users) == 0:
            continue
        s_vec = task_bits[users].astype(float)
        g_vec = g_all[users]
        f_hz = state.f_user_hz[users]
        xm = np.zeros(len(users), dtype=float)
        total_e += total_ud_energy_um(
            p, s_vec, g_vec, xm, f_hz, state.p_transmit_w[users], state.zeta_u[users], m
        )

    de, dc, dl = advance_queues_virtual_and_mobility(state, p, t_slot, task_bits, x_mec, x_cld)
    return total_e, de, dc, dl


def run_timeslot_fo(state: SystemState, t_slot: int, task_bits: np.ndarray) -> tuple[float, float, float, float]:
    """Full offloading（全卸载）：每个 MEC 簇内在连接容量 C_m 下尽可能多用户卸载（xm=1），
    按簇内用户下标顺序取前 k=min(C_m,|U_m|) 个，确定性。云侧仍用 Algorithm 2（与 RO 一致）。"""
    p = state.params
    U = task_bits.shape[0]
    x_mec = np.zeros(U, dtype=int)
    for m in range(p.num_mec):
        users = state.users_on_mec(m)
        if len(users) == 0:
            continue
        xm = np.zeros(len(users), dtype=int)
        k = min(p.connection_capacity_at(m), len(users))
        xm[:k] = 1
        x_mec[users] = xm

    g_all = _channel_gains_ud_to_serving_mec(state, p)

    x_cld = np.zeros(U, dtype=float)
    for m in range(p.num_mec):
        users = state.users_on_mec(m)
        if len(users) == 0:
            continue
        s_vec = task_bits[users].astype(float)
        g_vec = g_all[users]
        f_hz = state.f_user_hz[users]
        xm = x_mec[users].astype(float)
        xmc = algorithm2_cloud_decision(
            p,
            state.QE[m],
            state.QC[m],
            state.ZE[m],
            state.ZC[m],
            t_slot,
            users,
            s_vec,
            xm,
            state.sum_AE_hist[m],
            state.sum_AC_hist[m],
            state.rng,
        )
        x_cld[users] = xmc

    total_e = 0.0
    for m in range(p.num_mec):
        users = state.users_on_mec(m)
        if len(users) == 0:
            continue
        s_vec = task_bits[users].astype(float)
        g_vec = g_all[users]
        f_hz = state.f_user_hz[users]
        total_e += total_ud_energy_um(
            p, s_vec, g_vec, x_mec[users].astype(float), f_hz, state.p_transmit_w[users], state.zeta_u[users], m
        )

    de, dc, dl = advance_queues_virtual_and_mobility(state, p, t_slot, task_bits, x_mec, x_cld)
    return total_e, de, dc, dl


def run_timeslot_ro(state: SystemState, t_slot: int, task_bits: np.ndarray) -> tuple[float, float, float]:
    """Random offloading：满足 MEC 连接容量约束的均匀随机 xm。"""
    p = state.params
    U = task_bits.shape[0]
    x_mec = np.zeros(U, dtype=int)
    for m in range(p.num_mec):
        users = state.users_on_mec(m)
        if len(users) == 0:
            continue
        xm = np.zeros(len(users), dtype=int)
        k = min(p.connection_capacity_at(m), len(users))
        offload = state.rng.choice(len(users), size=k, replace=False)
        xm[offload] = 1
        x_mec[users] = xm

    g_all = _channel_gains_ud_to_serving_mec(state, p)

    x_cld = np.zeros(U, dtype=float)
    for m in range(p.num_mec):
        users = state.users_on_mec(m)
        if len(users) == 0:
            continue
        s_vec = task_bits[users].astype(float)
        g_vec = g_all[users]
        f_hz = state.f_user_hz[users]
        xm = x_mec[users].astype(float)
        xmc = algorithm2_cloud_decision(
            p,
            state.QE[m],
            state.QC[m],
            state.ZE[m],
            state.ZC[m],
            t_slot,
            users,
            s_vec,
            xm,
            state.sum_AE_hist[m],
            state.sum_AC_hist[m],
            state.rng,
        )
        x_cld[users] = xmc

    total_e = 0.0
    for m in range(p.num_mec):
        users = state.users_on_mec(m)
        if len(users) == 0:
            continue
        s_vec = task_bits[users].astype(float)
        g_vec = g_all[users]
        f_hz = state.f_user_hz[users]
        total_e += total_ud_energy_um(
            p, s_vec, g_vec, x_mec[users].astype(float), f_hz, state.p_transmit_w[users], state.zeta_u[users], m
        )

    de, dc, dl = advance_queues_virtual_and_mobility(state, p, t_slot, task_bits, x_mec, x_cld)
    return total_e, de, dc, dl


def _gjtora_fitness(
    state: SystemState,
    m: int,
    users: np.ndarray,
    s_vec: np.ndarray,
    g_vec: np.ndarray,
    f_hz_vec: np.ndarray,
    t_slot: int,
    xm: np.ndarray,
) -> float:
    p = state.params
    xmc = algorithm2_cloud_decision(
        p,
        state.QE[m],
        state.QC[m],
        state.ZE[m],
        state.ZC[m],
        t_slot,
        users,
        s_vec,
        xm.astype(float),
        state.sum_AE_hist[m],
        state.sum_AC_hist[m],
        state.rng,
    )
    E = total_ud_energy_um(
        p, s_vec, g_vec, xm.astype(float), f_hz_vec, state.p_transmit_w[users], state.zeta_u[users], m
    )
    Q = Q_function(
        p,
        state.QE[m],
        state.QC[m],
        state.ZE[m],
        state.ZC[m],
        t_slot,
        s_vec,
        xm.astype(float),
        xmc,
        state.sum_AE_hist[m],
        state.sum_AC_hist[m],
        m,
    )
    return p.V * E + Q


def run_timeslot_gjtora(state: SystemState, t_slot: int, task_bits: np.ndarray) -> tuple[float, float, float]:
    """GJTORA：遗传算法搜索 xm，带宽为式 (30)，云端用 Algorithm 2。"""
    p = state.params
    U = task_bits.shape[0]
    x_mec = np.zeros(U, dtype=int)
    x_cld = np.zeros(U, dtype=float)

    g_all = _channel_gains_ud_to_serving_mec(state, p)

    for m in range(p.num_mec):
        users = state.users_on_mec(m)
        n = len(users)
        if n == 0:
            continue
        s_vec = task_bits[users].astype(float)
        g_vec = g_all[users]
        f_hz = state.f_user_hz[users]
        pop_size = min(24, max(4, 2 * n))
        generations = 8
        pop: list[np.ndarray] = []
        fit: list[float] = []
        for _ in range(pop_size):
            xm = np.zeros(n, dtype=int)
            k = state.rng.integers(0, min(p.connection_capacity_at(m), n) + 1)
            if k > 0:
                xm[state.rng.choice(n, size=k, replace=False)] = 1
            pop.append(xm)
            fit.append(_gjtora_fitness(state, m, users, s_vec, g_vec, f_hz, t_slot, xm))
        for _ in range(generations):
            parent = pop[int(np.argmin(fit))].copy()
            child = parent.copy()
            for j in range(n):
                if state.rng.random() < 0.15:
                    child[j] = 1 - child[j]
            if np.sum(child) > p.connection_capacity_at(m):
                on = list(np.where(child == 1)[0])
                while len(on) > p.connection_capacity_at(m):
                    child[state.rng.choice(on)] = 0
                    on = list(np.where(child == 1)[0])
            f_child = _gjtora_fitness(state, m, users, s_vec, g_vec, f_hz, t_slot, child)
            worst = int(np.argmax(fit))
            if f_child < fit[worst]:
                pop[worst] = child
                fit[worst] = f_child
        best = pop[int(np.argmin(fit))]
        x_mec[users] = best
        x_cld[users] = algorithm2_cloud_decision(
            p,
            state.QE[m],
            state.QC[m],
            state.ZE[m],
            state.ZC[m],
            t_slot,
            users,
            s_vec,
            best.astype(float),
            state.sum_AE_hist[m],
            state.sum_AC_hist[m],
            state.rng,
        )

    total_e = 0.0
    for m in range(p.num_mec):
        users = state.users_on_mec(m)
        if len(users) == 0:
            continue
        s_vec = task_bits[users].astype(float)
        g_vec = g_all[users]
        f_hz = state.f_user_hz[users]
        total_e += total_ud_energy_um(
            p, s_vec, g_vec, x_mec[users].astype(float), f_hz, state.p_transmit_w[users], state.zeta_u[users], m
        )

    de, dc, dl = advance_queues_virtual_and_mobility(state, p, t_slot, task_bits, x_mec, x_cld)
    return total_e, de, dc, dl


def run_timeslot_ssc(state: SystemState, t_slot: int, task_bits: np.ndarray) -> tuple[float, float, float, float]:
    """SSC：收紧 Lyapunov 时延上界，近似“每时隙硬约束”下更保守的卸载（能耗介于 OJCTA 与 LC 之间）。"""
    from dataclasses import replace

    from .ojcta import run_timeslot_ojcta

    old = state.params
    state.params = replace(old, D_bar_E=0.35, D_bar_C=0.35)
    try:
        return run_timeslot_ojcta(state, t_slot, task_bits, ncc=False, ecf=False)
    finally:
        state.params = old


def run_timeslot_ncc(state: SystemState, t_slot: int, task_bits: np.ndarray) -> tuple[float, float, float, float]:
    """NCC：无云协同（x_{m→c}=0）。"""
    from .ojcta import run_timeslot_ojcta

    return run_timeslot_ojcta(state, t_slot, task_bits, ncc=True, ecf=False)
