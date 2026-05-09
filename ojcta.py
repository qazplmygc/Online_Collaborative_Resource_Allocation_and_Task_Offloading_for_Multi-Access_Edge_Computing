"""
OJCTA：论文 Section V、Algorithm 1–4。
虚拟队列为 ZE_m、ZC_m（式 (15)）；物理队列为 QE_m、QC_m（式 (10)(11)）。
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .channel import (
    advance_slot_shadowing_chi_um,
    beta_u,
    optimal_bandwidth_fractions,
    rayleigh_channel_gain,
    shannon_rate_bps,
)
from .params import SimulationParams
from .system_state import SystemState

IDX_LOCAL = 0
IDX_MEC = 1


def _local_energy(params: SimulationParams, s_bits: float, f_user_hz: float, zeta_u: float) -> float:
    return float(zeta_u) * (float(f_user_hz) ** 2) * s_bits * params.cpu_cycles_per_bit


def _edge_energy(
    params: SimulationParams,
    s_bits: float,
    g: float,
    a_frac: float,
    p_tx_w: float,
    bandwidth_hz: float,
) -> float:
    r = shannon_rate_bps(a_frac, g, params, p_tx_w, bandwidth_hz)
    return float(p_tx_w) * s_bits / max(r, 1e-12)


def total_ud_energy_um(
    params: SimulationParams,
    s_vec: np.ndarray,
    g_vec: np.ndarray,
    xm: np.ndarray,
    f_hz_vec: np.ndarray,
    p_tx_vec: np.ndarray,
    zeta_vec: np.ndarray,
    mec_index: int,
) -> float:
    """Um 上所有 UD 的 Eloc+Eoff（式 (7)(8)）；B_m、f_m、ζ_u、p_u^tra 与论文下标一致。"""
    xm = xm.astype(float)
    f_hz_vec = f_hz_vec.astype(float)
    p_tx_vec = p_tx_vec.astype(float)
    zeta_vec = zeta_vec.astype(float)
    bw = float(params.bandwidth_hz_at(mec_index))
    betas = np.array(
        [beta_u(params, float(s), float(g), float(pt), bw) for s, g, pt in zip(s_vec, g_vec, p_tx_vec)]
    )
    a = optimal_bandwidth_fractions(xm > 0.5, betas)
    eloc = (1.0 - xm) * zeta_vec * (f_hz_vec**2) * s_vec * params.cpu_cycles_per_bit
    rates = np.array(
        [
            shannon_rate_bps(float(ai), float(gi), params, float(pt), bw)
            for ai, gi, pt in zip(a, g_vec, p_tx_vec)
        ]
    )
    eoff = xm * p_tx_vec * s_vec / np.maximum(rates, 1e3)
    return float(np.sum(eloc + eoff))


def utility_ud(
    params: SimulationParams,
    s: float,
    g: float,
    on_mec: bool,
    a_frac: float,
    f_user_hz: float,
    p_tx_w: float,
    zeta_u: float,
    bandwidth_hz: float,
) -> float:
    if not on_mec:
        e = _local_energy(params, s, f_user_hz, zeta_u)
    else:
        e = _edge_energy(params, s, g, a_frac, p_tx_w, bandwidth_hz)
    return -float(e)


def server_utilities(
    params: SimulationParams,
    s_vec: np.ndarray,
    g_vec: np.ndarray,
    match: np.ndarray,
    a_frac: np.ndarray,
    f_hz_vec: np.ndarray,
    p_tx_vec: np.ndarray,
    zeta_vec: np.ndarray,
    bandwidth_hz: float,
) -> tuple[float, float]:
    """ϕ_i0, ϕ_i1（式 (38b) 的标量形式：服务器效用为所匹配 UD 能耗之和的相反数）。"""
    e0 = 0.0
    e1 = 0.0
    for k in range(len(s_vec)):
        if match[k] == IDX_LOCAL:
            e0 += _local_energy(params, float(s_vec[k]), float(f_hz_vec[k]), float(zeta_vec[k]))
        else:
            e1 += _edge_energy(
                params,
                float(s_vec[k]),
                float(g_vec[k]),
                float(a_frac[k]),
                float(p_tx_vec[k]),
                bandwidth_hz,
            )
    return -e0, -e1


def _alloc_for_match(match: np.ndarray, betas: np.ndarray) -> np.ndarray:
    xm = (match == IDX_MEC).astype(float)
    return optimal_bandwidth_fractions(xm > 0.5, betas)


def algorithm1_energy_matching(
    params: SimulationParams,
    mec_index: int,
    s_vec: np.ndarray,
    g_vec: np.ndarray,
    f_hz_vec: np.ndarray,
    p_tx_vec: np.ndarray,
    zeta_vec: np.ndarray,
    qi1: int,
    rng: np.random.Generator,
    max_swap_passes: int = 40,
    max_pair_tries_per_pass: int = 400,
) -> np.ndarray:
    """
    Algorithm 1：随机初始化 + Swap 至稳定 + 式 (41) 映射 + Removal。
    match[k] ∈ {IDX_LOCAL, IDX_MEC}
    """
    n_u = len(s_vec)
    if n_u == 0:
        return np.array([], dtype=int)

    match = rng.integers(0, 2, size=n_u)
    # 满足 MEC 容量 qi1
    def fix_capacity(m: np.ndarray) -> np.ndarray:
        m = m.copy()
        while np.sum(m == IDX_MEC) > qi1:
            idx = rng.choice(np.nonzero(m == IDX_MEC)[0])
            m[idx] = IDX_LOCAL
        return m

    match = fix_capacity(match)

    bw = float(params.bandwidth_hz_at(mec_index))
    betas = np.array([beta_u(params, float(s), float(g), float(pt), bw) for s, g, pt in zip(s_vec, g_vec, p_tx_vec)])

    def all_utilities(mch: np.ndarray) -> tuple[np.ndarray, float, float]:
        a = _alloc_for_match(mch, betas)
        phi_u = np.array(
            [
                utility_ud(
                    params,
                    float(s_vec[k]),
                    float(g_vec[k]),
                    mch[k] == IDX_MEC,
                    float(a[k]),
                    float(f_hz_vec[k]),
                    float(p_tx_vec[k]),
                    float(zeta_vec[k]),
                    bw,
                )
                for k in range(n_u)
            ]
        )
        p0, p1 = server_utilities(params, s_vec, g_vec, mch, a, f_hz_vec, p_tx_vec, zeta_vec, bw)
        return phi_u, p0, p1

    def srv_phi(p0: float, p1: float, idx: int) -> float:
        return p0 if idx == IDX_LOCAL else p1

    for _ in range(max_swap_passes):
        phi_u, phi0, phi1 = all_utilities(match)
        improved = False
        pairs: list[tuple[int, int]] = []
        if n_u <= 24:
            for k1 in range(n_u):
                for k2 in range(k1 + 1, n_u):
                    pairs.append((k1, k2))
        else:
            for _try in range(max_pair_tries_per_pass):
                k1 = int(rng.integers(0, n_u))
                k2 = int(rng.integers(0, n_u))
                if k1 != k2:
                    if k1 > k2:
                        k1, k2 = k2, k1
                    pairs.append((k1, k2))
        for k1, k2 in pairs:
            s1, s2 = int(match[k1]), int(match[k2])
            if s1 == s2:
                continue
            m_new = match.copy()
            m_new[k1], m_new[k2] = s2, s1
            if np.sum(m_new == IDX_MEC) > qi1:
                continue
            pu, p0n, p1n = all_utilities(m_new)
            old = [
                phi_u[k1],
                phi_u[k2],
                srv_phi(phi0, phi1, s1),
                srv_phi(phi0, phi1, s2),
            ]
            new = [
                pu[k1],
                pu[k2],
                srv_phi(p0n, p1n, s1),
                srv_phi(p0n, p1n, s2),
            ]
            if all(new[j] >= old[j] - 1e-9 for j in range(4)) and any(new[j] > old[j] + 1e-9 for j in range(4)):
                match = m_new
                improved = True
                break
        if not improved:
            break

    xm = (match == IDX_MEC).astype(float)
    # Removal（Definition 5）：若单独改回本地降低总能耗则执行
    changed = True
    while changed:
        changed = False
        betas = np.array([beta_u(params, float(s), float(g), float(pt), bw) for s, g, pt in zip(s_vec, g_vec, p_tx_vec)])
        a = _alloc_for_match(match, betas)
        e_vec = np.array(
            [
                _local_energy(params, float(s_vec[k]), float(f_hz_vec[k]), float(zeta_vec[k]))
                if match[k] == IDX_LOCAL
                else _edge_energy(
                    params, float(s_vec[k]), float(g_vec[k]), float(a[k]), float(p_tx_vec[k]), bw
                )
                for k in range(n_u)
            ]
        )
        E_tot = float(np.sum(e_vec))
        for k in range(n_u):
            if match[k] != IDX_MEC:
                continue
            m_try = match.copy()
            m_try[k] = IDX_LOCAL
            if np.sum(m_try == IDX_MEC) > qi1:
                continue
            a2 = _alloc_for_match(m_try, betas)
            e2 = np.array(
                [
                    _local_energy(params, float(s_vec[j]), float(f_hz_vec[j]), float(zeta_vec[j]))
                    if m_try[j] == IDX_LOCAL
                    else _edge_energy(
                        params, float(s_vec[j]), float(g_vec[j]), float(a2[j]), float(p_tx_vec[j]), bw
                    )
                    for j in range(n_u)
                ]
            )
            if float(np.sum(e2)) + 1e-9 < E_tot:
                match = m_try
                changed = True
                break
    return (match == IDX_MEC).astype(int)


def cloud_lp_coefficients(
    params: SimulationParams,
    QE: float,
    QC: float,
    ZE: float,
    ZC: float,
    t_slot: int,
    s_vec: np.ndarray,
) -> np.ndarray:
    """最小化式 (43)：对 y 的线性系数。"""
    t = max(t_slot, 1)
    c = np.zeros(len(s_vec), dtype=float)
    for k, s in enumerate(s_vec):
        su = float(s)
        c[k] = su * (
            -QE
            + QC
            - ZE * QE / t
            + ZC * QC / t
        )
    return c


def dependent_rounding_pair(
    x1: float,
    x2: float,
    eps1: float,
    eps2: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """式 (44)–(47) 单次舍入。"""
    i1 = min(1.0 - x1, (eps2 / max(eps1, 1e-18)) * x2)
    i2 = min(x1, (eps2 / max(eps1, 1e-18)) * (1.0 - x2))
    if i1 + i2 <= 1e-18:
        return x1, x2
    eta1 = i1 / (i1 + i2)
    eta2 = i2 / (i1 + i2)
    if eta1 >= eta2:
        return x1 - i2, x2 + i2 * eps1 / max(eps2, 1e-18)
    return x1 + i1, x2 - i1 * eps1 / max(eps2, 1e-18)


def algorithm2_cloud_decision(
    params: SimulationParams,
    QE: float,
    QC: float,
    ZE: float,
    ZC: float,
    t_slot: int,
    users_idx: np.ndarray,
    s_vec: np.ndarray,
    xm: np.ndarray,
    hist_AE_cum: float,
    hist_AC_cum: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Algorithm 2：线性松弛 + 依赖舍入。返回 x_{m→c} 向量（与 users_idx 对齐）。"""
    mask = xm > 0.5
    if not np.any(mask):
        return np.zeros_like(xm, dtype=float)

    s_sub = s_vec[mask]
    c = cloud_lp_coefficients(params, QE, QC, ZE, ZC, t_slot, s_sub)
    y = np.zeros_like(s_sub, dtype=float)
    # 闭式：min sum c y, y∈[0,1]
    y[c < 0.0] = 1.0
    y[c > 0.0] = 0.0
    amb = np.abs(c) < 1e-12
    y[amb] = 0.5

    y_work = y.copy()
    while True:
        frac = np.nonzero((y_work > 1e-9) & (y_work < 1.0 - 1e-9))[0]
        if len(frac) == 0:
            break
        if len(frac) == 1:
            u0 = int(frac[0])
            y_work[u0] = 0.0 if c[u0] >= 0 else 1.0
            break
        u1, u2 = int(frac[0]), int(frac[1])
        eps1, eps2 = float(s_sub[u1]), float(s_sub[u2])
        new1, new2 = dependent_rounding_pair(float(y_work[u1]), float(y_work[u2]), eps1, eps2, rng)
        y_work[u1], y_work[u2] = new1, new2

    out = np.zeros_like(xm, dtype=float)
    out[mask] = (y_work >= 0.5).astype(float)
    return out


def Q_function(
    params: SimulationParams,
    QE: float,
    QC: float,
    ZE: float,
    ZC: float,
    t_slot: int,
    s_vec: np.ndarray,
    xm: np.ndarray,
    xmc: np.ndarray,
    hist_AE_cum: float,
    hist_AC_cum: float,
    mec_index: int,
) -> float:
    """式 (34) Q(Xm,Xc)；f_m、r_m^c 取下标 m。"""
    t = max(t_slot, 1)
    ae = float(np.sum((xm - xmc) * s_vec))
    ac = float(np.sum(xmc * s_vec))
    fm_term = params.f_mec_hz_at(mec_index) * params.slot_duration / params.cpu_cycles_per_bit
    rc_term = params.rate_mec_to_cloud_at(mec_index) * params.slot_duration
    q = QE * (ae - fm_term) + QC * (ac - rc_term)
    q += ZE * QE / t * (hist_AE_cum + ae)
    q += ZC * QC / t * (hist_AC_cum + ac)
    return float(q)


def algorithm3_two_stage(
    state: SystemState,
    m: int,
    users: np.ndarray,
    s_vec: np.ndarray,
    g_vec: np.ndarray,
    f_hz_vec: np.ndarray,
    t_slot: int,
    hist_AE_cum: float,
    hist_AC_cum: float,
    mode: Literal["ojcta", "ecf"] = "ojcta",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Algorithm 3：n 从 Cm 递减，最小化 J = V·E + Q（论文叙述）。
    返回 (xm, xmc) 与 users 对齐。
    """
    p = state.params
    rng = state.rng
    p_tx_vec = state.p_transmit_w[users].astype(float)
    zeta_vec = state.zeta_u[users].astype(float)
    cap_m = p.connection_capacity_at(m)
    best_J = float("inf")
    best_xm = np.zeros(len(users), dtype=int)
    best_xmc = np.zeros(len(users), dtype=float)

    if mode == "ecf":
        xm = algorithm1_energy_matching(p, m, s_vec, g_vec, f_hz_vec, p_tx_vec, zeta_vec, cap_m, rng)
        xmc = np.zeros_like(xm, dtype=float)
        return xm.astype(int), xmc

    if p.fast_alg3:
        n_list = [cap_m, max(1, cap_m // 2), 1]
        n_list = sorted(set(n_list), reverse=True)
    else:
        n_list = list(range(cap_m, 0, -1))

    for n in n_list:
        xm = algorithm1_energy_matching(p, m, s_vec, g_vec, f_hz_vec, p_tx_vec, zeta_vec, n, rng)
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
            hist_AE_cum,
            hist_AC_cum,
            rng,
        )
        E = total_ud_energy_um(p, s_vec, g_vec, xm.astype(float), f_hz_vec, p_tx_vec, zeta_vec, m)
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
            hist_AE_cum,
            hist_AC_cum,
            m,
        )
        J = p.V * E + Q
        if J < best_J:
            best_J = J
            best_xm = xm.copy()
            best_xmc = xmc.copy()
    return best_xm, best_xmc


def run_timeslot_ojcta(
    state: SystemState,
    t_slot: int,
    task_bits: np.ndarray,
    ncc: bool = False,
    ecf: bool = False,
) -> tuple[float, float, float, float]:
    """
    Algorithm 4 单时隙。返回 (全网约能耗 J/时隙, 边缘 Q^E 型时延, 云 Q^C 型时延, 本地积压等效时延)。
    能耗为所有 UD 在本时隙之和；若论文纵轴为「每 UD 平均能耗」，需在绘图处除以 ``num_ud``。
    ncc: 强制无云协同；ecf: Energy considered first 基线。
    """
    p = state.params
    U = task_bits.shape[0]
    x_mec = np.zeros(U, dtype=int)
    x_cld = np.zeros(U, dtype=float)

    advance_slot_shadowing_chi_um(state, p)

    g_all = np.zeros(U, dtype=float)
    for u in range(U):
        m = int(state.user_mec[u])
        d = float(np.linalg.norm(state.user_pos[u] - state.bs_pos[m]))
        chi_link: float | None = None
        if p.shadowing_ar1_rho > 1e-12:
            chi_link = float(state.chi_um_db[u, m])
        g_all[u] = float(
            rayleigh_channel_gain(p, np.array([max(d, p.d_min_m)]), state.rng, chi_db_link=chi_link)[0]
        )

    total_e = 0.0
    for m in range(p.num_mec):
        users = state.users_on_mec(m)
        if len(users) == 0:
            continue
        s_vec = task_bits[users].astype(float)
        g_vec = g_all[users]
        f_hz_vec = state.f_user_hz[users]

        hist_ae = state.sum_AE_hist[m]
        hist_ac = state.sum_AC_hist[m]

        mode = "ecf" if ecf else "ojcta"
        xm, xmc = algorithm3_two_stage(
            state, m, users, s_vec, g_vec, f_hz_vec, t_slot, hist_ae, hist_ac, mode=mode
        )
        x_mec[users] = xm
        x_cld[users] = 0.0 if ncc else xmc
        total_e += total_ud_energy_um(
            p,
            s_vec,
            g_vec,
            xm.astype(float),
            f_hz_vec,
            state.p_transmit_w[users],
            state.zeta_u[users],
            m,
        )

    from .dynamics import advance_queues_virtual_and_mobility

    delay_edge, delay_cloud, delay_local = advance_queues_virtual_and_mobility(
        state, p, t_slot, task_bits, x_mec, x_cld
    )
    return total_e, delay_edge, delay_cloud, delay_local
