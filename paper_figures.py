"""
复现论文 Fig.3–6 风格曲线（与 ``params_paper_figures()`` 对齐：δ=1 s，U=200，¯D^E/¯D^C 等）。

仿真返回的能耗 ``e`` 为**全网约能耗**（各 UD 当隙之和）。若论文纵轴「Average UD energy」指
**(1/U)·Σ E_u**，绘图时应除以 ``num_ud`` 再与原文读数对比。

Fig.4(b) 对 LC/RO：边缘 Little 时延 + 本地积压等效时延（文献中 LC 的 Q^E 曲线上升主要来自系统级积压）。

用法：
  python -m ojcta_mec.paper_figures --outdir .

Fig.3 **默认**为 OJCTA 蒙特卡洛仿真（多 replica seed 取平均，与 ``params_paper_figures()`` 一致，含全 n 扫描）：
  python -m ojcta_mec.paper_figures --only fig3 --outdir .\\figs
  仅用 IEEE 图上的扫描点画图（非仿真）：加 ``--fig3-use-paper-points``。

Fig.5（全策略仿真 + 论文式 1×3 与 inset）：
  python -m ojcta_mec.paper_figures --only fig5 --outdir .\\figs
  快速试跑可减小 ``--fig5-num-slots`` / ``--fig5-num-ud``。

Fig.7：各 MEC 物理队列 ``QE``、``QC``（比特积压）随时间变化；含全卸载基线 ``FO`` 与多策略对比：
  python -m ojcta_mec.paper_figures --only fig7 --outdir .\\figs
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 论文出图用非交互后端，避免 Windows 下默认 Tk 崩溃

import numpy as np

from .baselines import (
    run_timeslot_fo,
    run_timeslot_gjtora,
    run_timeslot_lc,
    run_timeslot_ncc,
    run_timeslot_ro,
    run_timeslot_ssc,
)
from .ojcta import run_timeslot_ojcta
from dataclasses import replace

from .params import SimulationParams, params_paper_figures
from .system_state import SystemState


def qe_delay_plot(policy: str, de: float, dl: float) -> float:
    """与图示接近：OJCTA 等以边缘 Q^E 为主；LC/RO 叠加本地积压等效时延。"""
    if policy in ("LC", "RO"):
        return float(de + dl)
    return float(de)


def run_policy_slot(policy: str, state: SystemState, t: int, task: np.ndarray) -> tuple[float, float, float, float]:
    if policy == "OJCTA":
        return run_timeslot_ojcta(state, t, task, ncc=False, ecf=False)
    if policy == "ECF":
        return run_timeslot_ojcta(state, t, task, ncc=False, ecf=True)
    if policy == "GJTORA":
        return run_timeslot_gjtora(state, t, task)
    if policy == "LC":
        return run_timeslot_lc(state, t, task)
    if policy == "RO":
        return run_timeslot_ro(state, t, task)
    if policy == "SSC":
        return run_timeslot_ssc(state, t, task)
    if policy == "NCC":
        return run_timeslot_ncc(state, t, task)
    if policy == "FO":
        return run_timeslot_fo(state, t, task)
    raise ValueError(policy)


# Fig.3：IEEE 图上的扫描点（仅 ``--fig3-use-paper-points`` 时使用；非蒙特卡洛复现）
FIG3_V = np.array([5, 10, 15, 20, 25, 30, 35, 40], dtype=float)
FIG3_ENERGY_J = np.array([3.39, 3.30, 3.23, 3.17, 3.13, 3.08, 3.04, 3.035])
FIG3_DELAY_EDGE_S = np.array([0.90, 1.10, 1.20, 1.30, 1.25, 1.40, 1.60, 1.70])
FIG3_DELAY_CLOUD_S = np.array([0.60, 0.70, 0.72, 0.73, 0.72, 0.82, 0.95, 1.00])


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        out.append(int(part))
    return out


def fig3_v_tradeoff(
    outdir: Path,
    seed: int = 42,
    from_simulation: bool = True,
    fig3_num_ud: int | None = None,
    fig3_num_slots: int | None = None,
    replica_seeds: Sequence[int] | None = None,
) -> None:
    """
    Fig.3：Lyapunov 因子 V 与能耗/排队时延权衡。

    默认 ``from_simulation=True``：``params_paper_figures()`` + OJCTA，对每个 V 在多个
    replica seed 上各跑一条轨迹，再对尾部时隙均值做算术平均（降低方差）。

    ``from_simulation=False``：仅绘制 IEEE 图上的离散点（``FIG3_*``），非仿真复现。
    ``fig3_num_ud`` / ``fig3_num_slots`` 为 None 时用 ``params_paper_figures()`` 中的 U、T。
    """
    import matplotlib.pyplot as plt

    if from_simulation:
        pp = params_paper_figures()
        V_list = list(range(5, 45, 5))
        n_slots = int(fig3_num_slots if fig3_num_slots is not None else pp.num_slots)
        n_ud = int(fig3_num_ud if fig3_num_ud is not None else pp.num_ud)
        tail_win = min(180, max(80, n_slots - 60))
        reps = list(replica_seeds) if replica_seeds is not None else [0, 1, 2, 3, 4]
        if not reps:
            reps = [0]
        eng: list[float] = []
        dqe: list[float] = []
        dqc: list[float] = []
        for V in V_list:
            run_e: list[float] = []
            run_qe: list[float] = []
            run_qc: list[float] = []
            for rep in reps:
                p = replace(pp, V=float(V), num_slots=n_slots, num_ud=n_ud)
                rng = np.random.default_rng(int(seed) + int(rep) * 100_003 + int(V) * 10_007)
                st = SystemState(p, rng)
                es: list[float] = []
                qes: list[float] = []
                qcs: list[float] = []
                for t in range(1, p.num_slots + 1):
                    task = p.sample_task_bits(rng)
                    e, de, dc, dl = run_timeslot_ojcta(st, t, task, ncc=False, ecf=False)
                    es.append(e)
                    qes.append(qe_delay_plot("OJCTA", de, dl))
                    qcs.append(dc)
                tw = min(tail_win, len(es))
                run_e.append(float(np.mean(es[-tw:])))
                run_qe.append(float(np.mean(qes[-tw:])))
                run_qc.append(float(np.mean(qcs[-tw:])))
            eng.append(float(np.mean(run_e)))
            dqe.append(float(np.mean(run_qe)))
            dqc.append(float(np.mean(run_qc)))
        v_plot = np.asarray(V_list, dtype=float)
        e_plot = np.asarray(eng, dtype=float)
        qe_plot = np.asarray(dqe, dtype=float)
        qc_plot = np.asarray(dqc, dtype=float)
    else:
        v_plot = FIG3_V
        e_plot = FIG3_ENERGY_J
        qe_plot = FIG3_DELAY_EDGE_S
        qc_plot = FIG3_DELAY_CLOUD_S

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.set_xlabel("The value of $V$")
    ax1.set_xlim(5, 40)
    ax1.set_xticks(np.arange(5, 45, 5))
    ax1.set_ylabel("Average energy consumption (J)", color="green", fontsize=11)
    if from_simulation:
        e_lo, e_hi = float(np.min(e_plot)), float(np.max(e_plot))
        e_pad = max((e_hi - e_lo) * 0.12, 0.02 * max(abs(e_hi), 1.0))
        ax1.set_ylim(e_lo - e_pad, e_hi + e_pad)
    else:
        ax1.set_ylim(3.0, 3.4)
        ax1.yaxis.set_ticks(np.arange(3.0, 3.45, 0.1))
    (ln_e,) = ax1.plot(
        v_plot,
        e_plot,
        color="green",
        linestyle="-",
        marker="o",
        linewidth=1.8,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.6,
        markeredgecolor="green",
        label="UD energy consumption",
        zorder=3,
    )
    ax1.tick_params(axis="y", labelcolor="green")
    ax1.grid(True, linestyle="-", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Average queuing delay (s)", color="darkorange", fontsize=11)
    if from_simulation:
        d_lo = float(min(np.min(qe_plot), np.min(qc_plot)))
        d_hi = float(max(np.max(qe_plot), np.max(qc_plot)))
        d_pad = max((d_hi - d_lo) * 0.12, 0.05)
        ax2.set_ylim(max(0.0, d_lo - d_pad), d_hi + d_pad)
    else:
        ax2.set_ylim(0.5, 2.0)
        ax2.yaxis.set_ticks([0.5, 1.0, 1.5, 2.0])
    (ln_qe,) = ax2.plot(
        v_plot,
        qe_plot,
        color="darkorange",
        linestyle="--",
        marker="^",
        linewidth=1.8,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.4,
        markeredgecolor="darkorange",
        label="Queuing delay of edge computing",
        zorder=2,
    )
    (ln_qc,) = ax2.plot(
        v_plot,
        qc_plot,
        color="darkorange",
        linestyle=":",
        marker="x",
        linewidth=1.8,
        markersize=8,
        markeredgewidth=1.6,
        markeredgecolor="darkorange",
        label="Queuing delay of cloud computing",
        zorder=2,
    )
    ax2.tick_params(axis="y", labelcolor="darkorange")

    fig.legend(
        handles=[ln_e, ln_qe, ln_qc],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=3,
        frameon=True,
        fontsize=9,
    )
    title = "Fig. 3. Impact of Lyapunov penalty factor $V$ on system performance."
    if from_simulation:
        reps = list(replica_seeds) if replica_seeds is not None else [0, 1, 2, 3, 4]
        title += f" (OJCTA simulation, mean of {len(reps)} runs)"
    else:
        title += " (IEEE published scan points, not simulated)"
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(outdir / "fig3_V_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig4_delay_qe_display(
    pol: str,
    de: float,
    dl: float,
    t_slot: int,
    paper_ecf_ro_ramp: bool,
) -> float:
    """(b) 主图：ECF/RO 与原文一致的发散由「仿真 de + 线性项」近似（论文中二者不稳定）。"""
    base = qe_delay_plot(pol, de, dl)
    if not paper_ecf_ro_ramp:
        return float(base)
    if pol == "ECF":
        return float(min(35.0, base + 0.138 * float(t_slot)))
    if pol == "RO":
        return float(min(22.0, base + 0.098 * float(t_slot)))
    return float(base)


def _fig4_delay_qc_display(pol: str, dc: float, t_slot: int, paper_ecf_ro_ramp: bool) -> float:
    if not paper_ecf_ro_ramp:
        return float(dc)
    if pol == "ECF":
        return float(min(25.0, dc + 0.115 * float(t_slot)))
    if pol == "RO":
        return float(min(22.5, dc + 0.098 * float(t_slot)))
    return float(dc)


def fig4_time_slots(
    outdir: Path,
    seed: int = 42,
    num_ud: int = 100,
    num_slots: int = 200,
    paper_ecf_ro_ramp: bool = False,
) -> None:
    """
    Fig.4：与论文布局一致——(a)(b)(c) 三子图、(b)(c) 主图 + inset。

    ``paper_ecf_ro_ramp``：若为 True，对 ECF/RO 的 $Q^E$/$Q^C$ 在主图叠加线性发散项，
    仅用于复现旧版「与原文曲线形态接近」的视觉效果；默认 False，与 Table I 参数下纯仿真一致。
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    p = replace(
        params_paper_figures(),
        num_slots=int(num_slots),
        num_ud=int(num_ud),
    )
    policies = ["OJCTA", "LC", "RO", "FO", "ECF", "SSC", "NCC", "GJTORA"]
    series = {pol: {"e": [], "qe": [], "qc": []} for pol in policies}

    for pol in policies:
        rng = np.random.default_rng(seed + abs(hash(pol)) % 100000)
        st = SystemState(p, rng)
        for t in range(1, p.num_slots + 1):
            task = p.sample_task_bits(rng)
            e, de, dc, dl = run_policy_slot(pol, st, t, task)
            series[pol]["e"].append(e)
            series[pol]["qe"].append(_fig4_delay_qe_display(pol, de, dl, t, paper_ecf_ro_ramp))
            series[pol]["qc"].append(_fig4_delay_qc_display(pol, dc, t, paper_ecf_ro_ramp))

    slots = np.arange(1, p.num_slots + 1, dtype=float)
    x_plot = slots * float(p.slot_duration)

    # 与 IEEE 图例接近的配色/点型
    styles: dict[str, tuple[str, str, float]] = {
        "OJCTA": ("red", "o", 4.0),
        "LC": ("gold", "D", 4.0),
        "RO": ("forestgreen", "^", 4.0),
        "FO": ("chocolate", "P", 4.5),
        "ECF": ("green", "s", 4.0),
        "SSC": ("purple", "v", 4.0),
        "NCC": ("cyan", "*", 5.0),
        "GJTORA": ("darkorange", "x", 5.0),
    }

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.5), sharex=True)

    def plot_panel(
        ax: plt.Axes,
        key: str,
        ylim: tuple[float, float],
        title: str,
        ylabel: str,
        leg_loc: str = "upper right",
    ) -> None:
        for pol in policies:
            c, mk, ms = styles[pol]
            y = np.asarray(series[pol][key], dtype=float)
            if key != "e":
                k = min(15, max(3, len(y) // 15))
                y = np.convolve(y, np.ones(k) / k, mode="same")
            ax.plot(
                x_plot,
                y,
                color=c,
                linestyle="-",
                marker=mk,
                markersize=ms,
                linewidth=1.15,
                label=pol,
                markevery=max(1, len(x_plot) // 18),
                alpha=0.95,
            )
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.35)
        ax.legend(loc=leg_loc, fontsize=7, ncol=2, framealpha=0.9)

    plot_panel(
        axes[0],
        "e",
        (2.0, 6.0),
        "(a) Average UD energy consumption (J)",
        "Average UD energy consumption (J)",
    )

    axb = axes[1]
    plot_panel(
        axb,
        "qe",
        (0.0, 35.0),
        r"(b) Average queuing delay of $Q_m^E$ (s)",
        r"Average queuing delay of $Q_m^E$ (s)",
        leg_loc="upper left",
    )
    axins_b = axb.inset_axes([0.52, 0.12, 0.38, 0.42])
    for pol in policies:
        if pol in ("ECF", "RO"):
            continue
        c, mk, ms = styles[pol]
        y = np.asarray(series[pol]["qe"], dtype=float)
        k = min(15, max(3, len(y) // 15))
        y = np.convolve(y, np.ones(k) / k, mode="same")
        axins_b.plot(x_plot, y, color=c, marker=mk, markersize=ms - 1, linewidth=1, markevery=14)
    axins_b.set_xlim(0, float(x_plot[-1]))
    axins_b.set_ylim(1.2, 1.4)
    axins_b.set_title("Inset", fontsize=8)
    axins_b.grid(True, alpha=0.3)
    mark_inset(axb, axins_b, loc1=2, loc2=4, fc="none", ec="0.45", linestyle="--")

    axc = axes[2]
    plot_panel(
        axc,
        "qc",
        (0.0, 25.0),
        r"(c) Average queuing delay of $Q_m^C$ (s)",
        r"Average queuing delay of $Q_m^C$ (s)",
        leg_loc="upper left",
    )
    axins_c = axc.inset_axes([0.52, 0.12, 0.38, 0.42])
    for pol in policies:
        if pol in ("ECF", "RO"):
            continue
        c, mk, ms = styles[pol]
        y = np.asarray(series[pol]["qc"], dtype=float)
        k = min(15, max(3, len(y) // 15))
        y = np.convolve(y, np.ones(k) / k, mode="same")
        axins_c.plot(x_plot, y, color=c, marker=mk, markersize=ms - 1, linewidth=1, markevery=14)
    axins_c.set_xlim(0, float(x_plot[-1]))
    axins_c.set_ylim(0.7, 0.9)
    axins_c.set_title("Inset", fontsize=8)
    axins_c.grid(True, alpha=0.3)
    mark_inset(axc, axins_c, loc1=2, loc2=4, fc="none", ec="0.45", linestyle="--")

    axes[-1].set_xlabel("Time slot (s)")
    fig.suptitle("Fig. 4. System performance with time slots.", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_time_slots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig5_mean_tail(vals: list[float], tail: int) -> float:
    if not vals:
        return float("nan")
    w = min(len(vals), max(1, tail))
    return float(np.mean(np.asarray(vals[-w:], dtype=float)))


def _fig5_ylim_pad(yvals: list[float], pad_ratio: float = 0.12) -> tuple[float, float]:
    y = np.asarray(yvals, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.min(y)), float(np.max(y))
    if hi <= lo:
        d = max(abs(lo) * 0.05, 0.05)
        return lo - d, hi + d
    d = (hi - lo) * pad_ratio + 1e-9
    return lo - d, hi + d


def fig5_bandwidth(
    outdir: Path,
    seed: int = 42,
    num_ud: int = 200,
    num_slots: int = 220,
    tail_slots: int = 120,
) -> None:
    """
    Fig.5：各带宽下跑满 ``num_slots`` 仿真，对尾部 ``tail_slots`` 时隙取平均；
    排版与论文一致——1×3、(a) 能耗、(b)(c) 主图 + inset（(b) 为 5–10 MHz 与 15–20 MHz 双放大）。
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    bw_mhz = np.array([5, 10, 15, 20], dtype=float)
    policies = ["OJCTA", "LC", "RO", "FO", "ECF", "SSC", "NCC", "GJTORA"]
    res: dict[str, dict[str, list[float]]] = {pol: {"e": [], "qe": [], "qc": []} for pol in policies}

    tail = int(min(max(1, tail_slots), num_slots))

    for bw in bw_mhz:
        for pol in policies:
            p = replace(
                params_paper_figures(),
                num_slots=int(num_slots),
                num_ud=int(num_ud),
                bandwidth_hz=float(bw * 1e6),
            )
            rng = np.random.default_rng(seed + int(bw) * 17 + abs(hash(pol)) % 100000)
            st = SystemState(p, rng)
            es: list[float] = []
            qes: list[float] = []
            qcs: list[float] = []
            for t in range(1, p.num_slots + 1):
                task = p.sample_task_bits(rng)
                e, de, dc, dl = run_policy_slot(pol, st, t, task)
                es.append(e)
                qes.append(qe_delay_plot(pol, de, dl))
                qcs.append(dc)
            res[pol]["e"].append(_fig5_mean_tail(es, tail))
            res[pol]["qe"].append(_fig5_mean_tail(qes, tail))
            res[pol]["qc"].append(_fig5_mean_tail(qcs, tail))

    styles: dict[str, tuple[str, str, float]] = {
        "OJCTA": ("red", "o", 4.5),
        "LC": ("gold", "D", 4.0),
        "RO": ("forestgreen", "^", 4.0),
        "FO": ("chocolate", "P", 4.5),
        "ECF": ("green", "s", 4.0),
        "SSC": ("purple", "v", 4.0),
        "NCC": ("cyan", "*", 5.0),
        "GJTORA": ("darkorange", "x", 5.0),
    }

    def plot_vs_bw(ax: plt.Axes, key: str, ylim: tuple[float, float], legend_loc: str) -> None:
        for pol in policies:
            c, mk, ms = styles[pol]
            y = np.asarray(res[pol][key], dtype=float)
            ax.plot(
                bw_mhz,
                y,
                color=c,
                linestyle="-",
                marker=mk,
                markersize=ms,
                linewidth=1.15,
                label=pol,
                alpha=0.95,
            )
        ax.set_xlabel("Bandwidth (MHz)")
        ax.set_xlim(float(bw_mhz[0]), float(bw_mhz[-1]))
        ax.set_xticks(bw_mhz)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.35)
        ax.legend(loc=legend_loc, fontsize=7, ncol=2, framealpha=0.92)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    axa = axes[0]
    plot_vs_bw(axa, "e", (2.0, 6.0), "lower left")
    axa.set_title("(a) Average UD energy consumption (J)", fontsize=10)
    axa.set_ylabel("Average UD energy consumption (J)")

    axb = axes[1]
    plot_vs_bw(axb, "qe", (0.0, 35.0), "upper left")
    axb.set_title(r"(b) Average queuing delay of $Q_m^E$ (s)", fontsize=10)
    axb.set_ylabel(r"Average queuing delay of $Q_m^E$ (s)")

    stable_pol = [p for p in policies if p not in ("ECF", "RO", "FO")]
    # Inset 1：5–10 MHz（低时延区）
    y_b1: list[float] = []
    for pol in stable_pol:
        y_b1.extend([res[pol]["qe"][0], res[pol]["qe"][1]])
    y1lo, y1hi = _fig5_ylim_pad(y_b1, 0.15)
    axins_b1 = axb.inset_axes([0.06, 0.58, 0.34, 0.36])
    for pol in policies:
        c, mk, ms = styles[pol]
        axins_b1.plot(
            bw_mhz[:2],
            np.asarray(res[pol]["qe"][:2], dtype=float),
            color=c,
            marker=mk,
            markersize=ms - 0.5,
            linewidth=1.0,
        )
    axins_b1.set_xlim(5.0, 10.0)
    axins_b1.set_ylim(y1lo, y1hi)
    axins_b1.grid(True, alpha=0.3)
    axins_b1.tick_params(labelsize=7)
    mark_inset(axb, axins_b1, loc1=2, loc2=3, fc="none", ec="0.45", linestyle="--")

    # Inset 2：15–20 MHz
    y_b2: list[float] = []
    for pol in stable_pol:
        y_b2.extend([res[pol]["qe"][2], res[pol]["qe"][3]])
    y2lo, y2hi = _fig5_ylim_pad(y_b2, 0.12)
    axins_b2 = axb.inset_axes([0.58, 0.12, 0.36, 0.38])
    for pol in policies:
        c, mk, ms = styles[pol]
        axins_b2.plot(
            bw_mhz[2:],
            np.asarray(res[pol]["qe"][2:], dtype=float),
            color=c,
            marker=mk,
            markersize=ms - 0.5,
            linewidth=1.0,
        )
    axins_b2.set_xlim(15.0, 20.0)
    axins_b2.set_ylim(y2lo, y2hi)
    axins_b2.grid(True, alpha=0.3)
    axins_b2.tick_params(labelsize=7)
    mark_inset(axb, axins_b2, loc1=1, loc2=4, fc="none", ec="0.45", linestyle="--")

    axc = axes[2]
    plot_vs_bw(axc, "qc", (0.0, 25.0), "upper left")
    axc.set_title(r"(c) Average queuing delay of $Q_m^C$ (s)", fontsize=10)
    axc.set_ylabel(r"Average queuing delay of $Q_m^C$ (s)")

    y_c: list[float] = []
    for pol in stable_pol:
        y_c.extend(res[pol]["qc"])
    yclo, ychi = _fig5_ylim_pad(y_c, 0.12)
    axins_c = axc.inset_axes([0.52, 0.12, 0.38, 0.42])
    for pol in policies:
        c, mk, ms = styles[pol]
        axins_c.plot(
            bw_mhz,
            np.asarray(res[pol]["qc"], dtype=float),
            color=c,
            marker=mk,
            markersize=ms - 0.5,
            linewidth=1.0,
        )
    axins_c.set_xlim(float(bw_mhz[0]), float(bw_mhz[-1]))
    axins_c.set_ylim(yclo, ychi)
    axins_c.grid(True, alpha=0.3)
    axins_c.tick_params(labelsize=7)
    mark_inset(axc, axins_c, loc1=2, loc2=4, fc="none", ec="0.45", linestyle="--")

    fig.suptitle("Fig. 5. System performance with bandwidth.", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    fig.savefig(outdir / "fig5_bandwidth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig6_num_ud(outdir: Path, seed: int = 42) -> None:
    import matplotlib.pyplot as plt

    n_candidates = [30, 35, 40, 45, 50, 55, 60]
    n_ok = [n for n in n_candidates if n % 4 == 0]
    policies = ["OJCTA", "LC", "RO", "FO", "ECF", "SSC", "NCC", "GJTORA"]
    res = {pol: {"e": [], "qe": [], "qc": []} for pol in policies}

    for n_ud in n_ok:
        for pol in policies:
            p = replace(params_paper_figures(), num_slots=200, num_ud=int(n_ud))
            rng = np.random.default_rng(seed + n_ud + hash(pol) % 999)
            st = SystemState(p, rng)
            es, qes, qcs = [], [], []
            for t in range(1, p.num_slots + 1):
                task = p.sample_task_bits(rng)
                e, de, dc, dl = run_policy_slot(pol, st, t, task)
                es.append(e)
                qes.append(qe_delay_plot(pol, de, dl))
                qcs.append(dc)
            res[pol]["e"].append(float(np.mean(es[-100:])))
            res[pol]["qe"].append(float(np.mean(qes[-100:])))
            res[pol]["qc"].append(float(np.mean(qcs[-100:])))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    styles = {
        "OJCTA": ("red", "o"),
        "LC": ("goldenrod", "D"),
        "RO": ("blue", "s"),
        "FO": ("darkorange", "P"),
        "ECF": ("green", "^"),
        "SSC": ("magenta", ">"),
        "NCC": ("cyan", "v"),
        "GJTORA": ("brown", "x"),
    }
    titles = [
        "(a) Average UD energy consumption (J)",
        r"(b) Average queuing delay of $Q^E$ (s)",
        r"(c) Average queuing delay of $Q^C$ (s)",
    ]
    keys = ["e", "qe", "qc"]
    for ax, title, key in zip(axes, titles, keys):
        for pol in policies:
            c, mk = styles[pol]
            ax.plot(n_ok, res[pol][key], color=c, marker=mk, label=pol, linewidth=1.5)
        ax.set_xlabel("Number of UDs")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Fig.6-style: system performance with number of UDs", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "fig6_num_ud.png", dpi=150)
    plt.close(fig)


def fig7_queue_backlog_bits(
    outdir: Path,
    seed: int = 42,
    num_ud: int = 100,
    num_slots: int = 200,
) -> None:
    """
    物理队列 ``QE_m``、``QC_m`` 的原始比特积压随时间演化。

    多策略子图 (a)(b) 共用**同一**任务到达序列与**同一**初始状态 RNG 种子，仅策略不同，便于公平对比。
    (c)(d) 展示 OJCTA 下各 MEC 的分队列积压。

    为缩短出图时间，本图在参数上固定 ``fast_alg3=True``（Algorithm 3 仅扫 n∈{Cm,⌊Cm/2⌋,1}）。
    若需与论文全 n 扫描一致，可另写脚本在关闭 ``fast_alg3`` 下记录 ``state.QE``/``QC``。
    """
    import matplotlib.pyplot as plt

    p = replace(
        params_paper_figures(),
        num_ud=int(num_ud),
        num_slots=int(num_slots),
        fast_alg3=True,
    )
    rng_tasks = np.random.default_rng(int(seed) + 90211)
    tasks: list[np.ndarray] = [p.sample_task_bits(rng_tasks) for _ in range(p.num_slots)]

    policies = ["OJCTA", "FO", "RO", "LC"]
    colors = {"OJCTA": "red", "FO": "darkorange", "RO": "steelblue", "LC": "goldenrod"}

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True)
    slots_x = np.arange(1, p.num_slots + 1, dtype=float) * float(p.slot_duration)

    sum_series: dict[str, tuple[list[float], list[float]]] = {}
    ojcta_per_mec: tuple[list[list[float]], list[list[float]]] | None = None
    for pol in policies:
        rng_state = np.random.default_rng(int(seed))
        st = SystemState(p, rng_state)
        sqe: list[float] = []
        sqc: list[float] = []
        qe_m: list[list[float]] | None = [[] for _ in range(p.num_mec)] if pol == "OJCTA" else None
        qc_m: list[list[float]] | None = [[] for _ in range(p.num_mec)] if pol == "OJCTA" else None
        for t in range(1, p.num_slots + 1):
            run_policy_slot(pol, st, t, tasks[t - 1])
            sqe.append(float(np.sum(st.QE)))
            sqc.append(float(np.sum(st.QC)))
            if pol == "OJCTA" and qe_m is not None and qc_m is not None:
                for m in range(p.num_mec):
                    qe_m[m].append(float(st.QE[m]))
                    qc_m[m].append(float(st.QC[m]))
        sum_series[pol] = (sqe, sqc)
        if pol == "OJCTA" and qe_m is not None and qc_m is not None:
            ojcta_per_mec = (qe_m, qc_m)

    ax00, ax01 = axes[0, 0], axes[0, 1]
    for pol in policies:
        c = colors[pol]
        ax00.plot(slots_x, sum_series[pol][0], label=pol, color=c, linewidth=1.2)
        ax01.plot(slots_x, sum_series[pol][1], label=pol, color=c, linewidth=1.2)
    ax00.set_ylabel(r"$\sum_m Q_m^E$ (bits)")
    ax00.set_title("(a) Total edge computing backlog")
    ax00.legend(fontsize=8, loc="best")
    ax00.grid(True, alpha=0.3)
    ax01.set_ylabel(r"$\sum_m Q_m^C$ (bits)")
    ax01.set_title("(b) Total cloud-ward backlog")
    ax01.legend(fontsize=8, loc="best")
    ax01.grid(True, alpha=0.3)

    if ojcta_per_mec is None:
        raise RuntimeError("fig7: missing OJCTA per-MEC series")
    qe_rows, qc_rows = ojcta_per_mec

    ax10, ax11 = axes[1, 0], axes[1, 1]
    cm = plt.cm.tab10(np.linspace(0, 0.85, p.num_mec))
    for m in range(p.num_mec):
        ax10.plot(slots_x, qe_rows[m], color=cm[m], label=f"MEC {m}", linewidth=1.1)
        ax11.plot(slots_x, qc_rows[m], color=cm[m], label=f"MEC {m}", linewidth=1.1)
    ax10.set_ylabel(r"$Q_m^E$ (bits)")
    ax10.set_title(r"(c) OJCTA: per-server $Q_m^E$")
    ax10.legend(fontsize=7, ncol=2, loc="best")
    ax10.grid(True, alpha=0.3)
    ax11.set_ylabel(r"$Q_m^C$ (bits)")
    ax11.set_title(r"(d) OJCTA: per-server $Q_m^C$")
    ax11.legend(fontsize=7, ncol=2, loc="best")
    ax11.grid(True, alpha=0.3)
    for ax in (ax10, ax11):
        ax.set_xlabel(f"Time (s), slot $\\delta={p.slot_duration}$ s")

    fig.suptitle(r"Physical queue backlogs $Q^E$, $Q^C$ (bits)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "fig7_queue_backlog_bits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--only", type=str, default="all", help="fig3|fig4|fig5|fig6|fig7|all")
    ap.add_argument(
        "--fig3-use-paper-points",
        action="store_true",
        help="Fig.3 使用 IEEE 图上的离散数值点画图（非仿真）；默认用 OJCTA 仿真",
    )
    ap.add_argument(
        "--fig3-num-ud",
        type=int,
        default=None,
        help="Fig.3 仿真 UD 数（默认与 params_paper_figures 一致，200；须被 4 整除）",
    )
    ap.add_argument(
        "--fig3-num-slots",
        type=int,
        default=None,
        help="Fig.3 仿真每个 V 的时隙数（默认与 params_paper_figures 一致）",
    )
    ap.add_argument(
        "--fig3-seed",
        type=int,
        default=42,
        help="Fig.3 仿真基础随机种子（与各 replica 组合）",
    )
    ap.add_argument(
        "--fig3-replica-seeds",
        type=str,
        default="0,1,2,3,4",
        help="Fig.3 多次独立运行：逗号分隔的 replica 编号，与 --fig3-seed 组合成不同 RNG",
    )
    ap.add_argument("--fig4-num-ud", type=int, default=100, help="Fig.4 UD 数（须被 4 整除，默认 100）")
    ap.add_argument("--fig4-num-slots", type=int, default=200, help="Fig.4 时隙数（默认 200）")
    ap.add_argument(
        "--fig4-paper-ramp",
        action="store_true",
        help="对 Fig.4 中 ECF/RO 的 Q^E/Q^C 显示叠加大斜率项（旧版视觉辅助，非论文 Table I 纯仿真）",
    )
    ap.add_argument("--fig5-num-ud", type=int, default=200, help="Fig.5 UD 数（默认 200）")
    ap.add_argument("--fig5-num-slots", type=int, default=220, help="Fig.5 每带宽仿真时隙数（默认 220）")
    ap.add_argument(
        "--fig5-tail-slots",
        type=int,
        default=120,
        help="Fig.5 对最后若干时隙取平均（默认 120，勿大于 num_slots）",
    )
    ap.add_argument("--fig7-num-ud", type=int, default=100, help="Fig.7 UD 数（须被 4 整除，默认 100）")
    ap.add_argument("--fig7-num-slots", type=int, default=200, help="Fig.7 时隙数（默认 200）")
    ap.add_argument("--fig7-seed", type=int, default=42, help="Fig.7 初始状态与任务序列种子")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    if args.only in ("all", "fig3"):
        fig3_v_tradeoff(
            out,
            seed=int(args.fig3_seed),
            from_simulation=not bool(args.fig3_use_paper_points),
            fig3_num_ud=args.fig3_num_ud,
            fig3_num_slots=args.fig3_num_slots,
            replica_seeds=_parse_int_list(args.fig3_replica_seeds),
        )
    if args.only in ("all", "fig4"):
        fig4_time_slots(
            out,
            num_ud=int(args.fig4_num_ud),
            num_slots=int(args.fig4_num_slots),
            paper_ecf_ro_ramp=bool(args.fig4_paper_ramp),
        )
    if args.only in ("all", "fig5"):
        fig5_bandwidth(
            out,
            num_ud=int(args.fig5_num_ud),
            num_slots=int(args.fig5_num_slots),
            tail_slots=int(args.fig5_tail_slots),
        )
    if args.only in ("all", "fig6"):
        fig6_num_ud(out)
    if args.only in ("all", "fig7"):
        fig7_queue_backlog_bits(
            out,
            seed=int(args.fig7_seed),
            num_ud=int(args.fig7_num_ud),
            num_slots=int(args.fig7_num_slots),
        )
    print(f"已写入: {out.resolve()}")


if __name__ == "__main__":
    main()
