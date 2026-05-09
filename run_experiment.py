"""
运行仿真：V 扫描、多算法对比、绘图（需 matplotlib）。
用法：在项目根目录执行
  python -m ojcta_mec.run_experiment
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace

import numpy as np

from .baselines import run_timeslot_fo, run_timeslot_gjtora, run_timeslot_lc, run_timeslot_ro
from .ojcta import run_timeslot_ojcta
from .params import SimulationParams
from .system_state import SystemState


def simulate_policy(
    policy: str,
    params: SimulationParams,
    seed: int,
) -> tuple[list[float], list[float], list[float]]:
    rng = np.random.default_rng(seed)
    state = SystemState(params=params, rng=rng)
    energies: list[float] = []
    d_edge: list[float] = []
    d_cloud: list[float] = []

    for t in range(1, params.num_slots + 1):
        task_bits = params.sample_task_bits(rng)
        if policy == "ojcta":
            e, de, dc, _dl = run_timeslot_ojcta(state, t, task_bits, ncc=False, ecf=False)
        elif policy == "ojcta_ncc":
            e, de, dc, _dl = run_timeslot_ojcta(state, t, task_bits, ncc=True, ecf=False)
        elif policy == "ecf":
            e, de, dc, _dl = run_timeslot_ojcta(state, t, task_bits, ncc=False, ecf=True)
        elif policy == "lc":
            e, de, dc, _dl = run_timeslot_lc(state, t, task_bits)
        elif policy == "ro":
            e, de, dc, _dl = run_timeslot_ro(state, t, task_bits)
        elif policy == "fo":
            e, de, dc, _dl = run_timeslot_fo(state, t, task_bits)
        elif policy == "gjtora":
            e, de, dc, _dl = run_timeslot_gjtora(state, t, task_bits)
        else:
            raise ValueError(policy)
        energies.append(e)
        d_edge.append(de)
        d_cloud.append(dc)
    return energies, d_edge, d_cloud


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=None, help="时隙数（默认用 SimulationParams.num_slots）")
    ap.add_argument(
        "--lambda-arrival",
        type=float,
        default=None,
        help="任务到达率 λ∈[0.6,1.0]（Bernoulli）；默认 1.0 即每时隙必有任务",
    )
    ap.add_argument("--full-v", action="store_true", help="V 从 0 到 50 步长 5 全扫描（耗时更长）")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if args.full_v:
        V_list = list(range(0, 51, 5))
        if 35 not in V_list:
            V_list.append(35)
        V_list = sorted(set(V_list))
    else:
        V_list = [0, 10, 20, 30, 35, 40, 50]

    results: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for V in V_list:
        p = SimulationParams(V=float(V))
        if args.slots is not None:
            p = replace(p, num_slots=args.slots)
        if args.lambda_arrival is not None:
            p = replace(p, task_arrival_lambda=float(args.lambda_arrival))
        for pol in ("ojcta", "ecf", "gjtora", "ro", "fo", "lc"):
            e, de, dc = simulate_policy(pol, p, seed=p.rng_seed + int(V))
            results[pol]["energy"].append(float(np.mean(e)))
            results[pol]["edge_delay"].append(float(np.mean(de)))
            results[pol]["cloud_delay"].append(float(np.mean(dc)))

    print("V\tOJCTA_E\tECF_E\tGJTORA_E\tRO_E\tFO_E\tLC_E")
    for i, V in enumerate(V_list):
        print(
            f"{V}\t{results['ojcta']['energy'][i]:.4g}\t{results['ecf']['energy'][i]:.4g}\t"
            f"{results['gjtora']['energy'][i]:.4g}\t{results['ro']['energy'][i]:.4g}\t"
            f"{results['fo']['energy'][i]:.4g}\t{results['lc']['energy'][i]:.4g}"
        )

    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("未安装 matplotlib，跳过绘图。pip install matplotlib")
            return

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for pol, sty in [
            ("ojcta", "b-o"),
            ("ecf", "g--s"),
            ("gjtora", "m-^"),
            ("ro", "c-x"),
            ("fo", "orange"),
            ("lc", "k:"),
        ]:
            ax[0].plot(V_list, results[pol]["energy"], sty, label=pol.upper(), markersize=4)
            ax[1].plot(V_list, results[pol]["edge_delay"], sty, label=pol.upper(), markersize=4)
        ax[0].set_xlabel("V")
        ax[0].set_ylabel("Average UD energy (J/slot approx)")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        ax[1].set_xlabel("V")
        ax[1].set_ylabel("Avg edge queuing delay (s)")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        fig.tight_layout()
        out = "ojcta_V_tradeoff.png"
        fig.savefig(out, dpi=150)
        print(f"已保存图像: {out}")


if __name__ == "__main__":
    main()
