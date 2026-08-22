#!/usr/bin/env python3
"""Probe cooperative contact transitions from a saved PitchForward checkpoint.

This script does not continue DFS. It loads the checkpoint BEST state and
solves neighboring (A, R) transition edges only. Progress is printed for every
horizon and candidate so a long SLSQP solve is distinguishable from a hang.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lily_contact_planner.experimental_cooperative_transition import (
    CooperativeTransitionSettings,
    _enumerate_candidates,
    _solve_record,
)
from lily_contact_planner.experimental_cooperative_transition_numerics import (
    enable_consistent_cooperative_numerics,
    get_cooperative_nlp_diagnostics,
    reset_cooperative_nlp_diagnostics,
)
from lily_contact_planner.kinematics import LilyKinematics
from lily_contact_planner.tasks import PitchForwardTask
from lily_contact_planner.unified_planner import UnifiedContactPlanner
from lily_contact_planner.v004_success_seed import (
    _CandidateSolveTimeout,
    _candidate_wall_clock_timeout,
)


def _load_best_state(data):
    summary = data.get("best_summary", {})
    angle = data.get("best_angle_deg", summary.get("best_angle_deg"))
    support = data.get("best_support", summary.get("best_support"))
    q = data.get("best_q", summary.get("best_q_rad"))
    anchors = data.get("best_anchors", summary.get("best_anchors"))
    if angle is None or support is None or q is None or anchors is None:
        raise ValueError("checkpoint does not contain best angle/support/q/anchors")
    return (
        float(angle),
        np.asarray(q, dtype=float),
        tuple(int(x) for x in support),
        {int(k): np.asarray(v, dtype=float) for k, v in anchors.items()},
    )


def _summary(result):
    cand = result["candidate"]
    checker = result["checker"]
    return {
        "horizon_deg": float(result["horizon_deg"]),
        "exec_node": int(result["exec_node"]),
        "exec_fraction": float(result["exec_fraction"]),
        "angle_after_deg": float(result["angle_after_deg"]),
        "add": [int(x) for x in cand.touchdown_legs],
        "remove": [int(x) for x in cand.liftoff_legs],
        "support_after": [int(x) for x in result["support_after"]],
        "predicted_gain_deg": float(result["predicted_gain_deg"]),
        "objective": float(result["objective"]),
        "checker": {
            k: (float(v) if isinstance(v, (int, float, np.number)) else v)
            for k, v in checker.items()
        },
    }


def _probe(planner, angle, q, support, anchors, settings, all_feasible):
    phase_end = float(planner._v004_phase_end(angle))
    maxh = int(np.floor(min(
        float(settings.horizon_max_deg), phase_end - float(angle)
    ) + 1e-9))
    accepted = []
    totals = {
        "attempted": 0,
        "timeout": 0,
        "nlp": 0,
        "checker": 0,
        "predicted_gain": 0,
    }
    probe_start = time.monotonic()

    for h in range(maxh, 0, -1):
        records, counts = _enumerate_candidates(
            planner, angle, q, support, anchors, h, settings
        )
        print(
            "COOP_PROBE_HORIZON",
            json.dumps({
                "horizon_deg": h,
                **counts,
            }),
            flush=True,
        )

        for candidate_index, record in enumerate(records):
            cand = record["candidate"]
            attempt = {
                "horizon_deg": h,
                "candidate": candidate_index + 1,
                "of": len(records),
                "add": [int(x) for x in cand.touchdown_legs],
                "remove": [int(x) for x in cand.liftoff_legs],
                "support_after": [int(x) for x in record["support_after"]],
                "exec_angle_deg": float(record["exec_angle_deg"]),
            }
            print("COOP_PROBE_ATTEMPT", json.dumps(attempt), flush=True)
            totals["attempted"] += 1
            start = time.monotonic()
            reset_cooperative_nlp_diagnostics()

            try:
                with _candidate_wall_clock_timeout(
                    float(settings.candidate_timeout_s)
                ):
                    result, reject = _solve_record(
                        planner, angle, q, support, anchors,
                        h, record, settings
                    )
            except _CandidateSolveTimeout:
                totals["timeout"] += 1
                print(
                    "COOP_PROBE_REJECT",
                    json.dumps({
                        **attempt,
                        "reason": "timeout",
                        "elapsed_s": time.monotonic() - start,
                    }),
                    flush=True,
                )
                continue

            elapsed = time.monotonic() - start
            solver_diag = get_cooperative_nlp_diagnostics()
            if result is None:
                reason = str(reject or "unknown")
                totals[reason] = totals.get(reason, 0) + 1
                payload = {
                    **attempt,
                    "reason": reason,
                    "elapsed_s": elapsed,
                }
                if solver_diag is not None:
                    payload["solver"] = solver_diag
                print(
                    "COOP_PROBE_REJECT",
                    json.dumps(payload),
                    flush=True,
                )
                continue

            accepted.append(result)
            payload = {
                **attempt,
                "elapsed_s": elapsed,
                "angle_after_deg": float(result["angle_after_deg"]),
                "predicted_gain_deg": float(result["predicted_gain_deg"]),
                "objective": float(result["objective"]),
            }
            if solver_diag is not None:
                payload["solver"] = solver_diag
            print(
                "COOP_PROBE_ACCEPT",
                json.dumps(payload),
                flush=True,
            )
            if not all_feasible:
                return accepted, totals, time.monotonic() - probe_start

    return accepted, totals, time.monotonic() - probe_start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="pitch60_contactgraph_checkpoint.json")
    parser.add_argument("--pitch-deg", type=float, default=60.0)
    parser.add_argument("--forward-m-per-deg", type=float, default=1.0 / 300.0)
    parser.add_argument("--body-height-m", type=float, default=0.35)
    parser.add_argument("--max-support-count", type=int, default=5)
    parser.add_argument("--max-add", type=int, default=1)
    parser.add_argument("--max-release", type=int, default=2)
    parser.add_argument("--max-total-changes", type=int, default=3)
    parser.add_argument("--horizon-max-deg", type=int, default=5)
    parser.add_argument("--settling-nodes", type=int, default=1)
    parser.add_argument("--liftoff-clearance-m", type=float, default=0.02)
    parser.add_argument("--candidate-timeout-s", type=float, default=60.0)
    parser.add_argument("--max-candidates-per-horizon", type=int, default=48)
    parser.add_argument("--all-feasible", action="store_true")
    args = parser.parse_args()

    numerics_info = enable_consistent_cooperative_numerics()

    with Path(args.checkpoint).open("r", encoding="utf-8") as f:
        data = json.load(f)
    angle, q, support, anchors = _load_best_state(data)

    kin = LilyKinematics(
        a=0.15,
        L2=0.30,
        L3=0.30,
        delta_top=0.0,
        delta_bottom=0.0,
        eps_top=+1.0,
        eps_bottom=-1.0,
    )
    task = PitchForwardTask(
        body_height_m=float(args.body_height_m),
        forward_m_per_deg=float(args.forward_m_per_deg),
        pitch_deg=float(args.pitch_deg),
    )
    planner = UnifiedContactPlanner(
        kin,
        task,
        max_roll_deg=task.total_progress_deg,
        verbose=True,
    )

    settings = CooperativeTransitionSettings(
        max_support_count=int(args.max_support_count),
        max_add_per_transition=int(args.max_add),
        max_release_per_transition=int(args.max_release),
        max_total_contact_changes=int(args.max_total_changes),
        horizon_max_deg=int(args.horizon_max_deg),
        touchdown_seed_rank=1,
        settling_nodes=int(args.settling_nodes),
        liftoff_clearance_m=float(args.liftoff_clearance_m),
        candidate_timeout_s=float(args.candidate_timeout_s),
        max_candidates_per_horizon=int(args.max_candidates_per_horizon),
    )

    print(
        "COOP_PROBE_STATE",
        json.dumps({
            "angle_deg": angle,
            "support": list(support),
            "settings": settings.__dict__,
            "numerics": numerics_info,
        }),
        flush=True,
    )

    results, totals, elapsed = _probe(
        planner,
        angle,
        q,
        support,
        anchors,
        settings,
        all_feasible=bool(args.all_feasible),
    )
    print(
        "COOP_PROBE_RESULT",
        json.dumps({
            "feasible_count": len(results),
            "attempt_stats": totals,
            "elapsed_s": elapsed,
            "edges": [_summary(x) for x in results],
        }, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
