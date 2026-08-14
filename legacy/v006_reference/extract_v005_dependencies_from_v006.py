import json, os, sys
import numpy as np

V006 = "/mnt/data/Lily_v0.0.6_pitch45_roll45_search"
V004 = "/mnt/data/Lily_v0.0.4_receding_horizon_draft"
OUT = "/mnt/data/Lily_v0.0.5_pitch45_roll45_search"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, V004 + "/src")
from lily_contact_planner import LilyKinematics

D = np.load(V006 + "/trajectory.npz")
t, R, q, C = D["body_pos"], D["body_R"], D["q"], D["contact"].astype(bool)

td = None
for i in range(1, len(C) - 1):
    if (
        np.where(C[i - 1])[0].tolist() == [0, 5, 6]
        and np.where(C[i])[0].tolist() == [0, 4, 5, 6]
        and np.where(C[i + 1])[0].tolist() == [0, 4, 5]
    ):
        td = i
        break
if td is None:
    raise RuntimeError("v0.0.6 static event not found")

start = td - 1
while (
    start > 0
    and np.linalg.norm(t[start - 1] - t[td]) < 1e-12
    and np.linalg.norm(R[start - 1] - R[td]) < 1e-12
):
    start -= 1
prefix_end = start

if (td - 1 - prefix_end) % 20 != 0:
    raise RuntimeError("unexpected PRM interpolation layout")
node_frames = list(range(prefix_end, td, 20))
if node_frames[-1] != td - 1:
    node_frames.append(td - 1)
path = q[node_frames, 4].copy()

kin = LilyKinematics(a=0.15, L2=0.30, L3=0.30)
goal = kin.foot_world(t[td], R[td], 4, q[td, 4]).copy()
goal[2] = 0.0

np.savez_compressed(
    OUT + "/trajectory.npz",
    body_pos=t[: prefix_end + 1],
    body_R=R[: prefix_end + 1],
    q=q[: prefix_end + 1],
    contact=C[: prefix_end + 1],
)
np.savez_compressed(OUT + "/prm_leg4_static_path.npz", q=path, goal_xy=goal[:2])

rep = json.load(open(V006 + "/report.json"))
cycles = []
for c in rep["cycles"]:
    if c.get("version") == "v0.0.6":
        break
    cycles.append(c)
events = [
    e for e in rep["contact_events"]
    if e.get("version") != "v0.0.6-static-event"
]
oldrep = {
    "status": "stopped",
    "cycles": cycles,
    "contact_events": events,
    "final_support": np.where(C[prefix_end])[0].tolist(),
    "target_definition": rep.get("target_definition"),
    "recovered_from_v006": True,
}
json.dump(oldrep, open(OUT + "/report.json", "w"), indent=2)

print(json.dumps({
    "static_touchdown_frame": int(td),
    "prefix_end_frame": int(prefix_end),
    "prm_node_frames": node_frames,
    "prm_nodes": len(path),
    "support_at_prefix_end": np.where(C[prefix_end])[0].tolist(),
    "support_after_static_event": np.where(C[td + 1])[0].tolist(),
}, indent=2))
