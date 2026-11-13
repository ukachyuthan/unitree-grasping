#!/usr/bin/env python3
"""
Figures and statistics for the paper, from the tracked results (data/paper_v2/curve.txt, the
per-case data/vr_round3/round3/eval_*.json) and the local VR recordings.

    python scripts/paper_figures.py <out_dir>      (from the repo root; writes <out_dir>/figures/*.pdf, stats.json)
"""

from __future__ import annotations

import glob
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUT = Path(sys.argv[1])
(OUT / "figures").mkdir(parents=True, exist_ok=True)
CASES = Path("data/vr_round3/round3")
TRAIN = Path("data/vr_round3_train/round3")
RUNS = Path("wrappers/isaaclab/data/grasp_logs/paper_v2")
ITERS = (25, 50, 75, 100, 125, 150)
SEEDS = (1, 2, 3)
RL, DEMO, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
})


# ── Data ──────────────────────────────────────────────────────────────────────
def load_curve():
    """{(condition, seed, iteration): dict of the four test scores in %}"""
    rows = {}
    pat = (r"(\w)_s(\d) (\d+) flat (\d+)/(\d+) tilted (\d+)/(\d+) hard: demo cases: ([\d.]+)% "
           r"of \d+ attempts heldout cases: ([\d.]+)%")
    for line in open("data/paper_v2/curve.txt"):
        m = re.match(pat, line)
        if m:
            c, s, i, fk, fn, tk, tn, dm, hd = m.groups()
            rows[(c, int(s), int(i))] = {
                "flat": 100 * int(fk) / int(fn), "tilted": 100 * int(tk) / int(tn),
                "demo": float(dm), "heldout": float(hd),
                "flat_k": (int(fk), int(fn)), "tilted_k": (int(tk), int(tn)),
            }
    return rows


STAMP = {f"{c}_s{s}": Path(glob.glob(str(RUNS / f"{c}_s{s}" / "grasp_pose_envs*"))[0]).name
         for c in "AD" for s in SEEDS}
HELDOUT = set(json.load(open(CASES / "split.json"))["heldout"])
SHAPE = {c["case_id"]: c["shape"] for c in json.load(open(CASES / "manifest.json"))["cases"]}
DEMOED = {Path(p).stem for p in glob.glob(str(TRAIN / "cases" / "*.json"))}


def hard(cond, seed, it):
    path = glob.glob(str(CASES / f"eval_{STAMP[f'{cond}_s{seed}']}_grasp_pose_{it}_*.json"))[0]
    return json.load(open(path))["cases"]


def per_case(iters, heldout):
    """{case: [A success rate, D success rate]} pooled over seeds, repeats and the given iterations."""
    out = {}
    for ci, cond in enumerate("AD"):
        for it in iters:
            for seed in SEEDS:
                for k, v in hard(cond, seed, it).items():
                    if (k in HELDOUT) == heldout:
                        out.setdefault(k, ([], []))[ci].extend(int(x) for x in v)
    return {k: (np.mean(a), np.mean(d)) for k, (a, d) in out.items()}


rng = random.Random(0)


def sign_flip(d, n=100000):
    obs = abs(sum(d)) / len(d)
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in d)) / len(d) >= obs - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def boot_ci(d, n=10000):
    v = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return v[int(0.025 * n)], v[int(0.975 * n)]


def compare(pc):
    d = [b - a for a, b in pc.values()]
    lo, hi = boot_ci(d)
    return {"n": len(d), "A": 100 * np.mean([a for a, _ in pc.values()]),
            "D": 100 * np.mean([b for _, b in pc.values()]), "diff": 100 * np.mean(d),
            "ci": [100 * lo, 100 * hi], "p": sign_flip(d),
            "better": sum(x > 0 for x in d), "worse": sum(x < 0 for x in d), "same": sum(x == 0 for x in d)}


def warm_start():
    """Per-case results and test scores of the warm start, before any RL."""
    cases = json.load(open(glob.glob(str(CASES / "eval_warmstart_grasp_pose_0_*.json"))[0]))["cases"]
    line = [x for x in open("data/paper_v2/results.txt") if x.startswith("warmstart") and "demo cases" in x][-1]
    m = re.search(r"flat (\d+)/(\d+) tilted (\d+)/(\d+) hard: demo cases: ([\d.]+)% .* heldout cases: ([\d.]+)%", line)
    fk, fn, tk, tn, dm, hd = m.groups()
    return cases, {"flat": 100 * int(fk) / int(fn), "tilted": 100 * int(tk) / int(tn), "demo": float(dm), "heldout": float(hd)}


curve = load_curve()
WARM_CASES, WARM = warm_start()
stats = {"iter150": {}, "pooled100_150": {}, "by_iter": {}, "warm": WARM, "versus_warm": {}}
for name, flag in (("demo", False), ("heldout", True)):
    stats["iter150"][name] = compare(per_case([150], flag))
    stats["pooled100_150"][name] = compare(per_case([100, 125, 150], flag))
for it in ITERS:
    for c in "AD":
        for key in ("flat", "tilted", "demo", "heldout"):
            stats["by_iter"][f"{c}_{it}_{key}"] = float(np.mean([curve[(c, s, it)][key] for s in SEEDS]))
for key in ("flat", "tilted"):
    for c in "AD":
        k = sum(curve[(c, s, 150)][f"{key}_k"][0] for s in SEEDS)
        n = sum(curve[(c, s, 150)][f"{key}_k"][1] for s in SEEDS)
        stats["iter150"][f"{c}_{key}"] = [k, n, 100 * k / n]


for name, flag in (("demo", False), ("heldout", True)):
    pc = per_case([150], flag)
    stats["versus_warm"][name] = {
        "RL_alone": compare({k: (float(np.mean(WARM_CASES[k])), v[0]) for k, v in pc.items()}),
        "RL_demos": compare({k: (float(np.mean(WARM_CASES[k])), v[1]) for k, v in pc.items()}),
    }

# Test objects: per-object difference at iteration 150, flat and tilted.
def test_episodes(cond, seed, kind):
    pat = re.compile(r"ep\s+\d+/200\s+shape=(\S+).*lift_ok=(True|False)")
    path = f"data/paper_v2/curve/{cond}_s{seed}_150_{kind}.log"
    return [(m[1], m[2] == "True") for m in map(pat.search, open(path, errors="ignore")) if m]


for kind in ("flat", "tilted"):
    by_obj = defaultdict(lambda: ([], []))
    for ci, cond in enumerate("AD"):
        for seed in SEEDS:
            for shape, ok in test_episodes(cond, seed, kind):
                by_obj[shape][ci].append(int(ok))
    both = {k: (float(np.mean(a)), float(np.mean(b))) for k, (a, b) in by_obj.items() if a and b}
    stats["iter150"][f"objects_{kind}"] = compare(both)

# Hidden cases by how close they are to a demonstrated case.
demo_shapes = {SHAPE[k] for k in DEMOED}
demo_fams = {s.split(".")[0] for s in demo_shapes}
pc = per_case([150], True)
groups = {
    "same_instance": [k for k in pc if SHAPE[k] in demo_shapes],
    "same_family_other_instance": [k for k in pc if SHAPE[k] not in demo_shapes and SHAPE[k].split(".")[0] in demo_fams],
    "unseen_family": [k for k in pc if SHAPE[k].split(".")[0] not in demo_fams],
}
stats["heldout_by_overlap"] = {
    g: {"n": len(v), "A": 100 * float(np.mean([pc[k][0] for k in v])), "D": 100 * float(np.mean([pc[k][1] for k in v]))}
    for g, v in groups.items()}
(OUT / "stats.json").write_text(json.dumps(stats, indent=1, default=float))


# ── Learning curves ───────────────────────────────────────────────────────────
PANELS = (("flat", "Test objects, flat"), ("tilted", "Test objects, tilted"),
          ("demo", "Hard cases, demonstrated"), ("heldout", "Hard cases, hidden"))
fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.75))
for ax, (key, title) in zip(axes, PANELS):
    ax.scatter([0], [WARM[key]], s=26, color=MUTED, marker="D", edgecolors="white", linewidths=0.5, zorder=4,
               label="Warm start, before RL")
    for cond, color, marker in (("A", RL, "o"), ("D", DEMO, "s")):
        for s in SEEDS:
            ax.scatter(ITERS, [curve[(cond, s, i)][key] for i in ITERS], s=5, color=color, alpha=0.3,
                       linewidths=0, marker=marker, zorder=2)
        ax.plot(ITERS, [np.mean([curve[(cond, s, i)][key] for s in SEEDS]) for i in ITERS], color=color,
                lw=1.5, marker=marker, ms=3, mec="white", mew=0.5, zorder=3,
                label={"A": "RL alone", "D": "RL + human demos"}[cond])
    ax.set_title(title, color=INK, pad=3)
    ax.set_xlabel("Training iteration", labelpad=1)
    ax.set_xticks([0, 50, 100, 150])
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, width=0.5, pad=1.5)
    ax.set_ylim((55, 92) if key in ("flat", "tilted") else (0, 65))
axes[0].set_ylabel("Lift success (%)", labelpad=1)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04), handlelength=1.8)
fig.tight_layout(pad=0.2, w_pad=0.7, rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "figures/learning_curves.pdf")
plt.close(fig)


# ── Per-case change ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.3), sharey=True)
for ax, (flag, title) in zip(axes, ((False, "Demonstrated hard cases"), (True, "Hidden hard cases"))):
    d = sorted(100 * (b - a) for a, b in per_case([150], flag).values())
    ax.bar(range(len(d)), d, width=0.82, color=[DEMO if x > 0 else RL for x in d], linewidth=0)
    ax.axhline(0, color=MUTED, lw=0.6)
    s = compare(per_case([150], flag))
    ax.set_title(f"{title}: {s['better']} better, {s['same']} unchanged, {s['worse']} worse", color=INK, pad=2, loc="left")
    ax.set_xlim(-0.8, len(d) - 0.2)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, width=0.5, pad=1.5)
    ax.set_ylabel("Change (pts)", labelpad=1)
axes[1].set_xlabel("Cases, sorted by change; RL + demos minus RL alone at iteration 150", labelpad=2)
fig.tight_layout(pad=0.2, h_pad=0.6)
fig.savefig(OUT / "figures/per_case.pdf")
plt.close(fig)


# ── Recorded grasps ───────────────────────────────────────────────────────────
def rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def to_world(p, pose):
    return np.asarray(p) @ rot(pose["quat_wxyz"]).T + np.asarray(pose["pos"])


pc_demo = per_case([150], False)
want = ["ycb_025_mug", "ycb_037_scissors", "bracket", "ycb_042_adjustable_wrench", "ycb_024_bowl", "torus",
        "ycb_030_fork", "star_prism", "ycb_052_extra_large_clamp", "ycb_028_skillet_lid", "c_shape", "ycb_009_gelatin_box"]
picked = []
for fam in want:
    options = [k for k in DEMOED if SHAPE[k].split(".")[0] == fam and k in pc_demo]
    if options:
        picked.append(max(options, key=lambda k: pc_demo[k][1] - pc_demo[k][0]))

fig = plt.figure(figsize=(7.16, 2.75))
for i, k in enumerate(picked):
    ax = fig.add_subplot(2, 6, i + 1, projection="3d")
    case = json.load(open(CASES / "cases" / f"{k}.json"))
    train = json.load(open(glob.glob(str(TRAIN / "demos" / k / "*.json"))[0]))
    rec = json.load(open(train["from_vr_demo"].replace("data/vr_round3/round3/", str(CASES) + "/")))
    g = rec["grasp"]
    mesh = trimesh.load(CASES / "objects" / f"{case['object']['shape']}.obj", force="mesh")
    pts = to_world(mesh.sample(3500) * case["object"]["mesh_scale"], g["object_pose"])
    ax.scatter(*pts.T, s=0.5, c="#a9a79f", linewidths=0, alpha=0.75, rasterized=True)
    human = to_world([g["c1_local"], g["c2_local"]], g["object_pose"])
    ax.scatter(*human.T, s=20, c=DEMO, marker="o", edgecolors="white", linewidths=0.5, depthshade=False, zorder=5,
               label="Human grasp points")
    fails = [a for a in case["failed_attempts"] if "grasp" in a and "contact_left_local" in a["grasp"]]
    f = fails[-1]
    rl = to_world([f["grasp"]["contact_left_local"], f["grasp"]["contact_right_local"]], f["object_pose"])
    ax.scatter(*rl.T, s=17, c=RL, marker="X", edgecolors="white", linewidths=0.4, depthshade=False, zorder=5,
               label="Failed RL grasp points")
    c, r = pts.mean(0), max(np.ptp(pts, axis=0).max() * 0.46, 0.045)
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=32, azim=-58 + 25 * (i % 4)); ax.set_axis_off()
    name = re.sub(r"^ycb_\d+(-\w)?_", "", SHAPE[k].split(".")[0]).replace("_", " ")
    ax.set_title(name, y=0.9, color=INK, fontsize=7.5)
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02), handletextpad=0.2)
fig.subplots_adjust(left=-0.01, right=1.01, top=1.0, bottom=0.05, wspace=-0.12, hspace=-0.02)
fig.savefig(OUT / "figures/demo_grasps.pdf", dpi=300)
plt.close(fig)


# ── How the hard cases were reached ───────────────────────────────────────────
ROWS = (("Exported by RL, by lifts in six retests", ((120, "6 of 6", "#d9d7d0"), (327, "2-5 of 6", "#b5b3ac"), (153, "0-1 of 6", RL))),
        ("Hard cases", ((82, "Demonstration half", DEMO), (71, "Hidden half", RL))),
        ("Demonstration half", ((58, "Robust demos", DEMO), (23, "Marked impossible", "#b5b3ac"), (1, "", "#d9d7d0"))))
fig, axes = plt.subplots(3, 1, figsize=(3.45, 1.45))
for ax, (title, parts) in zip(axes, ROWS):
    left = 0
    total = sum(p[0] for p in parts)
    for n, label, color in parts:
        ax.barh(0, n, left=left, color=color, height=0.8, edgecolor="white", linewidth=1.2)
        if label:
            dark = color in (RL, DEMO)
            ax.text(left + n / 2, 0, f"{label}: {n}" if n / total > 0.2 else str(n), ha="center", va="center",
                    color="white" if dark else INK, fontsize=6.6)
        left += n
    ax.set_xlim(0, total)
    ax.set_title(f"{title} (n = {total})", loc="left", color=INK, pad=1.5, fontsize=7.5)
    ax.axis("off")
fig.tight_layout(pad=0.1, h_pad=0.25)
fig.savefig(OUT / "figures/funnel.pdf")
plt.close(fig)


# ── Hard cases by object family ───────────────────────────────────────────────
fam = defaultdict(lambda: [[], [], []])
for flag in (False, True):
    for k, (a, d) in per_case([150], flag).items():
        f = fam[SHAPE[k].split(".")[0]]
        f[0].append(a); f[1].append(d); f[2].append(float(np.mean(WARM_CASES[k])))
rows = sorted(((re.sub(r"^ycb_\d+(-\w)?_", "", k).replace("_", " "), len(v[0]), *(100 * float(np.mean(x)) for x in v))
               for k, v in fam.items() if len(v[0]) >= 4), key=lambda r: r[3] - r[2])
fig, ax = plt.subplots(figsize=(3.45, 0.2 * len(rows) + 0.55))
for y, (name, n, a, d, w) in enumerate(rows):
    ax.plot([a, d], [y, y], color=GRID, lw=2, zorder=1, solid_capstyle="round")
ys = range(len(rows))
ax.scatter([r[4] for r in rows], ys, s=16, color=MUTED, marker="D", edgecolors="white", linewidths=0.4, zorder=2, label="Warm start")
ax.scatter([r[2] for r in rows], ys, s=22, color=RL, edgecolors="white", linewidths=0.4, zorder=3, label="RL alone")
ax.scatter([r[3] for r in rows], ys, s=22, color=DEMO, marker="s", edgecolors="white", linewidths=0.4, zorder=3, label="RL + demos")
ax.set_yticks(list(ys)); ax.set_yticklabels([f"{r[0]} ({r[1]})" for r in rows], color=INK)
ax.set_xlabel("Lift success on hard cases at iteration 150 (%)", labelpad=1)
ax.set_xlim(-2, 102); ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.tick_params(length=2, width=0.5, pad=1.5); ax.spines["left"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.14), ncol=3, frameon=False, handletextpad=0.1, columnspacing=0.8)
fig.tight_layout(pad=0.15)
fig.savefig(OUT / "figures/families.pdf")
plt.close(fig)
stats["families"] = [dict(zip(("name", "n", "A", "D", "warm"), r)) for r in rows]
(OUT / "stats.json").write_text(json.dumps(stats, indent=1, default=float))
print(rows)
