"""Plot the flow-matching solver sweep: FID vs true NFE and FID vs runtime.

Reads the two CSVs produced by the sweep scripts and renders two figures so
Euler and RK4 can be compared at matched compute:

    1. FID vs true NFE            (x = function evaluations)
    2. FID vs runtime             (x = median seconds / sample)

Each solver is one series (solid line + markers = measured). With ``--fit`` an
exponential-saturation curve ``FID(x) = c + a * exp(-b * x)`` is fitted per
solver and drawn DASHED and clearly labelled as a fit, with its estimated floor
``c`` in the legend -- so measured points and extrapolated asymptotes are never
confused.

Inputs (defaults match scripts/solver_sweep_fm.sh + time_solver_sweep.sh):
    --fid-csv       samples/solver_sweep/fid_summary.csv    (solver,steps,nfe,num_fake,fid)
    --runtime-csv   samples/solver_sweep/runtime_median.csv (median_seconds_per_sample)
                    falls back to runtime_summary.csv        (seconds_per_sample)

Usage:
    python evaluations/plot_solver_sweep.py --fit
    python evaluations/plot_solver_sweep.py \
        --fid-csv samples/solver_sweep/fid_summary.csv \
        --runtime-csv samples/solver_sweep/runtime_median.csv \
        --solvers euler rk4 --out-dir samples/solver_sweep/plots --fit
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # headless / login-node safe
import matplotlib.pyplot as plt
import numpy as np

# --- Design-system parameters (from the dataviz skill's validated palette) ---
# Categorical slots 1 (blue) and 8 (orange): the most CVD-safe pair in the ramp.
# Each solver also gets a distinct marker so identity never rests on colour alone.
SOLVER_STYLE = {
    "euler":   {"color": "#2a78d6", "marker": "o", "label": "Euler"},
    "rk4":     {"color": "#eb6834", "marker": "s", "label": "RK4"},
    "heun":    {"color": "#1baf7a", "marker": "^", "label": "Heun"},
    "pc":      {"color": "#4a3aa7", "marker": "D", "label": "PC"},
    "pc_heun": {"color": "#e34948", "marker": "v", "label": "PC-Heun"},
}
DEFAULT_STYLE = {"color": "#52514e", "marker": "x"}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#d9d8d4"
SURFACE = "#fcfcfb"


def _style(solver):
    s = SOLVER_STYLE.get(solver, dict(DEFAULT_STYLE))
    s.setdefault("label", solver)
    return s


def read_fid(path):
    """Return {solver: {steps(int): fid(float)}} from the FID summary CSV."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            fid = r.get("fid", "").strip()
            if not fid:
                continue
            try:
                out.setdefault(r["solver"], {})[int(r["steps"])] = float(fid)
            except (ValueError, KeyError):
                continue
    return out


def read_runtime(path):
    """Return {solver: {steps(int): seconds_per_sample(float)}}.

    Accepts either the median CSV (``median_seconds_per_sample``) or the
    single-run summary CSV (``seconds_per_sample``).
    """
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        col = ("median_seconds_per_sample"
               if "median_seconds_per_sample" in (reader.fieldnames or [])
               else "seconds_per_sample")
        for r in reader:
            val = r.get(col, "").strip()
            if not val:
                continue
            try:
                out.setdefault(r["solver"], {})[int(r["steps"])] = float(val)
            except (ValueError, KeyError):
                continue
    return out


def nfe_of(solver, steps):
    return {"euler": 1, "heun": 2, "rk4": 4, "pc": 1, "pc_heun": 2}.get(solver, 1) * steps


def fit_saturation(x, y):
    """Fit FID(x) = c + a*exp(-b*x); return (c, a, b) or None.

    Needs >=3 points and scipy. ``c`` is the estimated FID floor.
    """
    if len(x) < 3:
        return None
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        print("[plot] scipy not available -- skipping saturation fit.")
        return None

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    model = lambda t, c, a, b: c + a * np.exp(-b * t)
    # c near the smallest observed FID, a spanning the range, b a gentle decay.
    p0 = [float(y.min()), float(max(y.max() - y.min(), 1e-6)), 1.0 / max(x.mean(), 1.0)]
    bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf])
    try:
        popt, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError) as exc:
        print(f"[plot] saturation fit failed: {exc}")
        return None
    return tuple(float(v) for v in popt)


def style_axes(ax, xlabel, ylabel, title):
    ax.set_title(title, color=INK_PRIMARY, fontsize=13, pad=10, loc="left")
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=11)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=11)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY, labelsize=10)


def make_plot(series, xlabel, title, out_path, do_fit):
    """series: list of (solver, xs, ys) already sorted by x."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    for solver, xs, ys in series:
        if not xs:
            continue
        st = _style(solver)
        ax.plot(xs, ys, color=st["color"], marker=st["marker"], markersize=6,
                linewidth=2.0, label=st["label"], zorder=3)

        if do_fit:
            fit = fit_saturation(xs, ys)
            if fit is not None:
                c, a, b = fit
                gx = np.linspace(min(xs), max(xs), 200)
                gy = c + a * np.exp(-b * gx)
                ax.plot(gx, gy, color=st["color"], linewidth=1.6, linestyle="--",
                        alpha=0.9, zorder=2,
                        label=f"{st['label']} fit (floor ≈ {c:.3f})")

    style_axes(ax, xlabel, "3D FID", title)
    ax.legend(frameon=False, fontsize=10, labelcolor=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description="Plot the FM solver sweep (FID vs NFE / runtime).")
    p.add_argument("--fid-csv", default="samples/solver_sweep/fid_summary.csv")
    p.add_argument("--runtime-csv", default="samples/solver_sweep/runtime_median.csv")
    p.add_argument("--solvers", nargs="+", default=["euler", "rk4"])
    p.add_argument("--out-dir", default="samples/solver_sweep/plots")
    p.add_argument("--fit", action="store_true",
                   help="Overlay dashed exponential-saturation fits (needs scipy, >=3 points).")
    p.add_argument("--max-nfe", type=int, default=None,
                   help="Drop points above this true NFE (both plots).")
    args = p.parse_args()

    if not os.path.exists(args.fid_csv):
        raise SystemExit(f"FID CSV not found: {args.fid_csv}")
    fid = read_fid(args.fid_csv)

    runtime = {}
    rt_path = args.runtime_csv
    if not os.path.exists(rt_path):
        fallback = os.path.join(os.path.dirname(rt_path), "runtime_summary.csv")
        if os.path.exists(fallback):
            print(f"[plot] {rt_path} missing; using {fallback} (single-run timings).")
            rt_path = fallback
    if os.path.exists(rt_path):
        runtime = read_runtime(rt_path)
    else:
        print("[plot] no runtime CSV found -- FID-vs-runtime plot will be skipped.")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- FID vs true NFE -----------------------------------------------------
    nfe_series = []
    for solver in args.solvers:
        pts = []
        for steps, val in fid.get(solver, {}).items():
            nfe = nfe_of(solver, steps)
            if args.max_nfe is not None and nfe > args.max_nfe:
                continue
            pts.append((nfe, val))
        pts.sort()
        nfe_series.append((solver, [x for x, _ in pts], [y for _, y in pts]))
    make_plot(nfe_series, "True NFE (function evaluations)",
              "FID vs true NFE", os.path.join(args.out_dir, "fid_vs_nfe.png"), args.fit)

    # --- FID vs runtime ------------------------------------------------------
    if runtime:
        rt_series = []
        for solver in args.solvers:
            pts = []
            for steps, val in fid.get(solver, {}).items():
                if args.max_nfe is not None and nfe_of(solver, steps) > args.max_nfe:
                    continue
                sps = runtime.get(solver, {}).get(steps)
                if sps is None:
                    continue
                pts.append((sps, val))
            pts.sort()
            rt_series.append((solver, [x for x, _ in pts], [y for _, y in pts]))
        make_plot(rt_series, "Median runtime (seconds / sample)",
                  "FID vs runtime", os.path.join(args.out_dir, "fid_vs_runtime.png"), args.fit)

    print("[plot] done.")


if __name__ == "__main__":
    main()
