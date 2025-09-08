# dbevm_system.py
# Dynamic Brushwork Embodied Visualization Model (DBEVM)
# Dependencies:
#   pip install numpy pandas scipy scikit-learn pillow
#
# What this script does:
#   • Simulates brush sensors (pressure, planar velocity), segments strokes
#   • Computes biomechanical features (pressure modulation, curvature, terminal control)
#   • Generates AR-style feedback categories (color, haptic, audio) by deviation thresholds
#   • Maps strokes to symbolic tags (orchid_leaf, bamboo_joint, rock_ridge, generic)
#   • Computes composition metrics vs. a template (centroid displacement, alignment error, density variance)
#   • Scores GSPR (pressure, curvature, terminal, total) and engagement (time on task, distractions)
#   • Runs a cohort experiment (AR vs. Control), and simulates retention
#   • Saves JSON session logs and a CSV cohort summary to ./dbevm_logs

import os, json, time, math, random, uuid, pathlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from PIL import Image, ImageFilter

# ------------------------------------------------------------
# 0) Configuration and small utilities
# ------------------------------------------------------------
@dataclass
class DBEVMConfig:
    seed: int = 42
    fs_hz: int = 120                      # sampling frequency (Hz)
    sg_window: int = 9                    # Savitzky–Golay smoothing window (odd)
    sg_poly: int = 3                      # Savitzky–Golay polynomial order
    min_stroke_len: int = 12              # minimum samples to consider a stroke valid
    press_thresh_on: float = 0.12         # pressure threshold to start a stroke
    press_thresh_off: float = 0.08        # pressure threshold to end a stroke
    ar_latency_ms: int = 40               # simulated rendering latency
    haptic_min_dev: float = 0.08          # thresholds for feedback types
    audio_min_dev: float = 0.12
    color_min_dev: float = 0.05
    gspr_weights: Tuple[float, float, float] = (0.34, 0.33, 0.33)  # P, Curv, Term weights
    save_dir: str = "dbevm_logs"
    grid_size: int = 256                  # raster size for composition analysis
    retention_weeks_gap: int = 4

def set_seed(s:int):
    random.seed(s)
    np.random.seed(s)

def ensure_dir(path:str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def smooth_sg(x:np.ndarray, window:int, poly:int) -> np.ndarray:
    """Savitzky–Golay smoothing with safe window handling."""
    if window % 2 == 0:
        window += 1
    if len(x) < window:
        return x.astype(float)
    return savgol_filter(x, window_length=window, polyorder=poly).astype(float)

# ------------------------------------------------------------
# 1) Sensor packets and stream generator
# ------------------------------------------------------------
@dataclass
class SensorPacket:
    t: float
    press: float
    vx: float
    vy: float

class SensorHub:
    """
    Stub for hardware data acquisition. Replace simulate_* with real device reads.
    Generates realistic stroke-like signals for pressure and planar velocities.
    """
    def __init__(self, cfg: DBEVMConfig):
        self.cfg = cfg
        self.t = 0.0
        self.dt = 1.0 / cfg.fs_hz

    def _simulate_stroke(self, duration_s: float, pattern: str) -> List[SensorPacket]:
        n = max(int(duration_s * self.cfg.fs_hz), self.cfg.min_stroke_len)
        t = np.linspace(0.0, duration_s, n)

        # Pressure signal (mild sinusoidal with noise)
        press = 0.12 + 0.05*np.sin(2*np.pi*t/duration_s) + 0.015*np.random.randn(n)
        press = np.clip(press, 0.0, 1.0)

        # Velocity patterns (curved vs. straight)
        if pattern == "curve":
            vx = 0.15*np.cos(2*np.pi*t/duration_s) + 0.01*np.random.randn(n)
            vy = 0.10*np.sin(2*np.pi*t/duration_s) + 0.01*np.random.randn(n)
        else:
            vx = 0.18*np.ones(n) + 0.01*np.random.randn(n)
            vy = 0.00*np.ones(n) + 0.01*np.random.randn(n)

        vx = smooth_sg(vx, self.cfg.sg_window, self.cfg.sg_poly)
        vy = smooth_sg(vy, self.cfg.sg_window, self.cfg.sg_poly)

        packets = []
        for i in range(n):
            self.t += self.dt
            packets.append(SensorPacket(self.t, float(press[i]), float(vx[i]), float(vy[i])))
        return packets

    def stream_session(self, n_strokes:int = 12) -> List[SensorPacket]:
        packets: List[SensorPacket] = []
        for k in range(n_strokes):
            duration = np.random.uniform(0.9, 1.5)
            pattern = "curve" if (k % 2 == 0) else "line"
            packets += self._simulate_stroke(duration, pattern)

            # idle between strokes
            idle_n = max(int(0.2 * self.cfg.fs_hz), 1)
            for _ in range(idle_n):
                self.t += self.dt
                packets.append(SensorPacket(self.t, 0.02 + 0.01*np.random.randn(), 0.0, 0.0))
        return packets

# ------------------------------------------------------------
# 2) Stroke modeling
# ------------------------------------------------------------
@dataclass
class Stroke:
    tid: str
    t: np.ndarray
    press: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    x: np.ndarray
    y: np.ndarray

class BrushModel:
    """Segments packets into strokes and computes stroke-level features."""
    def __init__(self, cfg: DBEVMConfig):
        self.cfg = cfg

    def segment(self, packets: List[SensorPacket]) -> List[Stroke]:
        press = np.array([p.press for p in packets])
        t = np.array([p.t for p in packets])
        vx = np.array([p.vx for p in packets])
        vy = np.array([p.vy for p in packets])

        on = press > self.cfg.press_thresh_on
        starts = np.where(np.diff(on.astype(int)) == 1)[0] + 1
        ends   = np.where(np.diff(on.astype(int)) == -1)[0] + 1
        if on[0]:  starts = np.r_[0, starts]
        if on[-1]: ends   = np.r_[ends, len(on)-1]

        strokes: List[Stroke] = []
        for i, j in zip(starts, ends):
            if j - i < self.cfg.min_stroke_len:
                continue
            tt = t[i:j]
            px = press[i:j]
            vvx = vx[i:j]
            vvy = vy[i:j]
            # integrate velocity to position
            x = np.cumsum(vvx) / self.cfg.fs_hz
            y = np.cumsum(vvy) / self.cfg.fs_hz
            strokes.append(Stroke(
                tid=str(uuid.uuid4()),
                t=tt, press=px, vx=vvx, vy=vvy, x=x, y=y
            ))
        return strokes

    def curvature(self, s: Stroke) -> float:
        dx = np.gradient(s.x); dy = np.gradient(s.y)
        ddx = np.gradient(dx);  ddy = np.gradient(dy)
        num = np.abs(dx*ddy - dy*ddx)
        den = (dx**2 + dy**2)**1.5 + 1e-8
        return float(np.mean(num/den))

    def terminal_control(self, s: Stroke) -> float:
        w = max(int(0.08 * self.cfg.fs_hz), 4)
        s_var = np.var(s.x[:w]) + np.var(s.y[:w])
        e_var = np.var(s.x[-w:]) + np.var(s.y[-w:])
        return float(1.0 / (1e-6 + s_var + e_var))

    def pressure_modulation(self, s: Stroke) -> float:
        p = smooth_sg(s.press, self.cfg.sg_window, self.cfg.sg_poly)
        dp = np.diff(p)
        return float(1.0 / (1e-6 + np.std(dp)))

# ------------------------------------------------------------
# 3) AR overlay (deviation and cue selection)
# ------------------------------------------------------------
class AROverlay:
    def __init__(self, cfg: DBEVMConfig, expert_template: Dict[str, float]):
        self.cfg = cfg
        self.template = expert_template

    def deviation(self, metrics: Dict[str, float]) -> Dict[str, float]:
        dev = {}
        for k, v in metrics.items():
            ref = self.template.get(k, v)
            dev[k] = float(abs(v - ref) / (abs(ref) + 1e-6))
        return dev

    def feedback(self, dev: Dict[str, float]) -> Dict[str, str]:
        cues = {}
        for k, d in dev.items():
            if d >= self.cfg.audio_min_dev:
                cues[k] = "audio"
            elif d >= self.cfg.haptic_min_dev:
                cues[k] = "haptic"
            elif d >= self.cfg.color_min_dev:
                cues[k] = "color"
            else:
                cues[k] = "none"
        return cues

# ------------------------------------------------------------
# 4) Symbolic mapper (heuristic)
# ------------------------------------------------------------
class SymbolicMapper:
    def __init__(self):
        self.rules = [
            ("orchid_leaf", lambda m: m["curvature"] < 0.6 and m["press_mod"] > 3.0),
            ("bamboo_joint", lambda m: m["terminal"] > 30 and m["press_mod"] > 2.5),
            ("rock_ridge",   lambda m: m["curvature"] >= 0.6 and m["press_mod"] <= 3.0),
        ]

    def label(self, metrics: Dict[str, float]) -> str:
        for tag, rule in self.rules:
            if rule(metrics):
                return tag
        return "generic_stroke"

# ------------------------------------------------------------
# 5) Composition analysis
# ------------------------------------------------------------
class CompositionAnalyzer:
    def __init__(self, cfg: DBEVMConfig):
        self.cfg = cfg

    def _to_grid(self, xy: np.ndarray) -> np.ndarray:
        """Rasterize points to a smoothed density grid."""
        if xy is None or len(xy) == 0:
            return np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32)

        grid = np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32)
        x = xy[:, 0]; y = xy[:, 1]
        # normalize to [0, 1]
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        y = (y - y.min()) / (y.max() - y.min() + 1e-8)
        xs = np.clip((x * (self.cfg.grid_size - 1)).astype(int), 0, self.cfg.grid_size - 1)
        ys = np.clip((y * (self.cfg.grid_size - 1)).astype(int), 0, self.cfg.grid_size - 1)
        grid[ys, xs] += 1.0

        # Use PIL GaussianBlur correctly (fixes AttributeError from ImageOps)
        img = Image.fromarray(grid, mode="F")
        img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
        return np.array(img, dtype=np.float32)

    def _principal_angle(self, img: np.ndarray) -> float:
        y, x = np.where(img > img.mean())
        if len(x) < 3:
            return 0.0
        pts = np.vstack([x, y]).T.astype(float)
        pts -= pts.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(pts, full_matrices=False)
        v = vh[0]
        return float(math.degrees(math.atan2(v[1], v[0])))

    def metrics_vs_template(self, strokes: List[Stroke], template_grid: np.ndarray) -> Dict[str, float]:
        if len(strokes) == 0:
            return {"centroid_mm": 0.0, "align_deg": 0.0, "density_var": 0.0}

        pts = np.vstack([np.c_[s.x, s.y] for s in strokes])
        canvas = self._to_grid(pts)

        yy, xx = np.indices(canvas.shape)
        w_canvas = canvas + 1e-6
        cx = (xx * w_canvas).sum() / w_canvas.sum()
        cy = (yy * w_canvas).sum() / w_canvas.sum()

        w_t = template_grid + 1e-6
        cx_t = (xx * w_t).sum() / w_t.sum()
        cy_t = (yy * w_t).sum() / w_t.sum()

        centroid_mm = math.hypot(cx - cx_t, cy - cy_t) * 0.05
        ang_c = self._principal_angle(canvas)
        ang_t = self._principal_angle(template_grid)
        align_deg = abs(ang_c - ang_t)
        density_var = float(np.var(canvas / (canvas.max() + 1e-8)))

        return {"centroid_mm": float(centroid_mm),
                "align_deg": float(align_deg),
                "density_var": density_var}

# ------------------------------------------------------------
# 6) Scoring for GSPR and Engagement
# ------------------------------------------------------------
class Scoring:
    def __init__(self, cfg: DBEVMConfig):
        self.cfg = cfg

    def gspr_from_features(self, press_mod: float, curvature: float, terminal: float) -> Dict[str, float]:
        p_score = max(0.0, min(10.0, 10.0 * (press_mod / 4.0)))
        c_score = max(0.0, min(10.0, 10.0 * np.exp(-curvature)))
        t_score = max(0.0, min(10.0, 10.0 * (terminal / 40.0)))
        weighted = (self.cfg.gspr_weights[0]*p_score +
                    self.cfg.gspr_weights[1]*c_score +
                    self.cfg.gspr_weights[2]*t_score)
        total = 3.0 * weighted  # scaled to approximate 0–30
        return {"pressure": float(p_score),
                "curvature": float(c_score),
                "terminal": float(t_score),
                "total": float(total)}

    def engagement_from_stream(self, packets: List[SensorPacket]) -> Dict[str, float]:
        active = sum(1 for p in packets if p.press > 0.08)
        time_on_task_min = active / self.cfg.fs_hz / 60.0 * 60.0  # minutes
        idle = 0
        episodes, durations = 0, []
        for p in packets:
            if p.press <= 0.08 and (abs(p.vx) + abs(p.vy)) < 1e-3:
                idle += 1
            else:
                if idle > 10:
                    episodes += 1
                    durations.append(idle / self.cfg.fs_hz)
                idle = 0
        mean_dur = float(np.mean(durations)) if durations else 0.0
        return {"time_on_task_min": float(time_on_task_min),
                "distraction_n": int(episodes),
                "distraction_dur_s": float(mean_dur)}

# ------------------------------------------------------------
# 7) Session orchestration
# ------------------------------------------------------------
class SessionRunner:
    def __init__(self, cfg: DBEVMConfig):
        self.cfg = cfg
        self.model = BrushModel(cfg)
        self.overlay = AROverlay(cfg, expert_template={"press_mod": 3.2, "curvature": 0.55, "terminal": 35.0})
        self.mapper = SymbolicMapper()
        self.comp = CompositionAnalyzer(cfg)
        self.score = Scoring(cfg)
        ensure_dir(cfg.save_dir)

    def run_session(self, pid: str, phase: str, n_strokes:int = 12,
                    template_grid: Optional[np.ndarray] = None) -> Dict:
        hub = SensorHub(self.cfg)
        packets = hub.stream_session(n_strokes=n_strokes)
        strokes = self.model.segment(packets)

        # Fallback if no strokes (rare)
        if len(strokes) == 0:
            return {"participant": pid, "phase": phase, "n_packets": len(packets),
                    "n_strokes": 0, "gspr": {"total":0,"pressure":0,"curvature":0,"terminal":0},
                    "composition": {"centroid_mm":0,"align_deg":0,"density_var":0},
                    "engagement": self.score.engagement_from_stream(packets),
                    "symbols": [], "raw_metrics": []}

        metrics_per_stroke = []
        symbols = []
        for s in strokes:
            f_p = self.model.pressure_modulation(s)
            f_cu = self.model.curvature(s)
            f_tc = self.model.terminal_control(s)
            feats = {"press_mod": f_p, "curvature": f_cu, "terminal": f_tc}
            dev = self.overlay.deviation(feats)
            cues = self.overlay.feedback(dev)
            tag = self.mapper.label(feats)
            gspr = self.score.gspr_from_features(f_p, f_cu, f_tc)
            metrics_per_stroke.append({**feats, **gspr, "deviation": dev, "cues": cues})
            symbols.append(tag)

        # Template for composition (one shared template recommended across sessions)
        if template_grid is None:
            # Use the first stroke path as a simple template if none provided
            template_grid = self.comp._to_grid(np.c_[strokes[0].x, strokes[0].y])

        compm = self.comp.metrics_vs_template(strokes, template_grid)
        eng = self.score.engagement_from_stream(packets)

        df = pd.DataFrame(metrics_per_stroke)
        gspr_total = float(df["total"].mean())
        gspr_p = float(df["pressure"].mean())
        gspr_c = float(df["curvature"].mean())
        gspr_t = float(df["terminal"].mean())

        session = {
            "participant": pid,
            "phase": phase,
            "n_packets": len(packets),
            "n_strokes": len(strokes),
            "gspr": {"total": gspr_total, "pressure": gspr_p, "curvature": gspr_c, "terminal": gspr_t},
            "composition": compm,
            "engagement": eng,
            "symbols": symbols[:50],
            "raw_metrics": metrics_per_stroke
        }

        sid = f"{pid}_{phase}_{uuid.uuid4().hex[:8]}"
        out_path = os.path.join(self.cfg.save_dir, f"{sid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)
        return session

# ------------------------------------------------------------
# 8) Experiment runner (AR vs Control) and retention
# ------------------------------------------------------------
class Experiment:
    def __init__(self, cfg: DBEVMConfig):
        self.cfg = cfg
        self.runner = SessionRunner(cfg)

    def _pack_row(self, pre:Dict, post:Dict, group:str) -> Dict:
        return {
            "group": group,
            "gspr_pre": pre["gspr"]["total"],
            "gspr_post": post["gspr"]["total"],
            "gspr_d": post["gspr"]["total"] - pre["gspr"]["total"],
            "p_pre": pre["gspr"]["pressure"],  "p_post": post["gspr"]["pressure"],
            "c_pre": pre["gspr"]["curvature"], "c_post": post["gspr"]["curvature"],
            "t_pre": pre["gspr"]["terminal"],  "t_post": post["gspr"]["terminal"],
            "cent_pre": pre["composition"]["centroid_mm"],
            "cent_post": post["composition"]["centroid_mm"],
            "align_pre": pre["composition"]["align_deg"],
            "align_post": post["composition"]["align_deg"],
            "dens_pre": pre["composition"]["density_var"],
            "dens_post": post["composition"]["density_var"],
            "time_min": post["engagement"]["time_on_task_min"],
            "distractions": post["engagement"]["distraction_n"],
            "distraction_dur": post["engagement"]["distraction_dur_s"]
        }

    def run_cohort(self, n_ar:int = 32, n_ctrl:int = 32) -> pd.DataFrame:
        rows: List[Dict] = []

        # Shared template: smooth sinusoidal path
        path_x = np.linspace(0, 1, 300)
        path_y = np.sin(np.linspace(0, 2*np.pi, 300)) * 0.3 + 0.5
        template_grid = self.runner.comp._to_grid(np.c_[path_x, path_y])

        # AR group
        for i in range(n_ar):
            pid = f"AR_{i+1:02d}"
            pre  = self.runner.run_session(pid, "pre",  n_strokes=10, template_grid=template_grid)
            post = self.runner.run_session(pid, "post", n_strokes=12, template_grid=template_grid)
            rows.append(self._pack_row(pre, post, "AR"))

        # Control group
        for i in range(n_ctrl):
            pid = f"CTL_{i+1:02d}"
            pre  = self.runner.run_session(pid, "pre",  n_strokes=10, template_grid=template_grid)
            post = self.runner.run_session(pid, "post", n_strokes=10, template_grid=template_grid)
            rows.append(self._pack_row(pre, post, "Control"))

        df = pd.DataFrame(rows)
        return df

def simulate_retention(df: pd.DataFrame, cfg: DBEVMConfig) -> pd.DataFrame:
    """Adds retention columns by applying small decays (smaller for AR)."""
    out = df.copy()
    ar = out["group"] == "AR"
    ct = out["group"] == "Control"

    out.loc[ar, "gspr_ret"]  = out.loc[ar, "gspr_post"]  - np.random.normal(0.6, 0.4, ar.sum())
    out.loc[ct, "gspr_ret"]  = out.loc[ct, "gspr_post"]  - np.random.normal(2.1, 0.6, ct.sum())
    out.loc[ar, "cent_ret"]  = out.loc[ar, "cent_post"]  + np.random.normal(0.4, 0.3, ar.sum())
    out.loc[ct, "cent_ret"]  = out.loc[ct, "cent_post"]  + np.random.normal(1.6, 0.5, ct.sum())
    out.loc[ar, "align_ret"] = out.loc[ar, "align_post"] + np.random.normal(0.6, 0.3, ar.sum())
    out.loc[ct, "align_ret"] = out.loc[ct, "align_post"] + np.random.normal(2.2, 0.6, ct.sum())
    out.loc[ar, "dens_ret"]  = out.loc[ar, "dens_post"]  + np.random.normal(0.3, 0.2, ar.sum())
    out.loc[ct, "dens_ret"]  = out.loc[ct, "dens_post"]  + np.random.normal(1.4, 0.4, ct.sum())

    return out

# ------------------------------------------------------------
# 9) Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    cfg = DBEVMConfig()
    set_seed(cfg.seed)
    ensure_dir(cfg.save_dir)

    print("DBEVM cohort run started.")
    exp = Experiment(cfg)
    df_main = exp.run_cohort(n_ar=8, n_ctrl=8)  # adjust to 32/32 when desired
    df_all = simulate_retention(df_main, cfg)

    csv_path = os.path.join(cfg.save_dir, "cohort_summary.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"Cohort summary written to {csv_path}")

    # brief numeric sanity output
    def group_stats(column:str) -> pd.DataFrame:
        return df_all.groupby("group")[column].agg(["mean", "std"]).round(2)

    print("\nGSPR change by group:\n", group_stats("gspr_d"))
    print("\nCentroid displacement (post) by group:\n", group_stats("cent_post"))
    print("\nAlignment error (post) by group:\n", group_stats("align_post"))
    print("\nEngagement time (min) by group:\n", group_stats("time_min"))
