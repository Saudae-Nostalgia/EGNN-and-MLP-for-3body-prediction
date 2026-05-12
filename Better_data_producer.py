import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

import matplotlib
import numpy as np
from scipy.integrate import solve_ivp

import imageio_ffmpeg

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


G_SI = 6.673e-11
RR = 1.496e11
MM = 6e24
TT = 365 * 24 * 60 * 60.0
GG = (MM * G_SI * TT**2) / (RR**3)

BASE_SUN_MASS = 2e30 / MM
BASE_PLANET_MASS = 1.0

TF = 20.0
DT_SAMPLE = 0.1
METHOD = "DOP853"
RTOL = 1e-9
ATOL = 1e-11

DATA_DIR = "data"
DEFAULT_X_PATH = os.path.join(DATA_DIR, "better_X.npy")
DEFAULT_Y_PATH = os.path.join(DATA_DIR, "better_Y.npy")
DEFAULT_METADATA_PATH = os.path.join(DATA_DIR, "better_metadata.jsonl")
DEFAULT_SAMPLE_X_PATH = os.path.join(DATA_DIR, "better_sample_X.npy")
DEFAULT_SAMPLE_Y_PATH = os.path.join(DATA_DIR, "better_sample_Y.npy")
DEFAULT_SAMPLE_METADATA_PATH = os.path.join(DATA_DIR, "better_sample_metadata.jsonl")

MIN_DISTANCE = 0.03
MAX_RADIUS_STABLE = 8.0
MAX_RADIUS_HIGH_ENERGY = 30.0
MAX_RADIUS_REJECT = 80.0
MAX_SPEED = 80.0
MAX_ACCEL = 1.0e4
MAX_REL_DRIFT = 5.0e-3
VELOCITY_SCALE_MEAN = 0.75
VELOCITY_SCALE_STD = 0.12
VELOCITY_ANGLE_JITTER = 0.35
ANIMATION_TRAIL_POINTS = 5

DEFAULT_TARGET_COUNTS = {
    "stable": 500,
    "high_energy": 100,
    "escape": 100,
}

SAMPLING_PROFILES = {
    "stable": {"r_mean": 3.0, "r_std": 1.0, "v_mean": 0.95, "v_std": 0.12},
    "high_energy": {"r_mean": 5.0, "r_std": 1.5, "v_mean": 1.20, "v_std": 0.12},
    "escape": {"r_mean": 4.0, "r_std": 1.5, "v_mean": 1.45, "v_std": 0.10},
}


@dataclass
class InitialCondition:
    seed: int
    masses: list
    state0: list
    params: dict


def sample_float_by_log_uniform(rng, min_val, max_val):
    log_min = math.log10(min_val)
    log_max = math.log10(max_val)
    return 10 ** rng.uniform(log_min, log_max)


def rotate_vector(vec, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x, y = vec
    return np.array([cos_a * x - sin_a * y, sin_a * x + cos_a * y], dtype=np.float64)


def sample_initial_condition(seed, target_label="stable"):
    rng = np.random.default_rng(seed)
    profile = SAMPLING_PROFILES.get(target_label, SAMPLING_PROFILES["stable"])

    sun_scale = sample_float_by_log_uniform(rng, 0.1, 2.0)
    b_mass_scale = sample_float_by_log_uniform(rng, 0.1, 1000.0)
    c_mass_scale = sample_float_by_log_uniform(rng, 0.1, 1000.0)

    m_a = BASE_SUN_MASS * sun_scale
    m_b = BASE_PLANET_MASS * b_mass_scale
    m_c = BASE_PLANET_MASS * c_mass_scale

    r_b_scale = max(0.3, rng.normal(profile["r_mean"], profile["r_std"]))
    r_c_scale = max(0.3, rng.normal(profile["r_mean"], profile["r_std"]))

    angle_r_b = rng.uniform(0.0, 2.0 * math.pi)
    angle_r_c = rng.uniform(0.0, 2.0 * math.pi)
    velocity_jitter_b = rng.normal(0.0, VELOCITY_ANGLE_JITTER)
    velocity_jitter_c = rng.normal(0.0, VELOCITY_ANGLE_JITTER)

    r_b = rotate_vector([r_b_scale, 0.0], angle_r_b)
    r_c = rotate_vector([r_c_scale, 0.0], angle_r_c)

    v_b_mag = math.sqrt(GG * m_a / max(np.linalg.norm(r_b), MIN_DISTANCE))
    v_c_mag = math.sqrt(GG * m_a / max(np.linalg.norm(r_c), MIN_DISTANCE))
    v_b_scale = max(0.1, rng.normal(profile["v_mean"], profile["v_std"]))
    v_c_scale = max(0.1, rng.normal(profile["v_mean"], profile["v_std"]))
    angle_v_b = angle_r_b + velocity_jitter_b
    angle_v_c = angle_r_c + velocity_jitter_c
    v_b = rotate_vector([0.0, v_b_mag * v_b_scale], angle_v_b)
    v_c = rotate_vector([0.0, v_c_mag * v_c_scale], angle_v_c)

    state0 = np.concatenate([r_b, v_b, r_c, v_c]).astype(np.float64)
    params = {
        "target_label": target_label,
        "r_mean": profile["r_mean"],
        "r_std": profile["r_std"],
        "v_mean": profile["v_mean"],
        "v_std": profile["v_std"],
        "sun_scale": sun_scale,
        "b_mass_scale": b_mass_scale,
        "c_mass_scale": c_mass_scale,
        "r_b_scale": r_b_scale,
        "r_c_scale": r_c_scale,
        "angle_r_b": angle_r_b,
        "angle_r_c": angle_r_c,
        "angle_v_b": angle_v_b,
        "angle_v_c": angle_v_c,
        "v_b_scale": v_b_scale,
        "v_c_scale": v_c_scale,
        "velocity_jitter_b": velocity_jitter_b,
        "velocity_jitter_c": velocity_jitter_c,
    }
    return InitialCondition(seed=seed, masses=[m_a, m_b, m_c], state0=state0.tolist(), params=params)


def pairwise_distances(state):
    r_b = state[..., 0:2]
    r_c = state[..., 4:6]
    d_ab = np.linalg.norm(r_b, axis=-1)
    d_ac = np.linalg.norm(r_c, axis=-1)
    d_bc = np.linalg.norm(r_b - r_c, axis=-1)
    return d_ab, d_ac, d_bc


def acceleration_terms(state, masses):
    m_a, m_b, m_c = masses
    r_b = state[..., 0:2]
    r_c = state[..., 4:6]
    rb_norm = np.linalg.norm(r_b, axis=-1, keepdims=True)
    rc_norm = np.linalg.norm(r_c, axis=-1, keepdims=True)
    rbc = r_c - r_b
    rbc_norm = np.linalg.norm(rbc, axis=-1, keepdims=True)

    rb_norm = np.maximum(rb_norm, 1e-12)
    rc_norm = np.maximum(rc_norm, 1e-12)
    rbc_norm = np.maximum(rbc_norm, 1e-12)

    a_b = -GG * m_a * r_b / rb_norm**3 + GG * m_c * rbc / rbc_norm**3
    a_c = -GG * m_a * r_c / rc_norm**3 - GG * m_b * rbc / rbc_norm**3
    return a_b, a_c


def derivative(t, state, masses):
    del t
    state = np.asarray(state, dtype=np.float64)
    a_b, a_c = acceleration_terms(state, masses)
    return np.array([
        state[2],
        state[3],
        a_b[0],
        a_b[1],
        state[6],
        state[7],
        a_c[0],
        a_c[1],
    ], dtype=np.float64)


def total_energy(states, masses):
    m_a, m_b, m_c = masses
    r_b = states[:, 0:2]
    v_b = states[:, 2:4]
    r_c = states[:, 4:6]
    v_c = states[:, 6:8]
    d_ab, d_ac, d_bc = pairwise_distances(states)

    kinetic = 0.5 * m_b * np.sum(v_b**2, axis=1) + 0.5 * m_c * np.sum(v_c**2, axis=1)
    potential = -GG * m_a * m_b / d_ab - GG * m_a * m_c / d_ac - GG * m_b * m_c / d_bc
    return kinetic + potential


def total_angular_momentum(states, masses):
    _, m_b, m_c = masses
    r_b = states[:, 0:2]
    v_b = states[:, 2:4]
    r_c = states[:, 4:6]
    v_c = states[:, 6:8]
    l_b = m_b * (r_b[:, 0] * v_b[:, 1] - r_b[:, 1] * v_b[:, 0])
    l_c = m_c * (r_c[:, 0] * v_c[:, 1] - r_c[:, 1] * v_c[:, 0])
    return l_b + l_c


def relative_drift(values):
    baseline = max(abs(values[0]), 1e-12)
    return float(np.max(np.abs(values - values[0])) / baseline)


def simulate_orbit(initial_condition, tf=TF, dt_sample=DT_SAMPLE, method=METHOD, rtol=RTOL, atol=ATOL):
    masses = np.array(initial_condition.masses, dtype=np.float64)
    state0 = np.array(initial_condition.state0, dtype=np.float64)
    sample_times = np.arange(0.0, tf + 0.5 * dt_sample, dt_sample, dtype=np.float64)

    solution = solve_ivp(
        fun=lambda t, y: derivative(t, y, masses),
        t_span=(0.0, tf),
        y0=state0,
        method=method,
        t_eval=sample_times,
        rtol=rtol,
        atol=atol,
    )

    diagnostics = {
        "success": bool(solution.success),
        "message": str(solution.message),
        "nfev": int(solution.nfev),
        "num_steps": int(solution.t.size),
    }

    if not solution.success or solution.y.shape[1] != sample_times.size:
        return None, diagnostics

    dynamic_states = solution.y.T.astype(np.float64)
    mass_cols = np.repeat(masses[None, :], dynamic_states.shape[0], axis=0)
    states = np.concatenate([mass_cols, dynamic_states], axis=1)

    d_ab, d_ac, d_bc = pairwise_distances(dynamic_states)
    speed = np.maximum(
        np.linalg.norm(dynamic_states[:, 2:4], axis=1),
        np.linalg.norm(dynamic_states[:, 6:8], axis=1),
    )
    a_b, a_c = acceleration_terms(dynamic_states, masses)
    accel = np.maximum(np.linalg.norm(a_b, axis=1), np.linalg.norm(a_c, axis=1))
    energy = total_energy(dynamic_states, masses)
    angular = total_angular_momentum(dynamic_states, masses)

    diagnostics.update({
        "min_distance": float(np.min([d_ab.min(), d_ac.min(), d_bc.min()])),
        "max_radius": float(max(d_ab.max(), d_ac.max())),
        "final_radius": float(max(d_ab[-1], d_ac[-1])),
        "max_speed": float(speed.max()),
        "max_accel": float(accel.max()),
        "energy_drift": relative_drift(energy),
        "angular_drift": relative_drift(angular),
    })
    return states, diagnostics


def classify_orbit(states, diagnostics):
    if states is None:
        return "reject", "integration_failed"
    if not np.all(np.isfinite(states)):
        return "reject", "nan_or_inf"
    if diagnostics["min_distance"] < MIN_DISTANCE:
        return "reject", "close_encounter"
    if diagnostics["max_radius"] > MAX_RADIUS_REJECT:
        return "reject", "radius_explosion"
    if diagnostics["max_speed"] > MAX_SPEED:
        return "reject", "speed_explosion"
    if diagnostics["max_accel"] > MAX_ACCEL:
        return "reject", "accel_explosion"
    if diagnostics["energy_drift"] > MAX_REL_DRIFT:
        return "reject", "energy_drift"
    if diagnostics["angular_drift"] > MAX_REL_DRIFT:
        return "reject", "angular_drift"

    if diagnostics["max_radius"] <= MAX_RADIUS_STABLE:
        return "stable", "accepted"
    if diagnostics["max_radius"] <= MAX_RADIUS_HIGH_ENERGY:
        return "high_energy", "accepted"
    return "escape", "accepted"


def states_to_xy(states):
    return states[:-1], states[1:]


def write_metadata(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_dataset(x_parts, y_parts, x_path, y_path):
    os.makedirs(os.path.dirname(x_path), exist_ok=True)
    if x_parts:
        X = np.concatenate(x_parts, axis=0).astype(np.float32)
        Y = np.concatenate(y_parts, axis=0).astype(np.float32)
    else:
        X = np.empty((0, 11), dtype=np.float32)
        Y = np.empty((0, 11), dtype=np.float32)
    np.save(x_path, X)
    np.save(y_path, Y)
    return X, Y


def choose_target_label(target_counts, accepted_counts):
    missing = {
        label: target_counts[label] - accepted_counts.get(label, 0)
        for label in target_counts
        if target_counts[label] > accepted_counts.get(label, 0)
    }
    if not missing:
        return "stable"
    return max(missing, key=missing.get)


def batch_generate(
    start_seed=0,
    max_attempts=0,
    target_counts=None,
    target_total=None,
    x_path=DEFAULT_X_PATH,
    y_path=DEFAULT_Y_PATH,
    metadata_path=DEFAULT_METADATA_PATH,
    sample_x_path=DEFAULT_SAMPLE_X_PATH,
    sample_y_path=DEFAULT_SAMPLE_Y_PATH,
    sample_metadata_path=DEFAULT_SAMPLE_METADATA_PATH,
    progress_interval=100,
):
    if target_counts is None:
        target_counts = dict(DEFAULT_TARGET_COUNTS)
    if target_total is not None:
        target_counts = {"stable": int(target_total), "high_energy": 0, "escape": 0}

    accepted_counts = {name: 0 for name in target_counts}
    x_parts = []
    y_parts = []
    metadata_records = []
    sample_x_parts = []
    sample_y_parts = []
    sample_metadata_records = []
    sample_labels = {label for label, count in target_counts.items() if count > 0}
    saved_sample_labels = set()
    reject_counts = {}

    seed = start_seed
    attempts = 0
    while max_attempts <= 0 or attempts < max_attempts:
        attempts += 1
        target_label = choose_target_label(target_counts, accepted_counts)
        initial_condition = sample_initial_condition(seed, target_label=target_label)
        states, diagnostics = simulate_orbit(initial_condition)
        label, reason = classify_orbit(states, diagnostics)

        record = {
            "seed": seed,
            "target_label": target_label,
            "label": label,
            "reason": reason,
            "initial_condition": asdict(initial_condition),
            "diagnostics": diagnostics,
        }

        if label in target_counts and accepted_counts[label] < target_counts[label]:
            X, Y = states_to_xy(states)
            x_parts.append(X)
            y_parts.append(Y)
            accepted_counts[label] += 1
            record["saved"] = True
            record["num_samples"] = int(X.shape[0])
            total_accepted = sum(accepted_counts.values())
            target_total_count = sum(target_counts.values())
            print(
                f"[accepted] seed={seed} label={label} "
                f"{accepted_counts[label]}/{target_counts[label]} "
                f"total={total_accepted}/{target_total_count} attempts={attempts}",
                flush=True,
            )

            if label in sample_labels and label not in saved_sample_labels:
                sample_x_parts.append(X)
                sample_y_parts.append(Y)
                sample_record = dict(record)
                sample_record["sample_saved"] = True
                sample_metadata_records.append(sample_record)
                saved_sample_labels.add(label)
        else:
            record["saved"] = False
            record["num_samples"] = 0
            if label == "reject":
                reject_counts[reason] = reject_counts.get(reason, 0) + 1

        metadata_records.append(record)
        if all(accepted_counts[name] >= target_counts[name] for name in target_counts):
            break
        if progress_interval > 0 and attempts % progress_interval == 0:
            print(
                f"[progress] attempts={attempts} accepted={accepted_counts} "
                f"rejects={reject_counts}",
                flush=True,
            )
        seed += 1

    X, Y = save_dataset(x_parts, y_parts, x_path, y_path)
    sample_X, sample_Y = save_dataset(sample_x_parts, sample_y_parts, sample_x_path, sample_y_path)
    write_metadata(metadata_path, metadata_records)
    write_metadata(sample_metadata_path, sample_metadata_records)
    missing_sample_labels = sorted(sample_labels - saved_sample_labels)
    return {
        "X_shape": X.shape,
        "Y_shape": Y.shape,
        "sample_X_shape": sample_X.shape,
        "sample_Y_shape": sample_Y.shape,
        "accepted_counts": accepted_counts,
        "target_counts": target_counts,
        "attempts": len(metadata_records),
        "missing_sample_labels": missing_sample_labels,
        "reject_counts": reject_counts,
        "x_path": x_path,
        "y_path": y_path,
        "metadata_path": metadata_path,
        "sample_x_path": sample_x_path,
        "sample_y_path": sample_y_path,
        "sample_metadata_path": sample_metadata_path,
    }


def single_preview(seed=0, save_animation=False, animation_path="better_orbit.mp4"):
    initial_condition = sample_initial_condition(seed)
    states, diagnostics = simulate_orbit(initial_condition)
    label, reason = classify_orbit(states, diagnostics)

    if save_animation and states is not None:
        save_orbit_animation(states, animation_path)

    return states, {
        "seed": seed,
        "label": label,
        "reason": reason,
        "initial_condition": asdict(initial_condition),
        "diagnostics": diagnostics,
    }


def save_orbit_animation(states, animation_path):
    import pylab as py
    from matplotlib import animation

    r_b = states[:, 3:5]
    r_c = states[:, 7:9]
    times = np.arange(states.shape[0]) * DT_SAMPLE

    fig, ax = py.subplots()
    ax.axis("square")
    radius = max(7.2, float(np.max(np.abs(states[:, [3, 4, 7, 8]]))) * 1.1)
    ax.set_xlim((-radius, radius))
    ax.set_ylim((-radius, radius))
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.plot(0, 0, "o", markersize=9, markerfacecolor="#FDB813", markeredgecolor="#FD7813")
    line_b, = ax.plot([], [], "o-", color="#d2eeff", markerfacecolor="#0077BE", lw=2)
    line_c, = ax.plot([], [], "o-", color="#e3dccb", markerfacecolor="#f66338", lw=2)
    title = ax.text(0.24, 1.05, "", transform=ax.transAxes, va="center")

    def init():
        line_b.set_data([], [])
        line_c.set_data([], [])
        title.set_text("")
        return line_b, line_c, title

    def animate(i):
        title.set_text("Elapsed time = " + str(round(times[i], 1)) + " years")
        start = max(0, i - ANIMATION_TRAIL_POINTS + 1)
        line_b.set_data(r_b[start:i + 1, 0], r_b[start:i + 1, 1])
        line_c.set_data(r_c[start:i + 1, 0], r_c[start:i + 1, 1])
        return line_b, line_c

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(times), interval=5, blit=True)
    anim.save(animation_path, fps=30, dpi=300, extra_args=["-vcodec", "libx264"])
    py.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate filtered 3-body datasets with adaptive integration.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    preview = subparsers.add_parser("preview")
    preview.add_argument("--seed", type=int, default=0)
    preview.add_argument("--save-animation", action="store_true")
    preview.add_argument("--animation-path", default="better_orbit.mp4")

    batch = subparsers.add_parser("batch")
    batch.add_argument("--start-seed", type=int, default=0)
    batch.add_argument("--max-attempts", type=int, default=0)
    batch.add_argument("--stable", type=int, default=DEFAULT_TARGET_COUNTS["stable"])
    batch.add_argument("--high-energy", type=int, default=DEFAULT_TARGET_COUNTS["high_energy"])
    batch.add_argument("--escape", type=int, default=DEFAULT_TARGET_COUNTS["escape"])
    batch.add_argument("--x-path", default=DEFAULT_X_PATH)
    batch.add_argument("--y-path", default=DEFAULT_Y_PATH)
    batch.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    batch.add_argument("--sample-x-path", default=DEFAULT_SAMPLE_X_PATH)
    batch.add_argument("--sample-y-path", default=DEFAULT_SAMPLE_Y_PATH)
    batch.add_argument("--sample-metadata-path", default=DEFAULT_SAMPLE_METADATA_PATH)
    batch.add_argument("--progress-interval", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "preview":
        states, info = single_preview(args.seed, args.save_animation, args.animation_path)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        if states is not None:
            print("states shape:", states.shape)
    elif args.mode == "batch":
        result = batch_generate(
            start_seed=args.start_seed,
            max_attempts=args.max_attempts,
            target_counts={
                "stable": args.stable,
                "high_energy": args.high_energy,
                "escape": args.escape,
            },
            x_path=args.x_path,
            y_path=args.y_path,
            metadata_path=args.metadata_path,
            sample_x_path=args.sample_x_path,
            sample_y_path=args.sample_y_path,
            sample_metadata_path=args.sample_metadata_path,
            progress_interval=args.progress_interval,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
