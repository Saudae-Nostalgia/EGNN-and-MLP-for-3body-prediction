import argparse
import json
import os
from dataclasses import asdict

import matplotlib
import numpy as np

import imageio_ffmpeg
from Better_data_producer import DT_SAMPLE, InitialCondition, simulate_orbit

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def load_prediction(eva_path):
    if not os.path.exists(eva_path):
        raise FileNotFoundError(f"Evaluation file does not exist: {eva_path}")
    pred = np.load(eva_path)
    if pred.ndim != 2 or pred.shape[1] != 11:
        raise ValueError(f"Expected evaluation array with shape [N, 11], got {pred.shape}")
    if pred.shape[0] < 2:
        raise ValueError(f"Expected at least 2 evaluation steps, got {pred.shape[0]}")
    if not np.all(np.isfinite(pred)):
        raise ValueError("Evaluation array contains NaN or Inf")
    return pred.astype(np.float64)


def make_initial_condition_from_prediction(pred):
    masses = pred[0, :3].astype(np.float64)
    state0 = pred[0, 3:11].astype(np.float64)
    return InitialCondition(
        seed=-1,
        masses=masses.tolist(),
        state0=state0.tolist(),
        params={"source": "evaluation_first_frame"},
    )


def make_truth_orbit(pred, dt):
    initial_condition = make_initial_condition_from_prediction(pred)
    tf = (pred.shape[0] - 1) * dt
    truth, diagnostics = simulate_orbit(initial_condition, tf=tf, dt_sample=dt)
    if truth is None:
        raise RuntimeError(f"Truth integration failed: {diagnostics}")
    if truth.shape[0] != pred.shape[0]:
        min_len = min(truth.shape[0], pred.shape[0])
        truth = truth[:min_len]
        pred = pred[:min_len]
    return truth, pred, diagnostics, initial_condition


def compute_metrics(pred, truth, dt, divergence_threshold=0.1):
    pred_pos = pred[:, [3, 4, 7, 8]].reshape(pred.shape[0], 2, 2)
    truth_pos = truth[:, [3, 4, 7, 8]].reshape(truth.shape[0], 2, 2)
    pred_vel = pred[:, [5, 6, 9, 10]].reshape(pred.shape[0], 2, 2)
    truth_vel = truth[:, [5, 6, 9, 10]].reshape(truth.shape[0], 2, 2)

    pos_err_body = np.linalg.norm(pred_pos - truth_pos, axis=2)
    vel_err_body = np.linalg.norm(pred_vel - truth_vel, axis=2)
    pos_err = np.sqrt(np.mean(pos_err_body**2, axis=1))
    vel_err = np.sqrt(np.mean(vel_err_body**2, axis=1))

    typical_pos_scale = float(np.sqrt(np.mean(np.sum(truth_pos**2, axis=2))))
    typical_pos_scale = max(typical_pos_scale, 1e-12)
    relative_pos_err = pos_err / typical_pos_scale

    divergence_indices = np.where(relative_pos_err > divergence_threshold)[0]
    divergence_step = None if divergence_indices.size == 0 else int(divergence_indices[0])

    return {
        "num_steps": int(pred.shape[0]),
        "dt": float(dt),
        "duration": float((pred.shape[0] - 1) * dt),
        "typical_position_scale": typical_pos_scale,
        "divergence_threshold": float(divergence_threshold),
        "divergence_step": divergence_step,
        "divergence_time": None if divergence_step is None else float(divergence_step * dt),
        "position_rmse": float(np.sqrt(np.mean(pos_err**2))),
        "position_mae": float(np.mean(pos_err)),
        "position_final_error": float(pos_err[-1]),
        "position_max_error": float(np.max(pos_err)),
        "velocity_rmse": float(np.sqrt(np.mean(vel_err**2))),
        "velocity_final_error": float(vel_err[-1]),
        "relative_position_rmse": float(np.sqrt(np.mean(relative_pos_err**2))),
        "relative_final_position_error": float(relative_pos_err[-1]),
        "body_position_rmse": {
            "B": float(np.sqrt(np.mean(pos_err_body[:, 0] ** 2))),
            "C": float(np.sqrt(np.mean(pos_err_body[:, 1] ** 2))),
        },
        "body_position_final_error": {
            "B": float(pos_err_body[-1, 0]),
            "C": float(pos_err_body[-1, 1]),
        },
        "body_velocity_rmse": {
            "B": float(np.sqrt(np.mean(vel_err_body[:, 0] ** 2))),
            "C": float(np.sqrt(np.mean(vel_err_body[:, 1] ** 2))),
        },
        "body_velocity_final_error": {
            "B": float(vel_err_body[-1, 0]),
            "C": float(vel_err_body[-1, 1]),
        },
        "position_error_series": pos_err.tolist(),
        "relative_position_error_series": relative_pos_err.tolist(),
    }


def save_metrics(metrics_path, metrics, integration_diagnostics, initial_condition):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    payload = {
        "metrics": metrics,
        "truth_integration_diagnostics": integration_diagnostics,
        "initial_condition": asdict(initial_condition),
    }
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_comparison_animation(pred, truth, metrics, out_video, dt):
    import pylab as py
    from matplotlib import animation

    os.makedirs(os.path.dirname(out_video), exist_ok=True)

    pred_b = pred[:, 3:5]
    pred_c = pred[:, 7:9]
    truth_b = truth[:, 3:5]
    truth_c = truth[:, 7:9]
    all_xy = np.concatenate([pred[:, [3, 4, 7, 8]], truth[:, [3, 4, 7, 8]]], axis=0)
    radius = max(7.2, float(np.max(np.abs(all_xy))) * 1.1)
    times = np.arange(pred.shape[0]) * dt
    pos_err = np.array(metrics["position_error_series"])
    rel_err = np.array(metrics["relative_position_error_series"])

    fig, ax = py.subplots(figsize=(7, 7))
    ax.axis("square")
    ax.set_xlim((-radius, radius))
    ax.set_ylim((-radius, radius))
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.plot(0, 0, "o", markersize=9, markerfacecolor="#FDB813", markeredgecolor="#FD7813", label="Sun")

    truth_b_line, = ax.plot([], [], "-", color="#0077BE", lw=2.0, label="Truth B")
    truth_c_line, = ax.plot([], [], "-", color="#f66338", lw=2.0, label="Truth C")
    pred_b_line, = ax.plot([], [], "--", color="#73c7ff", lw=1.8, label="Pred B")
    pred_c_line, = ax.plot([], [], "--", color="#ffad8d", lw=1.8, label="Pred C")
    truth_b_dot, = ax.plot([], [], "o", color="#0077BE", markersize=5)
    truth_c_dot, = ax.plot([], [], "o", color="#f66338", markersize=5)
    pred_b_dot, = ax.plot([], [], "o", color="#73c7ff", markersize=4)
    pred_c_dot, = ax.plot([], [], "o", color="#ffad8d", markersize=4)
    title = ax.text(0.02, 1.04, "", transform=ax.transAxes, va="center")
    ax.legend(loc="lower left", frameon=False)

    def init():
        for artist in [
            truth_b_line,
            truth_c_line,
            pred_b_line,
            pred_c_line,
            truth_b_dot,
            truth_c_dot,
            pred_b_dot,
            pred_c_dot,
        ]:
            artist.set_data([], [])
        title.set_text("")
        return (
            truth_b_line,
            truth_c_line,
            pred_b_line,
            pred_c_line,
            truth_b_dot,
            truth_c_dot,
            pred_b_dot,
            pred_c_dot,
            title,
        )

    def animate(i):
        truth_b_line.set_data(truth_b[:i + 1, 0], truth_b[:i + 1, 1])
        truth_c_line.set_data(truth_c[:i + 1, 0], truth_c[:i + 1, 1])
        pred_b_line.set_data(pred_b[:i + 1, 0], pred_b[:i + 1, 1])
        pred_c_line.set_data(pred_c[:i + 1, 0], pred_c[:i + 1, 1])
        truth_b_dot.set_data([truth_b[i, 0]], [truth_b[i, 1]])
        truth_c_dot.set_data([truth_c[i, 0]], [truth_c[i, 1]])
        pred_b_dot.set_data([pred_b[i, 0]], [pred_b[i, 1]])
        pred_c_dot.set_data([pred_c[i, 0]], [pred_c[i, 1]])
        title.set_text(
            f"t={times[i]:.1f} yr | pos err={pos_err[i]:.4g} | rel err={rel_err[i]:.3%}"
        )
        return (
            truth_b_line,
            truth_c_line,
            pred_b_line,
            pred_c_line,
            truth_b_dot,
            truth_c_dot,
            pred_b_dot,
            pred_c_dot,
            title,
        )

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=pred.shape[0], interval=40, blit=True)
    anim.save(out_video, fps=30, dpi=250, extra_args=["-vcodec", "libx264"])
    py.close(fig)


def compare_evaluation(eva_path, out_video, metrics_path, dt, divergence_threshold):
    pred = load_prediction(eva_path)
    truth, pred, diagnostics, initial_condition = make_truth_orbit(pred, dt)
    metrics = compute_metrics(pred, truth, dt, divergence_threshold)
    save_metrics(metrics_path, metrics, diagnostics, initial_condition)
    save_comparison_animation(pred, truth, metrics, out_video, dt)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Compare an evaluated orbit against a re-integrated truth orbit.")
    parser.add_argument("--eva-path", default="eva_results/MLP_eva.npy")
    parser.add_argument("--out-video", default="eva_results/MLP_compare.mp4")
    parser.add_argument("--metrics-path", default="eva_results/MLP_compare_metrics.json")
    parser.add_argument("--dt", type=float, default=DT_SAMPLE)
    parser.add_argument("--divergence-threshold", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = compare_evaluation(
        eva_path=args.eva_path,
        out_video=args.out_video,
        metrics_path=args.metrics_path,
        dt=args.dt,
        divergence_threshold=args.divergence_threshold,
    )
    summary_keys = [
        "position_rmse",
        "position_final_error",
        "position_max_error",
        "relative_position_rmse",
        "relative_final_position_error",
        "velocity_rmse",
        "velocity_final_error",
        "divergence_step",
        "divergence_time",
    ]
    print(json.dumps({key: metrics[key] for key in summary_keys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
