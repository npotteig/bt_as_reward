import argparse
import glob
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

# Enable LaTeX rendering in matplotlib
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 16,
        "font.size": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)


def smooth(values, weight=0.9):
    """Exponential moving average smoothing."""
    if len(values) == 0:
        return values
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(smoothed[-1] * weight + (1 - weight) * v)
    return np.array(smoothed)


def find_event_files(root_dir):
    pattern = os.path.join(root_dir, "**", "events.out.tfevents.*")
    return sorted(glob.glob(pattern, recursive=True))


def load_scalars(event_files, tag):
    """Load scalars from multiple event files and align by step."""
    data = {}
    for file in event_files:
        ea = event_accumulator.EventAccumulator(file)
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            continue
        events = ea.Scalars(tag)
        for e in events:
            if e.step not in data:
                data[e.step] = []
            data[e.step].append(e.value)
    # Sort by step
    steps = sorted(data.keys())
    values = [data[s] for s in steps]
    # Pad missing runs with NaN and compute mean/std
    max_len = max(len(v) for v in values)
    values_padded = np.array(
        [np.pad(v, (0, max_len - len(v)), constant_values=np.nan) for v in values],
        dtype=float,
    )
    mean = np.nanmean(values_padded, axis=1)
    std = np.nanstd(values_padded, axis=1)
    return np.array(steps), mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot TensorBoard Scalars")
    parser.add_argument("--logdir", type=str, help="Directory containing TensorBoard event files")
    parser.add_argument("--tag", type=str, required=True, help="Tag of the scalar to plot")
    parser.add_argument("--env_name", type=str, required=True, help="Environment name to filter logs")
    parser.add_argument("--out_file", type=str, default="plots/plot.png", help="Output image file name")
    parser.add_argument("--smooth", type=float, default=0.9, help="Smoothing factor")
    parser.add_argument("--uncertainty", type=str, default="false", help="Whether to plot uncertainty (shaded area)")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    event_files = find_event_files(args.logdir)
    if not event_files:
        print(f"No event files found in {args.logdir}")
        exit(1)

    # Each key can have multiple files
    tensorboard_dict = {
        "environment": {"label": "Task", "files": []},
        "proc_as_reward": {"label": "Procedure", "files": []},
        "bt_as_reward": {"label": "RBT", "files": []},
        "bt_as_reward_mask": {"label": "MRBT", "files": []},
    }

    for event_file in event_files:
        if args.env_name in event_file:
            if ((args.uncertainty == "true" and "uncertainty" in event_file) or (
                args.uncertainty == "false" and "uncertainty" not in event_file
            )) and ("dependent" not in event_file):
                for key in tensorboard_dict.keys():
                    if key in event_file:
                        if "mask" in event_file and "mask" not in key:
                            continue
                        tensorboard_dict[key]["files"].append(event_file)

    plt.figure(figsize=(5, 5))
    uncertainty_str = "Stochastic" if args.uncertainty == "true" else "Deterministic"
    plt.title(f"LockedRoom {uncertainty_str}")
    plt.xlabel("Step")
    plt.ylabel(f"Success Rate")
    plt.ylim(0, 1.05)

    for key, data in tensorboard_dict.items():
        if not data["files"]:
            print(f"No event files found for {key} with env {args.env_name}")
            continue

        steps, mean_values, std_values = load_scalars(data["files"], args.tag)
        smoothed_mean = smooth(mean_values, weight=args.smooth)
        smoothed_std = smooth(std_values, weight=args.smooth)
        
        plt.plot(steps, smoothed_mean, label=data["label"])
        plt.fill_between(
            steps,
            (smoothed_mean - smoothed_std),
            (smoothed_mean + smoothed_std),
            alpha=0.2,
        )

    # plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(args.out_file)
    plt.close()
    print(f"Plot saved to {args.out_file}")
