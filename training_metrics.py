import csv
from pathlib import Path

import matplotlib.pyplot as plt


class TrainingMetrics:
    def __init__(self, csv_path, plot_path):
        self.csv_path = Path(csv_path)
        self.plot_path = Path(plot_path)

    def reset(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.write_text("", encoding="utf-8")

    def record_episode(self, row):
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", newline="") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def plot(self):
        with self.csv_path.open(newline="") as metrics_file:
            rows = list(csv.DictReader(metrics_file))
        if not rows:
            return []

        charts = [
            (
                "Episode reward",
                "episode_reward",
                "Reward",
                "reward",
                "#2563eb",
                "rolling_mean_100",
                True,
            ),
            (
                "Cacti passed",
                "cacti_passed",
                "Cacti",
                "cacti_passed",
                "#16a34a",
                None,
                True,
            ),
            (
                "Episode length",
                "episode_length",
                "Steps",
                "steps",
                "#9333ea",
                None,
                True,
            ),
            (
                "Jump actions",
                "jump_actions",
                "Jumps",
                "jumps",
                "#ea580c",
                None,
                True,
            ),
            (
                "Exploration rate",
                "exploration_rate",
                "Epsilon",
                "epsilon",
                "#0891b2",
                None,
                False,
            ),
        ]
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        episodes = [int(row["episode"]) for row in rows]
        saved_paths = []

        for title, filename, label, key, color, mean_key, show_mean in charts:
            figure, axis = plt.subplots(figsize=(10, 5))
            values = [float(row[key]) for row in rows]
            axis.plot(episodes, values, label=label, color=color, linewidth=1.2)

            if show_mean:
                if mean_key:
                    means = [float(row[mean_key]) for row in rows]
                else:
                    window = 100
                    rolling_total = 0.0
                    means = []
                    for index, value in enumerate(values):
                        rolling_total += value
                        if index >= window:
                            rolling_total -= values[index - window]
                        means.append(rolling_total / min(index + 1, window))
                axis.plot(
                    episodes,
                    means,
                    label="Rolling mean (100)",
                    color="#dc2626",
                    linewidth=2,
                )

            axis.set_title(title)
            axis.set_xlabel("Episode")
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.3)
            if show_mean:
                axis.legend()
            figure.tight_layout()

            output_path = self.plot_path.with_name(
                f"{self.plot_path.stem}_{filename}.png"
            )
            try:
                figure.savefig(output_path, dpi=300, bbox_inches="tight")
            finally:
                plt.close(figure)
            saved_paths.append(output_path)

        return saved_paths
