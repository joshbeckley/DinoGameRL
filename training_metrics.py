import csv
from pathlib import Path


class TrainingMetrics:
    def __init__(self, csv_path, plot_path):
        self.csv_path = Path(csv_path)
        self.plot_path = Path(plot_path)

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
            return

        charts = [
            (
                "Episode reward",
                [
                    ("Reward", "reward", "#2563eb"),
                    ("Rolling mean (100)", "rolling_mean_100", "#dc2626"),
                ],
            ),
            ("Cacti passed", [("Cacti", "cacti_passed", "#16a34a")]),
            ("Episode length", [("Steps", "steps", "#9333ea")]),
            ("Jump actions", [("Jumps", "jumps", "#ea580c")]),
            ("Exploration rate", [("Epsilon", "epsilon", "#0891b2")]),
        ]
        width = 1000
        chart_height = 190
        height = 35 + len(charts) * chart_height
        left, right, top, bottom = 65, 20, 32, 30
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;font-size:12px}'
            '.title{font-size:15px;font-weight:bold}</style>',
        ]

        for chart_index, (title, series) in enumerate(charts):
            y_offset = 25 + chart_index * chart_height
            plot_top = y_offset + top
            plot_bottom = y_offset + chart_height - bottom
            plot_width = width - left - right
            all_values = [float(row[key]) for _, key, _ in series for row in rows]
            minimum, maximum = min(all_values), max(all_values)
            if minimum == maximum:
                minimum -= 1
                maximum += 1
            else:
                padding = (maximum - minimum) * 0.08
                minimum -= padding
                maximum += padding

            svg.append(
                f'<text class="title" x="{left}" y="{y_offset + 17}">{title}</text>'
            )
            for tick in range(5):
                ratio = tick / 4
                y = plot_bottom - ratio * (plot_bottom - plot_top)
                value = minimum + ratio * (maximum - minimum)
                svg.append(
                    f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" '
                    f'y2="{y:.1f}" stroke="#e5e7eb"/>'
                )
                svg.append(
                    f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end">'
                    f'{value:.2f}</text>'
                )

            for series_index, (name, key, color) in enumerate(series):
                values = [float(row[key]) for row in rows]
                stride = max(1, len(values) // 2000)
                sampled = list(enumerate(values))[::stride]
                if sampled[-1][0] != len(values) - 1:
                    sampled.append((len(values) - 1, values[-1]))
                points = []
                for index, value in sampled:
                    x = left + index / max(1, len(values) - 1) * plot_width
                    y = plot_bottom - (value - minimum) / (maximum - minimum) * (
                        plot_bottom - plot_top
                    )
                    points.append(f"{x:.1f},{y:.1f}")
                svg.append(
                    f'<polyline fill="none" stroke="{color}" stroke-width="1.5" '
                    f'points="{" ".join(points)}"/>'
                )
                legend_x = left + series_index * 190
                svg.append(
                    f'<line x1="{legend_x}" y1="{y_offset+17}" '
                    f'x2="{legend_x+18}" y2="{y_offset+17}" stroke="{color}" '
                    'stroke-width="2"/>'
                )
                svg.append(
                    f'<text x="{legend_x+24}" y="{y_offset+21}">{name}</text>'
                )

            first_episode = rows[0]["episode"]
            last_episode = rows[-1]["episode"]
            svg.append(
                f'<text x="{left}" y="{chart_height+y_offset-5}">'
                f'episode {first_episode}</text>'
            )
            svg.append(
                f'<text x="{width-right}" y="{chart_height+y_offset-5}" '
                f'text-anchor="end">episode {last_episode}</text>'
            )

        svg.append("</svg>")
        self.plot_path.write_text("\n".join(svg), encoding="utf-8")
