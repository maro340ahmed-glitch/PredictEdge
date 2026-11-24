
"""PredictEdge educational ML trainer.

Uses editable machine_data.csv.

The generated dataset is synthetic.
The failure prediction horizon is one hour.
Results are not validated on real industrial machines.
"""

from pathlib import Path
import argparse
import csv
import json
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parent

SUMMARY_PERIOD_S = 100
INITIAL_SUMMARIES = 6

TREND_WINDOW_S = (
    INITIAL_SUMMARIES - 1
) * SUMMARY_PERIOD_S

INFERENCE_PERIOD_S = 300
HORIZON_S = 3600
ALERT_THRESHOLD = 0.80

SEED = 20261113
RNG = np.random.default_rng(SEED)

FEATURE_NAMES = [
    "temp_c",
    "current_a",
    "vibration_g",
    "dtemp_c_s",
    "dcurrent_a_s",
    "dvibration_g_s"
]

SCENARIOS = [
    "normal",
    "temperature",
    "current",
    "vibration",
    "temp_current",
    "temp_vibration",
    "current_vibration",
    "combined"
]


def generate_csv(path: Path, number_of_runs: int = 400):
    """Generate simulated machine trajectories.

    Every row represents a 100-second summary.
    Failure time is a synthetic label.
    """

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "run_id",
            "scenario",
            "time_s",
            "temp_c",
            "current_a",
            "vibration_g",
            "failure_time_s"
        ])

        for run_id in range(number_of_runs):

            kind = SCENARIOS[
                run_id % len(SCENARIOS)
            ]

            fail_time = (
                int(
                    RNG.integers(
                        13000,
                        27501
                    ) // 100 * 100
                )
                if kind != "normal"
                else None
            )

            ramp_length = int(
                RNG.integers(6500, 12001)
            )

            onset = (
                fail_time - ramp_length
                if fail_time is not None
                else 40000
            )

            base = np.array([
                RNG.uniform(28, 39),
                RNG.uniform(3.6, 6.3),
                RNG.uniform(0.08, 0.25)
            ])

            change = np.array([
                RNG.uniform(15, 29),
                RNG.uniform(2.2, 5.1),
                RNG.uniform(0.43, 1.07)
            ])

            mask = np.array([
                "temperature" in kind
                or "temp_" in kind
                or kind == "combined",

                "current" in kind
                or kind == "combined",

                "vibration" in kind
                or kind == "combined"
            ], dtype=float)

            if kind == "temperature":
                mask = np.array([1, 0, 0], dtype=float)

            if kind == "current":
                mask = np.array([0, 1, 0], dtype=float)

            if kind == "vibration":
                mask = np.array([0, 0, 1], dtype=float)

            if kind == "temp_current":
                mask = np.array([1, 1, 0], dtype=float)

            if kind == "temp_vibration":
                mask = np.array([1, 0, 1], dtype=float)

            if kind == "current_vibration":
                mask = np.array([0, 1, 1], dtype=float)

            if kind == "combined":
                mask = np.ones(3)

            drift = RNG.normal(
                0,
                [0.12, 0.03, 0.005],
                (321, 3)
            ).cumsum(axis=0) * 0.18

            spike_at = int(
                RNG.integers(50, 280)
            )

            for idx in range(321):
                time_s = (
                    idx + 1
                ) * SUMMARY_PERIOD_S

                effective_time = (
                    time_s -
                    SUMMARY_PERIOD_S / 2
                )

                level = (
                    np.clip(
                        (
                            effective_time - onset
                        ) / ramp_length,
                        0,
                        1.25
                    )
                    if fail_time
                    else 0.0
                )

                degradation = np.array([
                    level,
                    level,
                    level ** 1.25
                ])

                vals = (
                    base +
                    mask * change * degradation +
                    drift[idx]
                )

                vals += RNG.normal(
                    0,
                    [0.22, 0.075, 0.013],
                    3
                )

                if idx in (
                    spike_at,
                    spike_at + 1
                ):
                    vals += np.array([
                        RNG.uniform(0, 2.5),
                        RNG.uniform(0, 0.32),
                        RNG.uniform(0.0, 0.13)
                    ])

                vals = np.maximum(
                    vals,
                    [0, 0, 0]
                )

                writer.writerow([
                    run_id,
                    kind,
                    time_s,
                    *[
                        round(float(v), 6)
                        for v in vals
                    ],
                    fail_time if fail_time else ""
                ])

    print(
        f"Generated {path} "
        f"({number_of_runs} synthetic runs)"
    )


def read_and_featurize(path: Path):

    groups = {}

    with path.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        required = {
            "run_id",
            "scenario",
            "time_s",
            "temp_c",
            "current_a",
            "vibration_g",
            "failure_time_s"
        }

        if not required.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                "CSV must include columns: "
                + ", ".join(sorted(required))
            )

        for row in reader:
            rid = row["run_id"]

            fail_time = (
                float(row["failure_time_s"])
                if row["failure_time_s"].strip()
                else None
            )

            vals = [
                float(row[k])
                for k in FEATURE_NAMES[:3]
            ]

            t = float(row["time_s"])

            if (
                not np.isfinite(vals).all()
                or not np.isfinite(t)
            ):
                raise ValueError(
                    f"Nonfinite measurement in run {rid}"
                )

            groups.setdefault(
                rid, []
            ).append((
                t,
                vals,
                fail_time,
                row["scenario"]
            ))

    runs = []

    for rid, entries in sorted(groups.items()):

        entries.sort(
            key=lambda item: item[0]
        )

        times = np.array([
            a[0] for a in entries
        ], dtype=np.float64)

        vals = np.array([
            a[1] for a in entries
        ], dtype=np.float64)

        failure_times = {
            a[2] for a in entries
        }

        if len(failure_times) != 1:
            raise ValueError(
                f"Inconsistent failure_time_s "
                f"within run {rid}"
            )

        failure_time = next(
            iter(failure_times)
        )

        if len(entries) < INITIAL_SUMMARIES:
            continue

        for i in range(
            INITIAL_SUMMARIES - 1,
            len(entries),
            INFERENCE_PERIOD_S // SUMMARY_PERIOD_S
        ):
            if (
                failure_time is not None
                and times[i] >= failure_time
            ):
                break

            elapsed = (
                times[i] - times[i - 5]
            )

            if abs(
                elapsed - TREND_WINDOW_S
            ) > 0.01:
                raise ValueError(
                    f"Irregular sampling in run {rid}: "
                    f"{elapsed}"
                )

            trend = (
                vals[i] - vals[i - 5]
            ) / elapsed

            x = np.r_[
                vals[i],
                trend
            ]

            y = int(
                failure_time is not None
                and 0 < failure_time - times[i] <= HORIZON_S
            )

            runs.append((
                rid,
                entries[i][3],
                times[i],
                x,
                y,
                failure_time
            ))

    if not runs:
        raise ValueError(
            "No usable 6-summary windows found"
        )

    return runs


def run_train(args):

    csv_path = (
        Path(args.csv).resolve()
        if args.csv
        else ROOT / "machine_data.csv"
    )

    if args.generate or not csv_path.exists():
        generate_csv(
            csv_path,
            args.runs
        )

    rows = read_and_featurize(csv_path)

    x = np.array([
        row[3] for row in rows
    ])

    y = np.array([
        row[4] for row in rows
    ])

    run_ids = np.array([
        row[0] for row in rows
    ])

    unique_ids = np.unique(run_ids)

    train_ids, temp_ids = train_test_split(
        unique_ids,
        test_size=0.30,
        random_state=22
    )

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.5,
        random_state=23
    )

    train_idx = np.isin(
        run_ids, train_ids
    )

    val_idx = np.isin(
        run_ids, val_ids
    )

    test_idx = np.isin(
        run_ids, test_ids
    )

    if len(
        np.unique(y[train_idx])
    ) != 2:
        raise ValueError(
            "Training must include both 0 and 1 labels"
        )

    scaler = StandardScaler().fit(
        x[train_idx]
    )

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=44
    )

    model.fit(
        scaler.transform(x[train_idx]),
        y[train_idx]
    )

    def evaluate(mask):

        scores = model.predict_proba(
            scaler.transform(x[mask])
        )[:, 1]

        truth = y[mask]

        pred = (
            scores >= ALERT_THRESHOLD
        )

        result = {
            "precision": float(
                precision_score(
                    truth,
                    pred,
                    zero_division=0
                )
            ),

            "recall": float(
                recall_score(
                    truth,
                    pred,
                    zero_division=0
                )
            ),

            "f1": float(
                f1_score(
                    truth,
                    pred,
                    zero_division=0
                )
            ),

            "confusion_matrix_TN_FP_FN_TP":
                confusion_matrix(
                    truth,
                    pred,
                    labels=[0, 1]
                ).ravel().tolist(),

            "roc_auc": (
                float(
                    roc_auc_score(
                        truth,
                        scores
                    )
                )
                if len(np.unique(truth)) == 2
                else None
            ),

            "samples": int(
                len(truth)
            )
        }

        return result

    results = {
        "dataset":
            "SYNTHETIC"
            if args.generate
            else "User-supplied CSV (provenance not verified)",

        "threshold": ALERT_THRESHOLD,
        "horizon_seconds": HORIZON_S,

        "training": evaluate(train_idx),
        "validation": evaluate(val_idx),
        "test": evaluate(test_idx)
    }

    scenarios = np.array([
        r[1] for r in rows
    ])

    results["test_by_scenario"] = {}

    for scenario in SCENARIOS:

        subset = (
            test_idx &
            (scenarios == scenario)
        )

        if subset.sum():

            metric = evaluate(subset)

            results[
                "test_by_scenario"
            ][scenario] = {
                "samples": metric["samples"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "false_positives": int(
                    metric[
                        "confusion_matrix_TN_FP_FN_TP"
                    ][1]
                )
            }

    print(json.dumps(
        results,
        indent=2
    ))

    def format_cpp(values):

        return ", ".join(
            f"{float(z):.10g}f"
            if float(z) != 0
            else "0.0f"
            for z in values
        )

    header = f'''#pragma once
#include <math.h>

namespace PredictEdgeML {{

static constexpr int FEATURES = 6;
static constexpr float ALERT_THRESHOLD = 0.80f;
static constexpr float RESET_THRESHOLD = 0.60f;

static constexpr unsigned long SUMMARY_PERIOD_SECONDS = 100UL;
static constexpr unsigned long TREND_WINDOW_SECONDS = 500UL;
static constexpr unsigned long INFERENCE_PERIOD_SECONDS = 300UL;
static constexpr unsigned long FAILURE_HORIZON_SECONDS = 3600UL;

static const float MEAN[FEATURES] = {{{format_cpp(scaler.mean_)}}};
static const float SCALE[FEATURES] = {{{format_cpp(scaler.scale_)}}};
static const float WEIGHT[FEATURES] = {{{format_cpp(model.coef_[0])}}};
static const float BIAS = {float(model.intercept_[0]):.10g}f;

inline float predictRisk(const float *x) {{
  float z = BIAS;

  for (int i = 0; i < FEATURES; ++i) {{
    const float normalized =
      (x[i] - MEAN[i]) / SCALE[i];

    z += WEIGHT[i] * normalized;
  }}

  z = fmaxf(-35.0f, fminf(35.0f, z));
  return 1.0f / (1.0f + expf(-z));
}}

}}
'''

    (ROOT / "PredictEdgeModel.h").write_text(
        header,
        encoding="utf-8"
    )

    (
        ROOT / "training_metrics.json"
    ).write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    test_samples = x[test_idx][:15]

    test_scores = model.predict_proba(
        scaler.transform(test_samples)
    )[:, 1]

    with (
        ROOT / "model_parity_test.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            FEATURE_NAMES + ["expected_score"]
        )

        writer.writerows([
            list(v) + [float(s)]
            for v, s in zip(
                test_samples,
                test_scores
            )
        ])

    print(
        "Generated PredictEdgeModel.h, "
        "training_metrics.json and "
        "model_parity_test.csv"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default=None,
        help="CSV of machine runs"
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate synthetic dataset"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=400
    )

    run_train(
        parser.parse_args()
    )
