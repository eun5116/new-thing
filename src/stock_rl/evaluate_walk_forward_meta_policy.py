from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stock_rl.config import project_path
from stock_rl.train_monthly_meta_policy import (
    CLASSES,
    FEATURES,
    _add_intercept,
    _prepare_xy,
    _returns_matrix,
    _strategy_indices,
    evaluate_baselines,
    evaluate_weights,
    train_label_softmax,
    train_return_softmax,
)


def _load_monthly(config_path: str, split: str, out_dir: Path) -> pd.DataFrame:
    return pd.read_csv(out_dir / f"meta_policy_monthly_strategy_returns_{split}.csv")


def _candidate_specs() -> list[dict[str, float | str]]:
    specs: list[dict[str, float | str]] = []
    for l2 in [0.0, 0.001, 0.01, 0.05, 0.1, 0.25]:
        specs.append({"model_type": "label", "l2": l2, "entropy_coef": 0.0})
    for l2 in [0.0, 0.001, 0.01, 0.05, 0.1, 0.25]:
        for entropy_coef in [0.0, 0.001, 0.01, 0.05]:
            specs.append({"model_type": "return", "l2": l2, "entropy_coef": entropy_coef})
    return specs


def _model_name(spec: dict[str, float | str]) -> str:
    if spec["model_type"] == "label":
        return f"wf_label_softmax_l2_{spec['l2']:g}"
    return f"wf_return_softmax_l2_{spec['l2']:g}_ent_{spec['entropy_coef']:g}"


def _fit_weights(train: pd.DataFrame, x_train: np.ndarray, spec: dict[str, float | str]) -> np.ndarray:
    if spec["model_type"] == "label":
        return train_label_softmax(x_train, _strategy_indices(train), l2=float(spec["l2"]))
    return train_return_softmax(
        x_train,
        _returns_matrix(train),
        l2=float(spec["l2"]),
        entropy_coef=float(spec["entropy_coef"]),
    )


def _build_design(train: pd.DataFrame, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_train, standardizer = _prepare_xy(train, train)
    values = frame[FEATURES].fillna(0.0).to_numpy(dtype=np.float64)
    x_frame = _add_intercept(standardizer.transform(values))
    return x_train, x_frame


def evaluate(config_path: str, out_dir: str | None = None) -> dict[str, Path]:
    output_dir = Path(out_dir) if out_dir else project_path(config_path, "reports")
    train = _load_monthly(config_path, "train", output_dir)
    valid = _load_monthly(config_path, "valid", output_dir)
    test = _load_monthly(config_path, "test", output_dir)

    x_train, x_valid = _build_design(train, valid)
    _, x_test = _build_design(train, test)
    train_valid = pd.concat([train, valid], ignore_index=True)

    valid_rows = []
    test_rows = []
    prediction_frames = []
    weights_by_model: dict[str, np.ndarray] = {}
    for spec in _candidate_specs():
        name = _model_name(spec)
        weights = _fit_weights(train, x_train, spec)
        weights_by_model[name] = weights
        valid_summary, valid_predictions = evaluate_weights(valid, x_valid, weights, "valid", name)
        test_summary, _ = evaluate_weights(test, x_test, weights, "test", name)
        valid_rows.append(valid_summary | {"trained_on": "train"})
        test_rows.append(test_summary | {"trained_on": "train"})
        prediction_frames.append(valid_predictions)

    valid_candidates = pd.DataFrame(valid_rows).sort_values("avg_monthly_return", ascending=False)
    test_candidates = pd.DataFrame(test_rows).sort_values("avg_monthly_return", ascending=False)
    selected_model = str(valid_candidates.iloc[0]["model"])
    selected_spec = next(spec for spec in _candidate_specs() if _model_name(spec) == selected_model)
    selected_weights = weights_by_model[selected_model]
    selected_valid_summary, selected_valid_predictions = evaluate_weights(
        valid, x_valid, selected_weights, "valid", selected_model
    )
    selected_test_summary, selected_test_predictions = evaluate_weights(test, x_test, selected_weights, "test", selected_model)

    selected_summary = pd.concat(
        [
            pd.DataFrame(
                [
                    selected_valid_summary | {"trained_on": "train"},
                    selected_test_summary | {"trained_on": "train"},
                ]
            ),
            evaluate_baselines(valid, "valid"),
            evaluate_baselines(test, "test"),
        ],
        ignore_index=True,
    )
    selected_predictions = pd.concat([selected_valid_predictions, selected_test_predictions], ignore_index=True)

    x_train_valid, x_test_refit = _build_design(train_valid, test)
    refit_weights = _fit_weights(train_valid, x_train_valid, selected_spec)
    refit_test_summary, refit_test_predictions = evaluate_weights(
        test, x_test_refit, refit_weights, "test", f"{selected_model}_refit_train_valid"
    )
    selected_summary = pd.concat(
        [selected_summary, pd.DataFrame([refit_test_summary | {"trained_on": "train_valid"}])],
        ignore_index=True,
    )
    selected_predictions = pd.concat([selected_predictions, refit_test_predictions], ignore_index=True)

    paths = {
        "valid_candidates": output_dir / "meta_policy_walk_forward_candidate_valid.csv",
        "test_candidates": output_dir / "meta_policy_walk_forward_candidate_test.csv",
        "selected_summary": output_dir / "meta_policy_walk_forward_selected_summary.csv",
        "selected_predictions": output_dir / "meta_policy_walk_forward_selected_predictions.csv",
    }
    valid_candidates.to_csv(paths["valid_candidates"], index=False)
    test_candidates.to_csv(paths["test_candidates"], index=False)
    selected_summary.to_csv(paths["selected_summary"], index=False)
    selected_predictions.to_csv(paths["selected_predictions"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E028_liquid48_target_hybrid_aggressive.yaml")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    for name, path in evaluate(args.config, args.out_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
