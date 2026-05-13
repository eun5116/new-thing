from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stock_rl.analyze_meta_policy_inputs import FEATURE_COLUMNS, STRATEGIES
from stock_rl.config import project_path


FEATURES = [f"start_{column}" for column in FEATURE_COLUMNS]
CLASSES = STRATEGIES


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _add_intercept(values: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(values)), values])


def _prepare_xy(train: pd.DataFrame, frame: pd.DataFrame) -> tuple[np.ndarray, Standardizer]:
    train_values = train[FEATURES].fillna(0.0).to_numpy(dtype=np.float64)
    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardizer = Standardizer(mean=mean, scale=scale)
    values = frame[FEATURES].fillna(0.0).to_numpy(dtype=np.float64)
    return _add_intercept(standardizer.transform(values)), standardizer


def _strategy_indices(frame: pd.DataFrame) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(CLASSES)}
    return frame["best_strategy"].map(mapping).to_numpy(dtype=np.int64)


def _returns_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[CLASSES].to_numpy(dtype=np.float64)


def train_label_softmax(
    x: np.ndarray,
    y: np.ndarray,
    l2: float,
    learning_rate: float = 0.05,
    epochs: int = 4000,
) -> np.ndarray:
    weights = np.zeros((x.shape[1], len(CLASSES)), dtype=np.float64)
    one_hot = np.eye(len(CLASSES))[y]
    for _ in range(epochs):
        probs = _softmax(x @ weights)
        grad = x.T @ (probs - one_hot) / len(x)
        grad[1:] += l2 * weights[1:]
        weights -= learning_rate * grad
    return weights


def train_return_softmax(
    x: np.ndarray,
    returns: np.ndarray,
    l2: float,
    entropy_coef: float,
    learning_rate: float = 0.03,
    epochs: int = 5000,
) -> np.ndarray:
    weights = np.zeros((x.shape[1], len(CLASSES)), dtype=np.float64)
    for _ in range(epochs):
        probs = _softmax(x @ weights)
        expected = (probs * returns).sum(axis=1, keepdims=True)
        grad_logits = probs * (returns - expected)
        if entropy_coef:
            log_probs = np.log(np.clip(probs, 1e-12, 1.0))
            entropy_grad = -probs * (log_probs + (probs * log_probs).sum(axis=1, keepdims=True))
            grad_logits += entropy_coef * entropy_grad
        grad = x.T @ grad_logits / len(x)
        grad[1:] -= l2 * weights[1:]
        weights += learning_rate * grad
    return weights


def evaluate_weights(frame: pd.DataFrame, x: np.ndarray, weights: np.ndarray, split: str, model: str) -> tuple[dict[str, object], pd.DataFrame]:
    probs = _softmax(x @ weights)
    choices = probs.argmax(axis=1)
    strategy_names = np.array(CLASSES)[choices]
    chosen_returns = np.array([row[strategy] for strategy, (_, row) in zip(strategy_names, frame.iterrows())])
    predictions = frame[["ticker", "month", "best_strategy", "best_return"] + CLASSES].copy()
    predictions["split"] = split
    predictions["model"] = model
    predictions["choice"] = strategy_names
    predictions["chosen_return"] = chosen_returns
    predictions["regret_to_best"] = predictions["chosen_return"] - predictions["best_return"]
    for index, strategy in enumerate(CLASSES):
        predictions[f"prob_{strategy}"] = probs[:, index]
    summary = {
        "split": split,
        "model": model,
        "rows": len(frame),
        "avg_monthly_return": float(predictions["chosen_return"].mean()),
        "avg_regret_to_best": float(predictions["regret_to_best"].mean()),
        "match_best_share": float((predictions["choice"] == predictions["best_strategy"]).mean()),
    }
    for strategy in CLASSES:
        summary[f"{strategy}_choice_share"] = float((predictions["choice"] == strategy).mean())
    return summary, predictions


def evaluate_baselines(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for strategy in CLASSES:
        rows.append(
            {
                "split": split,
                "model": f"always_{strategy}",
                "rows": len(frame),
                "avg_monthly_return": frame[strategy].mean(),
                "avg_regret_to_best": (frame[strategy] - frame["best_return"]).mean(),
                "match_best_share": (frame["best_strategy"] == strategy).mean(),
                **{f"{name}_choice_share": float(name == strategy) for name in CLASSES},
            }
        )
    rows.append(
        {
            "split": split,
            "model": "oracle_best",
            "rows": len(frame),
            "avg_monthly_return": frame["best_return"].mean(),
            "avg_regret_to_best": 0.0,
            "match_best_share": 1.0,
            **{f"{name}_choice_share": (frame["best_strategy"] == name).mean() for name in CLASSES},
        }
    )
    return pd.DataFrame(rows)


def train_and_evaluate(config_path: str, out_dir: str | None = None) -> dict[str, Path]:
    output_dir = Path(out_dir) if out_dir else project_path(config_path, "reports")
    valid = pd.read_csv(output_dir / "meta_policy_monthly_strategy_returns_valid.csv")
    test = pd.read_csv(output_dir / "meta_policy_monthly_strategy_returns_test.csv")
    x_valid, standardizer = _prepare_xy(valid, valid)
    x_test = _add_intercept(standardizer.transform(test[FEATURES].fillna(0.0).to_numpy(dtype=np.float64)))
    y_valid = _strategy_indices(valid)
    returns_valid = _returns_matrix(valid)

    summaries = [evaluate_baselines(valid, "valid"), evaluate_baselines(test, "test")]
    prediction_frames = []
    candidate_rows = []
    test_candidate_rows = []
    weights_by_model: dict[str, np.ndarray] = {}

    for l2 in [0.0, 0.001, 0.01, 0.05, 0.1]:
        name = f"label_softmax_l2_{l2:g}"
        weights = train_label_softmax(x_valid, y_valid, l2=l2)
        weights_by_model[name] = weights
        summary, predictions = evaluate_weights(valid, x_valid, weights, "valid", name)
        test_summary, _ = evaluate_weights(test, x_test, weights, "test", name)
        candidate_rows.append(summary)
        test_candidate_rows.append(test_summary)
        prediction_frames.append(predictions)

    for l2 in [0.0, 0.001, 0.01, 0.05, 0.1]:
        for entropy_coef in [0.0, 0.001, 0.01]:
            name = f"return_softmax_l2_{l2:g}_ent_{entropy_coef:g}"
            weights = train_return_softmax(x_valid, returns_valid, l2=l2, entropy_coef=entropy_coef)
            weights_by_model[name] = weights
            summary, predictions = evaluate_weights(valid, x_valid, weights, "valid", name)
            test_summary, _ = evaluate_weights(test, x_test, weights, "test", name)
            candidate_rows.append(summary)
            test_candidate_rows.append(test_summary)
            prediction_frames.append(predictions)

    valid_candidates = pd.DataFrame(candidate_rows).sort_values("avg_monthly_return", ascending=False)
    test_candidates = pd.DataFrame(test_candidate_rows).sort_values("avg_monthly_return", ascending=False)
    selected_model = str(valid_candidates.iloc[0]["model"])
    selected_weights = weights_by_model[selected_model]
    selected_valid_summary, selected_valid_predictions = evaluate_weights(valid, x_valid, selected_weights, "valid", selected_model)
    selected_test_summary, selected_test_predictions = evaluate_weights(test, x_test, selected_weights, "test", selected_model)

    selected_summary = pd.concat(
        [
            pd.DataFrame([selected_valid_summary, selected_test_summary]),
            evaluate_baselines(valid, "valid"),
            evaluate_baselines(test, "test"),
        ],
        ignore_index=True,
    )
    selected_predictions = pd.concat([selected_valid_predictions, selected_test_predictions], ignore_index=True)

    candidate_path = output_dir / "meta_policy_softmax_candidate_valid.csv"
    selected_summary_path = output_dir / "meta_policy_softmax_selected_summary.csv"
    prediction_path = output_dir / "meta_policy_softmax_selected_predictions.csv"
    test_candidate_path = output_dir / "meta_policy_softmax_candidate_test.csv"
    valid_candidates.to_csv(candidate_path, index=False)
    test_candidates.to_csv(test_candidate_path, index=False)
    selected_summary.to_csv(selected_summary_path, index=False)
    selected_predictions.to_csv(prediction_path, index=False)
    return {
        "valid_candidates": candidate_path,
        "test_candidates": test_candidate_path,
        "selected_summary": selected_summary_path,
        "selected_predictions": prediction_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E028_liquid48_target_hybrid_aggressive.yaml")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    for name, path in train_and_evaluate(args.config, args.out_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
