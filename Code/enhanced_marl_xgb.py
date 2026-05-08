from __future__ import annotations

import argparse
import json
import math
import random
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


RANDOM_SEED = 42
DATASET_CONFIG = {
    "unsw": {
        "train_zip": "unsw.zip",
        "train_member": "unsw/UNSW_NB15_training-set.csv",
        "test_zip": "unsw.zip",
        "test_member": "unsw/UNSW_NB15_testing-set.csv",
        "label": "label",
        "drop": ["id", "attack_cat"],
    },
    "ton": {
        "single_zip": "ton.zip",
        "single_member": "ton/ion_iot_train_test.csv",
        "label_candidates": ["label", "class", "type"],
        "drop": [],
    },
    "cic": {
        "zip_glob": "cic/*.zip",
        "label_candidates": ["label", "class", "type"],
        "drop": [],
    },
}


@dataclass
class ExperimentResult:
    dataset: str
    method: str
    alpha: float
    agents: int
    features: int
    rows_train: int
    rows_test: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    threshold: float


def _read_csv_from_zip(zip_path: Path, member: str | None = None, **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        target = member
        if target is None:
            target = next(
                name
                for name in archive.namelist()
                if name.endswith(".csv") and not name.startswith("__MACOSX/")
            )
        with archive.open(target) as handle:
            return pd.read_csv(handle, **kwargs)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _find_label(df: pd.DataFrame, candidates: list[str]) -> str:
    lowered = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for col in df.columns:
        lowered_col = col.lower()
        if "label" in lowered_col or "class" in lowered_col or "type" in lowered_col:
            return col
    raise ValueError(f"No label column found. Available columns: {list(df.columns)}")


def _binary_labels(series: pd.Series) -> np.ndarray:
    values = series.astype(str).str.strip().str.lower()
    benign = {"0", "normal", "benign"}
    return values.apply(lambda value: 0 if value in benign or "benign" in value else 1).astype(int).to_numpy()


def _sample_balanced(df: pd.DataFrame, label_col: str, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    parts = []
    per_class = max(1, max_rows // max(2, df[label_col].nunique()))
    for _, group in df.groupby(label_col):
        parts.append(group.sample(n=min(len(group), per_class), random_state=seed))
    sampled = pd.concat(parts, ignore_index=True)
    remaining = max_rows - len(sampled)
    if remaining > 0:
        rest = df.drop(sampled.index, errors="ignore")
        if len(rest):
            sampled = pd.concat(
                [sampled, rest.sample(n=min(remaining, len(rest)), random_state=seed)],
                ignore_index=True,
            )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all")
    return df.dropna(axis=0, how="any")


def load_dataset(name: str, root: Path, max_rows: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    config = DATASET_CONFIG[name]
    if name == "unsw":
        train = _clean_frame(_read_csv_from_zip(root / config["train_zip"], config["train_member"]))
        test = _clean_frame(_read_csv_from_zip(root / config["test_zip"], config["test_member"]))
        label_col = config["label"]
        train = train[train[label_col].isin([0, 1])]
        test = test[test[label_col].isin([0, 1])]
        train = _sample_balanced(train, label_col, max_rows, RANDOM_SEED)
        test = _sample_balanced(test, label_col, max(1000, max_rows // 3), RANDOM_SEED + 1)
    elif name == "ton":
        df = _clean_frame(_read_csv_from_zip(root / config["single_zip"], config["single_member"]))
        label_col = _find_label(df, config["label_candidates"])
        df[label_col] = _binary_labels(df[label_col])
        df = _sample_balanced(df, label_col, max_rows, RANDOM_SEED)
        train, test = train_test_split(df, test_size=0.3, stratify=df[label_col], random_state=RANDOM_SEED)
    elif name == "cic":
        frames = []
        for zip_path in sorted(root.glob(config["zip_glob"])):
            frames.append(_clean_frame(_read_csv_from_zip(zip_path, low_memory=False)))
        df = pd.concat(frames, ignore_index=True)
        label_col = _find_label(df, config["label_candidates"])
        df[label_col] = _binary_labels(df[label_col])
        df = _sample_balanced(df, label_col, max_rows, RANDOM_SEED)
        train, test = train_test_split(df, test_size=0.3, stratify=df[label_col], random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    y_train = train[label_col].astype(int).reset_index(drop=True)
    y_test = test[label_col].astype(int).reset_index(drop=True)
    drop_cols = [label_col, *config.get("drop", [])]
    x_train = train.drop(columns=[col for col in drop_cols if col in train.columns])
    x_test = test.drop(columns=[col for col in drop_cols if col in test.columns])
    return x_train, y_train, x_test, y_test


def preprocess(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    train = x_train.copy()
    test = x_test.copy()
    common = [col for col in train.columns if col in test.columns]
    train = train[common]
    test = test[common]

    categorical = train.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    numeric = [col for col in train.columns if col not in categorical]

    encoded_train_parts = []
    encoded_test_parts = []
    if numeric:
        encoded_train_parts.append(train[numeric].apply(pd.to_numeric, errors="coerce").reset_index(drop=True))
        encoded_test_parts.append(test[numeric].apply(pd.to_numeric, errors="coerce").reset_index(drop=True))
    if categorical:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoded_train_parts.append(
            pd.DataFrame(encoder.fit_transform(train[categorical].astype(str)), columns=categorical)
        )
        encoded_test_parts.append(
            pd.DataFrame(encoder.transform(test[categorical].astype(str)), columns=categorical)
        )

    train_encoded = pd.concat(encoded_train_parts, axis=1).fillna(0.0)
    test_encoded = pd.concat(encoded_test_parts, axis=1).fillna(0.0)

    scaler = StandardScaler()
    return scaler.fit_transform(train_encoded), scaler.transform(test_encoded)


def partition_dirichlet(y: np.ndarray, agents: int, alpha: float, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    partitions: list[list[int]] = [[] for _ in range(agents)]
    for label in np.unique(y):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)
        proportions = rng.dirichlet([alpha] * agents)
        splits = np.split(indices, (np.cumsum(proportions) * len(indices)).astype(int)[:-1])
        for agent_idx, split in enumerate(splits):
            partitions[agent_idx].extend(split.tolist())
    valid = [np.array(sorted(idx), dtype=int) for idx in partitions if len(np.unique(y[idx])) >= 2]
    if len(valid) < 2:
        raise ValueError("Dirichlet split produced fewer than two valid agents")
    return valid


def select_features(x: np.ndarray, y: np.ndarray, max_features: int, seed: int) -> list[int]:
    scores = mutual_info_classif(x, y, random_state=seed, discrete_features=False)
    count = min(max_features, x.shape[1])
    return sorted(np.argsort(scores)[-count:].tolist())


def make_model(y: np.ndarray, seed: int) -> xgb.XGBClassifier:
    positives = max(1, int(y.sum()))
    negatives = max(1, int(len(y) - positives))
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=120,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        min_child_weight=2,
        scale_pos_weight=negatives / positives,
        random_state=seed,
        n_jobs=1,
    )


def optimize_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.2, 0.8, 31):
        current = f1_score(y_true, probabilities >= threshold, zero_division=0)
        if current > best_f1:
            best_f1 = current
            best_threshold = float(threshold)
    return best_threshold


def score_predictions(
    dataset: str,
    method: str,
    alpha: float,
    agents: int,
    features: int,
    rows_train: int,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> ExperimentResult:
    predictions = (probabilities >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        auc = math.nan
    return ExperimentResult(
        dataset=dataset,
        method=method,
        alpha=alpha,
        agents=agents,
        features=features,
        rows_train=rows_train,
        rows_test=len(y_true),
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        roc_auc=auc,
        threshold=threshold,
    )


def run_legacy_baseline(
    dataset: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    partitions: list[np.ndarray],
    alpha: float,
    feature_count: int,
) -> ExperimentResult:
    rng = random.Random(RANDOM_SEED)
    features = sorted(rng.sample(range(x_train.shape[1]), min(feature_count, x_train.shape[1])))
    first = partitions[0]
    model = make_model(y_train[first], RANDOM_SEED)
    model.fit(x_train[first][:, features], y_train[first])
    probabilities = model.predict_proba(x_test[:, features])[:, 1]
    return score_predictions(
        dataset,
        "legacy_random_first_agent",
        alpha,
        len(partitions),
        len(features),
        len(y_train),
        y_test,
        probabilities,
        0.5,
    )


def run_enhanced(
    dataset: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_calibration: np.ndarray,
    y_calibration: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    partitions: list[np.ndarray],
    alpha: float,
    feature_count: int,
) -> ExperimentResult:
    rng = random.Random(RANDOM_SEED)
    feature_sets = {
        "mutual_info": select_features(x_train, y_train, feature_count, RANDOM_SEED),
        "epsilon_random": sorted(rng.sample(range(x_train.shape[1]), min(feature_count, x_train.shape[1]))),
    }
    candidates = []
    selected_feature_count = feature_count

    for policy_name, selected in feature_sets.items():
        calibration_probabilities = []
        test_probabilities = []
        trust_scores = []

        for agent_id, indices in enumerate(partitions):
            if len(indices) < 20:
                continue
            y_local = y_train[indices]
            if len(np.unique(y_local)) < 2:
                continue
            train_idx, val_idx = train_test_split(
                indices,
                test_size=0.25,
                stratify=y_train[indices],
                random_state=RANDOM_SEED + agent_id,
            )
            model = make_model(y_train[train_idx], RANDOM_SEED + agent_id)
            model.fit(x_train[train_idx][:, selected], y_train[train_idx])
            val_prob = model.predict_proba(x_train[val_idx][:, selected])[:, 1]
            local_threshold = optimize_threshold(y_train[val_idx], val_prob)
            val_pred = val_prob >= local_threshold
            local_f1 = f1_score(y_train[val_idx], val_pred, zero_division=0)
            trust = max(0.05, local_f1)
            calibration_probabilities.append(model.predict_proba(x_calibration[:, selected])[:, 1])
            test_probabilities.append(model.predict_proba(x_test[:, selected])[:, 1])
            trust_scores.append(trust)

        if not test_probabilities:
            continue

        weights = np.asarray(trust_scores, dtype=float)
        weights = weights / weights.sum()
        ensemble_calibration = np.average(np.vstack(calibration_probabilities), axis=0, weights=weights)
        ensemble_test = np.average(np.vstack(test_probabilities), axis=0, weights=weights)
        candidates.append((f"{policy_name}_trust_ensemble", ensemble_calibration, ensemble_test, len(selected)))
        for agent_id, (calibration_prob, test_prob) in enumerate(
            zip(calibration_probabilities, test_probabilities, strict=False)
        ):
            candidates.append((f"{policy_name}_agent_{agent_id}", calibration_prob, test_prob, len(selected)))

    if not candidates:
        raise ValueError("No enhanced agent produced a valid model")

    best_name = "trust_ensemble"
    best_threshold = 0.5
    best_score = -1.0
    best_test = ensemble_test
    for name, calibration_prob, test_prob, candidate_feature_count in candidates:
        candidate_threshold = optimize_threshold(y_calibration, calibration_prob)
        candidate_score = f1_score(
            y_calibration,
            calibration_prob >= candidate_threshold,
            zero_division=0,
        )
        if candidate_score > best_score:
            best_name = name
            best_threshold = candidate_threshold
            best_score = candidate_score
            best_test = test_prob
            selected_feature_count = candidate_feature_count

    threshold = best_threshold
    method = (
        "enhanced_trust_weighted_ensemble"
        if best_name.endswith("trust_ensemble")
        else "enhanced_adaptive_best_peer"
    )
    return score_predictions(
        dataset,
        method,
        alpha,
        len(partitions),
        selected_feature_count,
        len(y_train),
        y_test,
        best_test,
        threshold,
    )


def run_dataset(dataset: str, root: Path, args: argparse.Namespace) -> list[ExperimentResult]:
    x_train_df, y_train_s, x_test_df, y_test_s = load_dataset(dataset, root, args.max_rows)
    x_train, x_test = preprocess(x_train_df, x_test_df)
    y_train = y_train_s.to_numpy(dtype=int)
    y_test = y_test_s.to_numpy(dtype=int)
    train_idx, calibration_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=0.2,
        stratify=y_train,
        random_state=RANDOM_SEED,
    )
    x_model_train = x_train[train_idx]
    y_model_train = y_train[train_idx]
    x_calibration = x_train[calibration_idx]
    y_calibration = y_train[calibration_idx]
    partitions = partition_dirichlet(y_model_train, args.agents, args.alpha, RANDOM_SEED)
    feature_count = min(args.features, max(2, x_train.shape[1]))
    return [
        run_legacy_baseline(
            dataset,
            x_model_train,
            y_model_train,
            x_test,
            y_test,
            partitions,
            args.alpha,
            feature_count,
        ),
        run_enhanced(
            dataset,
            x_model_train,
            y_model_train,
            x_calibration,
            y_calibration,
            x_test,
            y_test,
            partitions,
            args.alpha,
            feature_count,
        ),
    ]


def summarize(results: list[ExperimentResult]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(result) for result in results])
    baseline = df[df["method"] == "legacy_random_first_agent"].set_index("dataset")
    enhanced = df[df["method"].str.startswith("enhanced_")].set_index("dataset")
    deltas = enhanced[["accuracy", "precision", "recall", "f1", "roc_auc"]] - baseline[
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ]
    deltas = deltas.add_prefix("delta_").reset_index()
    return df.merge(deltas, on="dataset", how="left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced MARL-XGBoost comparison runner")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository/data root")
    parser.add_argument("--datasets", nargs="+", default=["unsw", "ton", "cic"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--features", type=int, default=24)
    parser.add_argument("--max-rows", type=int, default=12000)
    parser.add_argument("--output", type=Path, default=Path("results/enhanced/enhanced_comparison.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/enhanced/enhanced_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    results: list[ExperimentResult] = []
    for dataset in args.datasets:
        print(f"[INFO] Running {dataset} with alpha={args.alpha}, agents={args.agents}")
        results.extend(run_dataset(dataset, args.root, args))
    summary = summarize(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    payload = {
        "config": {
            "datasets": args.datasets,
            "alpha": args.alpha,
            "agents": args.agents,
            "features": args.features,
            "max_rows": args.max_rows,
        },
        "results": json.loads(summary.to_json(orient="records")),
    }
    args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"[INFO] Wrote {args.output}")
    print(f"[INFO] Wrote {args.summary}")


if __name__ == "__main__":
    main()
