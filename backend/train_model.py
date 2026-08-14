"""
Phase 3: 모델 학습 스크립트

정규화된 DB(plate/product_spec/furnace_process/rolling_process/quality_inspection)에서
학습 데이터를 뽑아 6개 후보(로지스틱회귀/의사결정트리/랜덤포레스트/XGBoost/LightGBM/SVM)를
각각 GridSearchCV(5-fold, train 700행 내부에서만 탐색)로 하이퍼파라미터를 튜닝한 뒤,
시간 기준 검증셋(150행) ROC-AUC가 가장 높은 모델을 model_version에 등록한다.
해당 name의 첫 등록이면 자동으로 활성화하고, 이후 재학습(retrain.py)은 비활성 상태로
등록해 /models/{id}/activate로 수동 전환하게 한다.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949)에서 em-dash 등 출력 깨짐 방지

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import select, text
from sqlalchemy.orm import Session
from xgboost import XGBClassifier

import features
from db import engine, init_db
from models import ModelVersion

MODEL_NAME = "scale_defect_classifier"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

QUERY = """
WITH latest_furnace AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY plate_no ORDER BY recorded_at DESC) AS rn
    FROM furnace_process
),
latest_inspection AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY plate_no ORDER BY inspected_at DESC) AS rn
    FROM quality_inspection
)
SELECT
    p.plate_no, p.pt_thick, p.pt_width, p.pt_length,
    ps.spec_country, ps.steel_kind,
    f.fur_no, f.fur_input_row, f.fur_heat_temp, f.fur_heat_time,
    f.fur_soak_temp, f.fur_soak_time, f.fur_total_time,
    r.hsb_applied, r.rolling_method, r.rolling_temp, r.descaling_count,
    r.work_group, r.rolling_date,
    qi.defect_type
FROM plate p
JOIN product_spec ps ON ps.spec_code = p.spec_code
JOIN latest_furnace f ON f.plate_no = p.plate_no AND f.rn = 1
JOIN rolling_process r ON r.plate_no = p.plate_no
JOIN latest_inspection qi ON qi.plate_no = p.plate_no AND qi.rn = 1
"""


def load_training_frame() -> pd.DataFrame:
    df = pd.read_sql(text(QUERY), engine, parse_dates=["rolling_date"])
    if df["hsb_applied"].dtype != bool:
        df["hsb_applied"] = df["hsb_applied"].astype(bool)
    return df


def impute_rolling_temp(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, float]:
    df = df.copy()
    global_median = float(df["rolling_temp"].median())
    group_median = df.groupby(["steel_kind", "rolling_method"])["rolling_temp"].median()
    group_median_map = {f"{k[0]}|{k[1]}": float(v) for k, v in group_median.items()}

    def fill(row):
        if pd.notna(row["rolling_temp"]) and row["rolling_temp"] > 0:
            return row["rolling_temp"]
        key = f"{row['steel_kind']}|{row['rolling_method']}"
        return group_median_map.get(key, global_median)

    mask = df["rolling_temp"].isna() | (df["rolling_temp"] <= 0)
    df.loc[mask, "rolling_temp"] = df.loc[mask].apply(fill, axis=1)
    return df, group_median_map, global_median


def build_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    engineered = df.apply(lambda row: pd.Series(features.engineer_row(row.to_dict())), axis=1)
    y = (df["defect_type"] == "Scale").astype(int)

    X = pd.get_dummies(
        engineered, columns=features.CATEGORICAL_FEATURES, drop_first=True
    )
    # get_dummies가 만드는 bool 컬럼 + 원본 수치 컬럼이 섞이면 프레임 전체가
    # object dtype으로 잡히는 경우가 있어(pandas 3.x), 학습/서빙 모두 float64로 고정한다.
    X = X.astype("float64")
    return X, y, list(X.columns)


def time_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, valid_size=0.15, test_size=0.15):
    order = df["rolling_date"].sort_values().index
    n = len(order)
    n_test = int(n * test_size)
    n_valid = int(n * valid_size)
    n_train = n - n_valid - n_test

    train_idx = order[:n_train]
    valid_idx = order[n_train:n_train + n_valid]
    test_idx = order[n_train + n_valid:]
    return (
        X.loc[train_idx], y.loc[train_idx],
        X.loc[valid_idx], y.loc[valid_idx],
        X.loc[test_idx], y.loc[test_idx],
    )


def evaluate(model, X, y) -> dict:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y, prob)), 4) if y.nunique() > 1 else None,
    }


def _build_search_space(y_train) -> dict[str, tuple]:
    """(estimator, param_grid) 쌍. param_grid 키는 GridSearchCV가 그대로 받는 sklearn 파라미터명."""
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    xgb_scale_pos_weight = neg / pos if pos else 1.0

    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"C": [0.01, 0.1, 1, 10]},
        ),
        "DecisionTree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=42),
            {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 5, 10]},
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {"n_estimators": [100, 300], "max_depth": [5, 8, None]},
        ),
        "XGBoost": (
            XGBClassifier(
                eval_metric="logloss", random_state=42, scale_pos_weight=xgb_scale_pos_weight
            ),
            {"n_estimators": [100, 300], "max_depth": [3, 5, 8], "learning_rate": [0.05, 0.1]},
        ),
        "LightGBM": (
            LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1),
            {"n_estimators": [100, 300], "max_depth": [3, 5, -1], "learning_rate": [0.05, 0.1]},
        ),
        # SVM은 스케일에 민감해 StandardScaler와 파이프라인으로 묶는다 (트리 계열은 스케일 불필요).
        "SVM": (
            Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=True, class_weight="balanced", random_state=42))]),
            {"svc__C": [0.1, 1, 10], "svc__kernel": ["rbf", "linear"]},
        ),
    }


def train_and_compare(X_train, y_train, X_valid, y_valid):
    search_space = _build_search_space(y_train)
    comparison = {}
    fitted = {}

    for name, (estimator, param_grid) in search_space.items():
        search = GridSearchCV(estimator, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        metrics = evaluate(best, X_valid, y_valid)
        metrics["best_params"] = search.best_params_
        metrics["cv_best_auc"] = round(float(search.best_score_), 4)
        comparison[name] = metrics
        fitted[name] = best

        print(f"  [{name}] best_params={search.best_params_} cv_auc={search.best_score_:.4f} valid_auc={metrics['auc']}")

    best_name = max(comparison, key=lambda n: (comparison[n]["auc"] or 0))
    return best_name, fitted, comparison


def register_model_version(
    session: Session,
    algorithm: str,
    version: str,
    metrics: dict,
    train_row_count: int,
    artifact_path: str,
    is_active: bool,
) -> ModelVersion:
    row = ModelVersion(
        name=MODEL_NAME,
        version=version,
        algorithm=algorithm,
        trained_at=datetime.now(timezone.utc),
        train_row_count=train_row_count,
        metrics=metrics,
        threshold=0.5,
        artifact_path=artifact_path,
        is_active=is_active,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def main() -> list[dict]:
    """
    후보 6종을 전부 학습·등록한다(대시보드의 "모델 관리"에서 전부 골라 쓸 수 있어야
    하므로, 예전처럼 1등만 저장하지 않는다). 이 이름(name)에 활성 모델이 하나도 없는
    최초 실행(bootstrap)일 때만 valid AUC 1위를 자동 활성화하고, 그 외에는 전부
    비활성으로 등록해 사람이 대시보드에서 직접 고르게 한다.

    반환: [{id, algorithm, version, is_active}, ...] — 방금 등록한 항목들.
    """
    init_db()
    ARTIFACT_DIR.mkdir(exist_ok=True)

    df = load_training_frame()
    print(f"[데이터] 학습용 레코드 {len(df)}건 로드")
    if len(df) < 50:
        raise SystemExit("학습 데이터가 너무 적습니다 (Phase 1 마이그레이션을 먼저 실행하세요)")

    df, group_median_map, global_median = impute_rolling_temp(df)
    X, y, model_columns = build_design_matrix(df)
    X_train, y_train, X_valid, y_valid, X_test, y_test = time_split(df, X, y)
    print(f"[분할] train={len(X_train)} valid={len(X_valid)} test={len(X_test)}")

    best_name, fitted, comparison = train_and_compare(X_train, y_train, X_valid, y_valid)

    print("[비교] valid set 성능")
    for name, m in comparison.items():
        marker = " <= 최고 AUC" if name == best_name else ""
        print(f"  {name:16s} precision={m['precision']} recall={m['recall']} f1={m['f1']} auc={m['auc']}{marker}")

    background_sample = X_train.sample(n=min(100, len(X_train)), random_state=42)
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    registered: list[dict] = []
    with Session(engine) as session:
        is_bootstrap = (
            session.scalar(
                select(ModelVersion).where(ModelVersion.name == MODEL_NAME, ModelVersion.is_active.is_(True))
            )
            is None
        )

        for name, model in fitted.items():
            test_metrics = evaluate(model, X_test, y_test)
            bundle = {
                "model": model,
                "algorithm": name,
                "model_columns": model_columns,
                "rolling_temp_group_median": group_median_map,
                "rolling_temp_global_median": global_median,
                "background_sample": background_sample,
            }
            artifact_path = ARTIFACT_DIR / f"{MODEL_NAME}_{name}_{run_tag}.joblib"
            joblib.dump(bundle, artifact_path)

            is_active = is_bootstrap and name == best_name
            row = register_model_version(
                session,
                algorithm=name,
                version=f"{name}-{run_tag}",
                metrics={"valid": comparison[name], "test": test_metrics},
                train_row_count=len(X_train),
                artifact_path=str(artifact_path),
                is_active=is_active,
            )
            registered.append(
                {"id": row.id, "algorithm": row.algorithm, "version": row.version, "is_active": row.is_active}
            )
            marker = " -> 활성화됨(최초 등록)" if is_active else ""
            print(f"[등록] {name} {row.version} test_auc={test_metrics['auc']}{marker}")

    print(f"[완료] {len(registered)}개 모델 버전 등록. 대시보드 '모델 관리'에서 원하는 걸 활성화하세요.")
    return registered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()
