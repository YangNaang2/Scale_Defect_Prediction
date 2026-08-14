"""
Phase 3: 실시간 예측 서비스

활성(is_active=True) 모델 아티팩트를 로드해 코일 단위로 추론하고, SHAP로 근거를
계산해 prediction / shap_explanation 테이블에 저장한다. 활성 모델이 없으면(Phase 2
단계처럼 아직 학습 전이면) 조용히 None을 반환해 Ingestion API가 정상 동작하게 한다.
"""

from datetime import date, datetime, timezone

import joblib
import pandas as pd
import shap
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import features
from models import Alert, FurnaceProcess, ModelVersion, Plate, Prediction, ProductSpec, RollingProcess, ShapExplanation
from rules import evaluate_rules_from_raw
from schemas import AlertOut, PredictionOut, ShapFeatureOut

MODEL_NAME = "scale_defect_classifier"
TOP_N_FEATURES = 5

_bundle_cache: dict[str, dict] = {}
_explainer_cache: dict[str, "shap.Explainer"] = {}


def _get_active_model_version(session: Session) -> ModelVersion | None:
    return session.scalar(
        select(ModelVersion).where(ModelVersion.name == MODEL_NAME, ModelVersion.is_active.is_(True))
    )


def _load_bundle(artifact_path: str) -> dict:
    bundle = _bundle_cache.get(artifact_path)
    if bundle is None:
        bundle = joblib.load(artifact_path)
        _bundle_cache[artifact_path] = bundle
    return bundle


def _get_explainer(artifact_path: str, bundle: dict) -> shap.Explainer:
    explainer = _explainer_cache.get(artifact_path)
    if explainer is None:
        explainer = shap.Explainer(bundle["model"], bundle["background_sample"])
        _explainer_cache[artifact_path] = explainer
    return explainer


def _resolve_rolling_temp(raw_temp, steel_kind: str, rolling_method: str, bundle: dict) -> float:
    if raw_temp is not None and raw_temp > 0:
        return float(raw_temp)
    key = f"{steel_kind}|{rolling_method}"
    return bundle["rolling_temp_group_median"].get(key, bundle["rolling_temp_global_median"])


def _build_feature_row(session: Session, plate_no: str, bundle: dict) -> dict | None:
    plate = session.get(Plate, plate_no)
    if plate is None:
        return None
    spec = session.get(ProductSpec, plate.spec_code)
    furnace = session.scalar(
        select(FurnaceProcess).where(FurnaceProcess.plate_no == plate_no).order_by(FurnaceProcess.recorded_at.desc())
    )
    rolling = session.scalar(select(RollingProcess).where(RollingProcess.plate_no == plate_no))
    if furnace is None or rolling is None:
        return None

    raw = {
        "pt_thick": plate.pt_thick,
        "pt_width": plate.pt_width,
        "pt_length": plate.pt_length,
        "spec_country": spec.spec_country,
        "steel_kind": spec.steel_kind,
        "fur_no": furnace.fur_no,
        "fur_input_row": furnace.fur_input_row,
        "fur_heat_temp": furnace.fur_heat_temp,
        "fur_heat_time": furnace.fur_heat_time,
        "fur_soak_temp": furnace.fur_soak_temp,
        "fur_soak_time": furnace.fur_soak_time,
        "fur_total_time": furnace.fur_total_time,
        "hsb_applied": rolling.hsb_applied,
        "rolling_method": rolling.rolling_method,
        "descaling_count": rolling.descaling_count,
        "work_group": rolling.work_group,
        "rolling_date": rolling.rolling_date,
    }
    raw["rolling_temp"] = _resolve_rolling_temp(rolling.rolling_temp, spec.steel_kind, rolling.rolling_method, bundle)
    return raw


def get_active_bundle(session: Session) -> tuple[ModelVersion, dict] | None:
    """batch_scoring.py 등 외부 모듈이 활성 모델 아티팩트를 얻기 위한 공개 진입점."""
    model_version = _get_active_model_version(session)
    if model_version is None:
        return None
    return model_version, _load_bundle(model_version.artifact_path)


def _score_raw_row(raw: dict, model_version: ModelVersion, bundle: dict):
    """
    raw: rolling_temp가 이미 대치(결측 없음)된 단일 레코드 dict (features.engineer_row 입력 형태).
    반환: (prob, label, top_features[(name, value, shap_value)], engineered_dict)
    predict_for_plate(DB 기록)와 predict_adhoc(DB 미기록)가 공유하는 채점 핵심 로직.
    """
    engineered = features.engineer_row(raw)
    row_df = pd.DataFrame([engineered])
    row_encoded = pd.get_dummies(row_df, columns=features.CATEGORICAL_FEATURES)
    row_encoded = features.align_to_model_columns(row_encoded, bundle["model_columns"])

    model = bundle["model"]
    prob = float(model.predict_proba(row_encoded)[0, 1])
    threshold = float(model_version.threshold)
    label = prob >= threshold

    explainer = _get_explainer(model_version.artifact_path, bundle)
    shap_values = explainer(row_encoded)
    values = shap_values.values[0]
    # 이진 분류 SHAP은 (n_features, n_classes) 형태로 나올 수 있어 양성 클래스만 취한다.
    if values.ndim == 2:
        values = values[:, 1]

    contrib = sorted(
        zip(bundle["model_columns"], row_encoded.iloc[0].tolist(), values.tolist()),
        key=lambda t: abs(t[2]),
        reverse=True,
    )[:TOP_N_FEATURES]

    return prob, bool(label), contrib, engineered


def predict_for_plate(session: Session, plate_no: str) -> PredictionOut | None:
    model_version = _get_active_model_version(session)
    if model_version is None:
        return None

    bundle = _load_bundle(model_version.artifact_path)
    raw = _build_feature_row(session, plate_no, bundle)
    if raw is None:
        return None

    prob, label, contrib, engineered = _score_raw_row(raw, model_version, bundle)

    prediction = Prediction(
        plate_no=plate_no,
        model_version_id=model_version.id,
        predicted_prob=round(prob, 4),
        predicted_label=label,
        feature_snapshot=engineered,
    )
    session.add(prediction)
    session.flush()

    for rank, (fname, fval, sval) in enumerate(contrib, start=1):
        session.add(
            ShapExplanation(
                prediction_id=prediction.id,
                feature_name=fname,
                feature_value=float(fval),
                shap_value=float(sval),
                rank=rank,
            )
        )
    session.commit()

    return PredictionOut(
        predicted_prob=prob,
        predicted_label=label,
        model_version=f"{model_version.name} {model_version.version}",
        top_features=[
            ShapFeatureOut(feature_name=f, feature_value=v, shap_value=s) for f, v, s in contrib
        ],
    )


def predict_adhoc(raw: dict, model_version: ModelVersion, bundle: dict) -> dict:
    """
    사용자가 화면에서 직접 입력한 값 하나를 채점한다. DB에는 아무것도 쓰지 않는다
    (실제 코일이 아니라 "이 조합이면 어떻게 될까"를 확인하는 what-if 용도).
    raw는 rolling_temp가 None일 수 있다(비워두면 그룹 중앙값으로 대치).
    """
    raw = dict(raw)
    raw.setdefault("rolling_date", datetime.now(timezone.utc))
    raw["rolling_temp"] = _resolve_rolling_temp(
        raw.get("rolling_temp"), raw["steel_kind"], raw["rolling_method"], bundle
    )

    fired = evaluate_rules_from_raw(raw)
    prob, label, contrib, _ = _score_raw_row(raw, model_version, bundle)

    return {
        "alerts": [AlertOut(rule_code=r.rule_code, severity=r.severity, message=r.message) for r in fired],
        "prediction": PredictionOut(
            predicted_prob=prob,
            predicted_label=label,
            model_version=f"{model_version.name} {model_version.version}",
            top_features=[
                ShapFeatureOut(feature_name=f, feature_value=v, shap_value=s) for f, v, s in contrib
            ],
        ),
    }


def get_latest_prediction(session: Session, plate_no: str) -> PredictionOut | None:
    prediction = session.scalar(
        select(Prediction).where(Prediction.plate_no == plate_no).order_by(Prediction.predicted_at.desc())
    )
    if prediction is None:
        return None

    model_version = session.get(ModelVersion, prediction.model_version_id)
    shap_rows = session.scalars(
        select(ShapExplanation).where(ShapExplanation.prediction_id == prediction.id).order_by(ShapExplanation.rank)
    ).all()

    return PredictionOut(
        predicted_prob=float(prediction.predicted_prob),
        predicted_label=prediction.predicted_label,
        model_version=f"{model_version.name} {model_version.version}",
        top_features=[
            ShapFeatureOut(feature_name=s.feature_name, feature_value=float(s.feature_value), shap_value=float(s.shap_value))
            for s in shap_rows
        ],
    )


def build_dashboard_summary(session: Session) -> dict:
    today_str = date.today().isoformat()

    today_plate_count = session.scalar(
        select(func.count()).select_from(Plate).where(func.date(Plate.created_at) == today_str)
    ) or 0

    today_predictions = session.scalars(
        select(Prediction).where(func.date(Prediction.predicted_at) == today_str)
    ).all()
    if today_predictions:
        defect_rate = sum(1 for p in today_predictions if p.predicted_label) / len(today_predictions)
    else:
        defect_rate = None

    open_alert_count = session.scalar(
        select(func.count()).select_from(Alert).where(Alert.resolved_at.is_(None))
    ) or 0

    active_model = _get_active_model_version(session)
    active_model_label = f"{active_model.name} {active_model.version}" if active_model else None

    return {
        "today_plate_count": today_plate_count,
        "today_predicted_defect_rate": defect_rate,
        "open_alert_count": open_alert_count,
        "active_model": active_model_label,
    }
