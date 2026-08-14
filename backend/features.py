"""
피처 엔지니어링 — preprocessing.py의 engineer_features()와 동일한 로직을
단일 레코드(dict) 기준으로 다시 구현한 것. 학습(train_model.py)과 실시간
서빙(prediction_service.py)이 이 모듈 하나만 공유해 train/serve 스큐를 막는다.
"""

from datetime import datetime

import pandas as pd

NUMERIC_FEATURES = [
    "pt_thick", "pt_width", "pt_length",
    "fur_heat_temp", "fur_heat_time",
    "fur_soak_temp", "fur_soak_time", "fur_total_time",
    "rolling_temp", "descaling_count",
    "temp_drop", "heat_soak_gap", "cooling_rate_proxy",
    "hsb_x_rolling_temp", "hsb_x_descaling",
    "rolling_hour", "rolling_dow",
]
BINARY_FEATURES = ["hsb_applied", "rolling_temp_over_1000", "hsb_x_over1000", "descaling_is_odd"]
CATEGORICAL_FEATURES = ["spec_country", "steel_kind", "fur_no", "fur_input_row", "rolling_method", "work_group", "thick_bucket"]


def bucket_thickness(pt_thick: float) -> str:
    # XGBoost/LightGBM은 피처명에 '[', ']', '<' 문자를 허용하지 않아 이를 피해서 표기한다.
    if pt_thick < 20:
        return "박(20미만)"
    if pt_thick < 40:
        return "중(20-40)"
    if pt_thick < 60:
        return "후(40-60)"
    return "극후(60이상)"


def engineer_row(raw: dict) -> dict:
    """raw: rolling_temp가 이미 대치(결측 없음)된 단일 레코드 dict."""
    hsb_applied = int(bool(raw["hsb_applied"]))
    rolling_temp = float(raw["rolling_temp"])
    fur_soak_temp = float(raw["fur_soak_temp"])
    fur_heat_temp = float(raw["fur_heat_temp"])
    fur_total_time = float(raw["fur_total_time"])
    descaling_count = int(raw["descaling_count"])
    pt_thick = float(raw["pt_thick"])
    rolling_date: datetime = raw["rolling_date"]

    temp_drop = fur_soak_temp - rolling_temp
    heat_soak_gap = fur_heat_temp - fur_soak_temp
    rolling_temp_over_1000 = int(rolling_temp >= 1000)
    cooling_rate_proxy = temp_drop / fur_total_time if fur_total_time else 0.0

    return {
        "pt_thick": pt_thick,
        "pt_width": float(raw["pt_width"]),
        "pt_length": float(raw["pt_length"]),
        "fur_heat_temp": fur_heat_temp,
        "fur_heat_time": float(raw["fur_heat_time"]),
        "fur_soak_temp": fur_soak_temp,
        "fur_soak_time": float(raw["fur_soak_time"]),
        "fur_total_time": fur_total_time,
        "rolling_temp": rolling_temp,
        "descaling_count": descaling_count,
        "temp_drop": temp_drop,
        "heat_soak_gap": heat_soak_gap,
        "cooling_rate_proxy": cooling_rate_proxy,
        "hsb_x_rolling_temp": hsb_applied * rolling_temp,
        "hsb_x_descaling": hsb_applied * descaling_count,
        "rolling_hour": rolling_date.hour,
        "rolling_dow": rolling_date.weekday(),
        "hsb_applied": hsb_applied,
        "rolling_temp_over_1000": rolling_temp_over_1000,
        "hsb_x_over1000": hsb_applied * rolling_temp_over_1000,
        "descaling_is_odd": int(descaling_count % 2 == 1),
        "spec_country": raw["spec_country"],
        "steel_kind": raw["steel_kind"],
        "fur_no": str(raw["fur_no"]),
        "fur_input_row": str(raw["fur_input_row"]),
        "rolling_method": raw["rolling_method"],
        "work_group": raw["work_group"],
        "thick_bucket": bucket_thickness(pt_thick),
    }


def align_to_model_columns(df_encoded: pd.DataFrame, model_columns: list[str]) -> pd.DataFrame:
    """
    원-핫 인코딩된 프레임을 학습 시점 컬럼 순서/구성에 맞춰 재정렬한다.
    pandas 3.x는 get_dummies의 bool 컬럼과 원본 수치 컬럼이 섞이면 프레임 전체가
    object dtype이 되는 경우가 있어(TreeExplainer가 여기서 죽는다), 학습/실시간
    서빙/일괄 예측 세 경로 모두 이 함수 하나로 정렬+float64 캐스팅을 통일한다.
    """
    return df_encoded.reindex(columns=model_columns, fill_value=0).astype("float64")
