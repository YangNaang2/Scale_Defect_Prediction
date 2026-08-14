"""Pydantic 요청/응답 스키마 (API 계약)."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PlateIn(BaseModel):
    plate_no: str
    spec_long: str
    spec_country: str
    steel_kind: Literal["C", "T"]
    pt_thick: int = Field(gt=0)
    pt_width: int = Field(gt=0)
    pt_length: int = Field(gt=0)


class FurnaceRecordIn(BaseModel):
    plate_no: str
    fur_no: int
    fur_input_row: int
    fur_heat_temp: int
    fur_heat_time: int
    fur_soak_temp: int
    fur_soak_time: int
    fur_total_time: int
    recorded_at: datetime


class RollingRecordIn(BaseModel):
    plate_no: str
    hsb_applied: bool
    rolling_method: Literal["TMCP", "CR"]
    rolling_temp: Optional[int] = None  # None 또는 0 이하 = 센서 미기록(결측)
    descaling_count: int = Field(ge=0)
    work_group: str
    rolling_date: datetime


class AlertOut(BaseModel):
    rule_code: str
    severity: str
    message: str


class ShapFeatureOut(BaseModel):
    feature_name: str
    feature_value: Optional[float]
    shap_value: float


class PredictionOut(BaseModel):
    predicted_prob: float
    predicted_label: bool
    model_version: str
    top_features: list[ShapFeatureOut] = []


class RollingRecordResponse(BaseModel):
    plate_no: str
    alerts: list[AlertOut]
    prediction: Optional[PredictionOut] = None


class QualityInspectionIn(BaseModel):
    plate_no: str
    defect_type: Literal["없음", "압입흠", "Scratch", "두께부족", "Scale"]
    inspector_email: str
    note: Optional[str] = None


class AlertResolveIn(BaseModel):
    resolver_email: str


class ModelVersionOut(BaseModel):
    id: int
    name: str
    version: str
    algorithm: str
    metrics: dict
    is_active: bool
    trained_at: datetime


class DashboardSummaryOut(BaseModel):
    today_plate_count: int
    today_predicted_defect_rate: Optional[float]
    open_alert_count: int
    active_model: Optional[str]


class BatchPredictRowOut(BaseModel):
    plate_no: str
    predicted_prob: float
    predicted_label: bool
    fired_rules: list[str]
    actual_label: Optional[bool] = None
    correct: Optional[bool] = None


class BatchPredictSummaryOut(BaseModel):
    total_rows: int
    predicted_defect_count: int
    predicted_defect_rate: float
    model_version: str
    has_actual_labels: bool
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    confusion_matrix: Optional[dict] = None


class BatchPredictResponse(BaseModel):
    summary: BatchPredictSummaryOut
    rows: list[BatchPredictRowOut]


class AdhocPredictIn(BaseModel):
    spec_country: str
    steel_kind: Literal["C", "T"]
    pt_thick: int = Field(gt=0)
    pt_width: int = Field(gt=0)
    pt_length: int = Field(gt=0)
    fur_no: int
    fur_input_row: int
    fur_heat_temp: int
    fur_heat_time: int
    fur_soak_temp: int
    fur_soak_time: int
    fur_total_time: int
    hsb_applied: bool
    rolling_method: Literal["TMCP", "CR"]
    rolling_temp: Optional[int] = None  # 비워두면 그룹 중앙값으로 대치
    descaling_count: int = Field(ge=0)
    work_group: str


class AdhocPredictResponse(BaseModel):
    alerts: list[AlertOut]
    prediction: PredictionOut
