"""
Scale 불량 예측 시스템 — FastAPI 백엔드

Phase 2: Ingestion API(/plates, /furnace-records, /rolling-records) + Rule Engine
Phase 3: /rolling-records 응답에 실시간 예측 결과 포함, /predictions/{plate_no}
Phase 4: /dashboard/summary, /alerts 목록/처리
Phase 5: /quality-inspections, /models 목록/활성화 전환
"""

import io
import sys
from datetime import date, datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949)에서 한글/특수문자 로그 출력 깨짐 방지

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import batch_scoring
import prediction_service
from crud import get_or_create_spec
from csv_parsing import load_raw_scale_csv
from db import engine, get_session, init_db
from models import Alert, FurnaceProcess, ModelVersion, Plate, QualityInspection, RollingProcess, User
from rules import evaluate_rules
from schemas import (
    AdhocPredictIn,
    AdhocPredictResponse,
    AlertOut,
    AlertResolveIn,
    BatchPredictResponse,
    DashboardSummaryOut,
    FurnaceRecordIn,
    ModelVersionOut,
    PlateIn,
    PredictionOut,
    QualityInspectionIn,
    RollingRecordIn,
    RollingRecordResponse,
    ShapFeatureOut,
)

app = FastAPI(title="Scale 불량 예측 시스템 API", version="0.1.0")

# 개발 편의를 위한 전체 허용. 운영 배포 시 프런트엔드 도메인으로 제한할 것.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


DEMO_USERS = [
    {"name": "현장 엔지니어 데모", "email": "field.engineer@plant.local", "role": "FIELD_ENGINEER", "work_group": "1조"},
    {"name": "데이터 분석가 데모", "email": "data.analyst@plant.local", "role": "DATA_ANALYST", "work_group": None},
]


def _seed_demo_users(session: Session) -> None:
    """대시보드 UI에서 알림 처리/판정 입력을 바로 시연할 수 있도록 데모 계정을 보장한다."""
    for u in DEMO_USERS:
        if session.scalar(select(User).where(User.email == u["email"])) is None:
            session.add(User(**u))
    session.commit()


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    with Session(engine) as session:
        _seed_demo_users(session)


# ----------------------------------------------------------------------------
# 코일(제품) 등록 — 가열로/압연 데이터가 들어오기 전 선행 등록
# ----------------------------------------------------------------------------
@app.post("/api/v1/plates", status_code=201)
def create_plate(payload: PlateIn, session: Session = Depends(get_session)):
    if session.get(Plate, payload.plate_no) is not None:
        raise HTTPException(409, f"이미 등록된 plate_no: {payload.plate_no}")

    spec_code = get_or_create_spec(session, payload.spec_long, payload.spec_country, payload.steel_kind)
    plate = Plate(
        plate_no=payload.plate_no,
        spec_code=spec_code,
        pt_thick=payload.pt_thick,
        pt_width=payload.pt_width,
        pt_length=payload.pt_length,
    )
    session.add(plate)
    session.commit()
    return {"plate_no": plate.plate_no, "spec_code": spec_code}


# ----------------------------------------------------------------------------
# 가열로 공정 데이터 적재
# ----------------------------------------------------------------------------
@app.post("/api/v1/furnace-records", status_code=201)
def create_furnace_record(payload: FurnaceRecordIn, session: Session = Depends(get_session)):
    if session.get(Plate, payload.plate_no) is None:
        raise HTTPException(404, f"등록되지 않은 plate_no: {payload.plate_no} (먼저 /plates로 등록 필요)")

    record = FurnaceProcess(**payload.model_dump())
    session.add(record)
    session.commit()
    return {"id": record.id, "plate_no": record.plate_no}


# ----------------------------------------------------------------------------
# 압연 공정 데이터 적재 — 규칙 평가 + 예측 트리거
# ----------------------------------------------------------------------------
@app.post("/api/v1/rolling-records", status_code=201, response_model=RollingRecordResponse)
def create_rolling_record(payload: RollingRecordIn, session: Session = Depends(get_session)):
    plate = session.get(Plate, payload.plate_no)
    if plate is None:
        raise HTTPException(404, f"등록되지 않은 plate_no: {payload.plate_no} (먼저 /plates로 등록 필요)")

    furnace = session.scalar(
        select(FurnaceProcess)
        .where(FurnaceProcess.plate_no == payload.plate_no)
        .order_by(FurnaceProcess.recorded_at.desc())
    )
    if furnace is None:
        raise HTTPException(409, f"{payload.plate_no}의 가열로 기록이 없습니다 (furnace-records 선행 필요)")

    data = payload.model_dump()
    raw_temp = data.pop("rolling_temp")
    rolling = RollingProcess(
        **data,
        rolling_temp=None if raw_temp is None or raw_temp <= 0 else raw_temp,
    )
    session.add(rolling)
    session.flush()

    rule_hits = evaluate_rules(furnace, rolling)
    alerts_out: list[AlertOut] = []
    for hit in rule_hits:
        alert = Alert(
            plate_no=payload.plate_no,
            rule_code=hit.rule_code,
            severity=hit.severity,
            message=hit.message,
        )
        session.add(alert)
        alerts_out.append(AlertOut(rule_code=hit.rule_code, severity=hit.severity, message=hit.message))

    session.commit()

    prediction_out = prediction_service.predict_for_plate(session, payload.plate_no)

    return RollingRecordResponse(plate_no=payload.plate_no, alerts=alerts_out, prediction=prediction_out)


# ----------------------------------------------------------------------------
# 예측 결과 조회
# ----------------------------------------------------------------------------
@app.get("/api/v1/predictions/{plate_no}", response_model=PredictionOut)
def get_prediction(plate_no: str, session: Session = Depends(get_session)):
    result = prediction_service.get_latest_prediction(session, plate_no)
    if result is None:
        raise HTTPException(404, f"{plate_no}에 대한 예측 결과가 없습니다")
    return result


# ----------------------------------------------------------------------------
# 일괄 예측 — CSV 파일 전체를 활성 모델로 한 번에 채점 (DB에는 쓰지 않음)
# ----------------------------------------------------------------------------
@app.post("/api/v1/batch-predict", response_model=BatchPredictResponse)
async def batch_predict(file: UploadFile, session: Session = Depends(get_session)):
    active = prediction_service.get_active_bundle(session)
    if active is None:
        raise HTTPException(409, "활성화된 모델이 없습니다. 먼저 모델을 학습/활성화하세요.")
    model_version, bundle = active

    raw_bytes = await file.read()
    try:
        df = load_raw_scale_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(400, f"CSV 파싱 실패: {e}")

    required_cols = {
        "plate_no", "spec_long", "spec_country", "steel_kind", "pt_thick", "pt_width", "pt_length",
        "hsb", "fur_no", "fur_input_row", "fur_heat_temp", "fur_heat_time", "fur_soak_temp",
        "fur_soak_time", "fur_total_time", "rolling_method", "rolling_temp", "descaling_count",
        "work_group", "rolling_date",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(400, f"CSV에 필요한 컬럼이 없습니다: {sorted(missing)}")

    rows, summary = batch_scoring.score_dataframe(df, model_version, bundle)
    return BatchPredictResponse(summary=summary, rows=rows)


# ----------------------------------------------------------------------------
# 임의값 예측 (what-if) — 코일 등록 없이 값 하나를 그 자리에서 채점, DB에 안 씀
# ----------------------------------------------------------------------------
@app.post("/api/v1/predict-adhoc", response_model=AdhocPredictResponse)
def predict_adhoc(payload: AdhocPredictIn, session: Session = Depends(get_session)):
    active = prediction_service.get_active_bundle(session)
    if active is None:
        raise HTTPException(409, "활성화된 모델이 없습니다. 먼저 모델을 학습/활성화하세요.")
    model_version, bundle = active

    result = prediction_service.predict_adhoc(payload.model_dump(), model_version, bundle)
    return result


# ----------------------------------------------------------------------------
# 알림
# ----------------------------------------------------------------------------
@app.get("/api/v1/alerts")
def list_alerts(status: str = "open", plate_no: str | None = None, session: Session = Depends(get_session)):
    stmt = select(Alert).order_by(Alert.triggered_at.desc())
    if status == "open":
        stmt = stmt.where(Alert.resolved_at.is_(None))
    if plate_no:
        stmt = stmt.where(Alert.plate_no == plate_no)
    alerts = session.scalars(stmt).all()
    return [
        {
            "id": a.id,
            "plate_no": a.plate_no,
            "rule_code": a.rule_code,
            "severity": a.severity,
            "message": a.message,
            "triggered_at": a.triggered_at,
            "resolved_at": a.resolved_at,
        }
        for a in alerts
    ]


@app.patch("/api/v1/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: int, payload: AlertResolveIn, session: Session = Depends(get_session)):
    alert = session.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(404, "존재하지 않는 알림입니다")
    if alert.resolved_at is not None:
        raise HTTPException(409, "이미 처리된 알림입니다")

    resolver = session.scalar(select(User).where(User.email == payload.resolver_email))
    if resolver is None:
        raise HTTPException(404, f"존재하지 않는 사용자: {payload.resolver_email}")

    alert.resolved_at = datetime.now(timezone.utc)
    alert.resolved_by = resolver.id
    session.commit()
    return {"id": alert.id, "resolved_at": alert.resolved_at}


# ----------------------------------------------------------------------------
# 대시보드 요약
# ----------------------------------------------------------------------------
@app.get("/api/v1/dashboard/summary", response_model=DashboardSummaryOut)
def dashboard_summary(session: Session = Depends(get_session)):
    return DashboardSummaryOut(**prediction_service.build_dashboard_summary(session))


# ----------------------------------------------------------------------------
# 육안 판정 입력 (피드백 루프)
# ----------------------------------------------------------------------------
@app.post("/api/v1/quality-inspections", status_code=201)
def create_quality_inspection(payload: QualityInspectionIn, session: Session = Depends(get_session)):
    if session.get(Plate, payload.plate_no) is None:
        raise HTTPException(404, f"등록되지 않은 plate_no: {payload.plate_no}")

    inspector = session.scalar(select(User).where(User.email == payload.inspector_email))
    if inspector is None:
        raise HTTPException(404, f"존재하지 않는 사용자: {payload.inspector_email}")

    inspection = QualityInspection(
        plate_no=payload.plate_no,
        defect_type=payload.defect_type,
        inspector_id=inspector.id,
        note=payload.note,
    )
    session.add(inspection)
    session.commit()
    return {"id": inspection.id, "plate_no": inspection.plate_no, "defect_type": inspection.defect_type}


# ----------------------------------------------------------------------------
# 모델 관리
# ----------------------------------------------------------------------------
@app.get("/api/v1/models", response_model=list[ModelVersionOut])
def list_models(session: Session = Depends(get_session)):
    versions = session.scalars(select(ModelVersion).order_by(ModelVersion.trained_at.desc())).all()
    return list(versions)


@app.post("/api/v1/models/{model_id}/activate")
def activate_model(model_id: int, session: Session = Depends(get_session)):
    target = session.get(ModelVersion, model_id)
    if target is None:
        raise HTTPException(404, "존재하지 않는 모델 버전입니다")

    # 부분 유니크 인덱스(name당 활성 1개)를 어기지 않도록, 같은 이름의 기존 활성 모델을
    # 먼저 비활성화한 뒤 대상 모델을 활성화한다.
    others = session.scalars(
        select(ModelVersion).where(ModelVersion.name == target.name, ModelVersion.is_active.is_(True))
    ).all()
    for m in others:
        m.is_active = False
    session.flush()

    target.is_active = True
    session.commit()
    return {"id": target.id, "name": target.name, "version": target.version, "is_active": True}


# ----------------------------------------------------------------------------
# 정적 프런트엔드 (/app 이하) — API 라우트 등록 이후에 마운트해야 경로 충돌이 없다.
# ----------------------------------------------------------------------------
app.mount("/app", StaticFiles(directory="static", html=True), name="dashboard")
