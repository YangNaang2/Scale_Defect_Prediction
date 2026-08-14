"""
Scale 불량 예측 시스템 — SQLAlchemy 2.0 ORM 모델

schema.sql(PostgreSQL 운영 DDL)과 1:1로 대응한다. 로컬 개발/테스트에서는
DATABASE_URL 미지정 시 SQLite(scale_system.db)를 기본값으로 쓰고, 운영에서는
DATABASE_URL=postgresql+psycopg2://... 로 교체한다.

주의: JSONB(Postgres 전용)는 SQLAlchemy의 범용 JSON 타입으로 선언했다.
      Postgres에서는 JSON 컬럼으로 매핑되므로, 운영 DB는 schema.sql을 그대로
      적용해 JSONB로 생성하는 쪽을 우선한다(이 모델은 SELECT/INSERT 용도로는
      두 타입 모두와 호환된다).
"""

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Enum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# BIGSERIAL(Postgres)에 대응하는 기본 키 타입. SQLite는 BigInteger를 rowid
# 별칭(자동증가)으로 인식하지 못하므로, SQLite에서만 Integer로 내려 쓴다.
BigIntPK = BigInteger().with_variant(Integer, "sqlite")


# ----------------------------------------------------------------------------
# 1. 사용자
# ----------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    email: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    role: Mapped[str] = mapped_column(
        Enum("FIELD_ENGINEER", "DATA_ANALYST", "ADMIN", name="user_role"),
        nullable=False,
    )
    work_group: Mapped[str | None] = mapped_column(String(10))
    created_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)


# ----------------------------------------------------------------------------
# 2. 제품 사양
# ----------------------------------------------------------------------------
class ProductSpec(Base):
    __tablename__ = "product_spec"

    spec_code: Mapped[str] = mapped_column(String(20), primary_key=True)
    spec_long: Mapped[str] = mapped_column(String(50), nullable=False)
    spec_country: Mapped[str] = mapped_column(String(20), nullable=False)
    steel_kind: Mapped[str] = mapped_column(String(1), nullable=False)

    __table_args__ = (
        UniqueConstraint("spec_long", "spec_country", "steel_kind", name="uq_product_spec_combo"),
    )

    plates: Mapped[list["Plate"]] = relationship(back_populates="spec")


# ----------------------------------------------------------------------------
# 3. 코일(제품) 마스터
# ----------------------------------------------------------------------------
class Plate(Base):
    __tablename__ = "plate"

    plate_no: Mapped[str] = mapped_column(String(20), primary_key=True)
    spec_code: Mapped[str] = mapped_column(ForeignKey("product_spec.spec_code"), nullable=False)
    pt_thick: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    pt_width: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    pt_length: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)

    __table_args__ = (
        CheckConstraint("pt_thick > 0", name="ck_plate_thick_positive"),
        CheckConstraint("pt_width > 0", name="ck_plate_width_positive"),
        CheckConstraint("pt_length > 0", name="ck_plate_length_positive"),
    )

    spec: Mapped["ProductSpec"] = relationship(back_populates="plates")
    furnace_records: Mapped[list["FurnaceProcess"]] = relationship(back_populates="plate")
    rolling_record: Mapped["RollingProcess"] = relationship(back_populates="plate", uselist=False)
    inspections: Mapped[list["QualityInspection"]] = relationship(back_populates="plate")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="plate")
    alerts: Mapped[list["Alert"]] = relationship(back_populates="plate")


# ----------------------------------------------------------------------------
# 4. 가열로 공정 기록
# ----------------------------------------------------------------------------
class FurnaceProcess(Base):
    __tablename__ = "furnace_process"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    plate_no: Mapped[str] = mapped_column(ForeignKey("plate.plate_no"), nullable=False)
    fur_no: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    fur_input_row: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    fur_heat_temp: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    fur_heat_time: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    fur_soak_temp: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    fur_soak_time: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    fur_total_time: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(nullable=False)

    __table_args__ = (
        UniqueConstraint("plate_no", "recorded_at", name="uq_furnace_plate_recorded"),
        Index("idx_furnace_plate", "plate_no"),
        Index("idx_furnace_recorded_at", "recorded_at"),
    )

    plate: Mapped["Plate"] = relationship(back_populates="furnace_records")


# ----------------------------------------------------------------------------
# 5. 압연 공정 기록
# ----------------------------------------------------------------------------
class RollingProcess(Base):
    __tablename__ = "rolling_process"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    plate_no: Mapped[str] = mapped_column(ForeignKey("plate.plate_no"), nullable=False, unique=True)
    hsb_applied: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rolling_method: Mapped[str] = mapped_column(Enum("TMCP", "CR", name="rolling_method_type"), nullable=False)
    rolling_temp: Mapped[int | None] = mapped_column(SmallInteger)  # NULL = 센서 미기록(결측)
    descaling_count: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    work_group: Mapped[str] = mapped_column(String(10), nullable=False)
    rolling_date: Mapped[datetime] = mapped_column(nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)

    __table_args__ = (
        CheckConstraint("descaling_count >= 0", name="ck_rolling_descaling_nonneg"),
        Index("idx_rolling_date", "rolling_date"),
    )

    plate: Mapped["Plate"] = relationship(back_populates="rolling_record")


# 부분 인덱스(hsb_applied = FALSE)는 테이블 확정 후 컬럼 객체를 참조해 별도로 정의한다.
Index(
    "idx_rolling_hsb",
    RollingProcess.__table__.c.hsb_applied,
    postgresql_where=RollingProcess.__table__.c.hsb_applied.is_(False),
    sqlite_where=RollingProcess.__table__.c.hsb_applied.is_(False),
)


# ----------------------------------------------------------------------------
# 6. 육안 판정 (실측 라벨)
# ----------------------------------------------------------------------------
class QualityInspection(Base):
    __tablename__ = "quality_inspection"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    plate_no: Mapped[str] = mapped_column(ForeignKey("plate.plate_no"), nullable=False)
    defect_type: Mapped[str] = mapped_column(
        Enum("없음", "압입흠", "Scratch", "두께부족", "Scale", name="defect_type"),
        nullable=False,
    )
    inspector_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    inspected_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)
    note: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (Index("idx_inspection_plate", "plate_no"),)

    plate: Mapped["Plate"] = relationship(back_populates="inspections")
    inspector: Mapped["User"] = relationship()


# ----------------------------------------------------------------------------
# 7. 모델 레지스트리
# ----------------------------------------------------------------------------
class ModelVersion(Base):
    __tablename__ = "model_version"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(30), nullable=False)
    trained_at: Mapped[datetime] = mapped_column(nullable=False)
    train_row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    threshold: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False, default=0.5)
    artifact_path: Mapped[str] = mapped_column(String(200), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)

    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_name_version"),)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model_version")


Index(
    "idx_model_active_unique",
    ModelVersion.__table__.c.name,
    unique=True,
    postgresql_where=ModelVersion.__table__.c.is_active.is_(True),
    sqlite_where=ModelVersion.__table__.c.is_active.is_(True),
)


# ----------------------------------------------------------------------------
# 8. 예측 결과
# ----------------------------------------------------------------------------
class Prediction(Base):
    __tablename__ = "prediction"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    plate_no: Mapped[str] = mapped_column(ForeignKey("plate.plate_no"), nullable=False)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_version.id"), nullable=False)
    predicted_prob: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    predicted_label: Mapped[bool] = mapped_column(Boolean, nullable=False)
    feature_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)

    __table_args__ = (
        CheckConstraint("predicted_prob >= 0 AND predicted_prob <= 1", name="ck_prediction_prob_range"),
        Index("idx_prediction_plate", "plate_no"),
        Index("idx_prediction_at", "predicted_at"),
    )

    plate: Mapped["Plate"] = relationship(back_populates="predictions")
    model_version: Mapped["ModelVersion"] = relationship(back_populates="predictions")
    shap_values: Mapped[list["ShapExplanation"]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan"
    )


# ----------------------------------------------------------------------------
# 9. SHAP 근거
# ----------------------------------------------------------------------------
class ShapExplanation(Base):
    __tablename__ = "shap_explanation"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("prediction.id", ondelete="CASCADE"), nullable=False
    )
    feature_name: Mapped[str] = mapped_column(String(60), nullable=False)
    feature_value: Mapped[float | None] = mapped_column(Numeric)
    shap_value: Mapped[float] = mapped_column(Numeric, nullable=False)
    rank: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    __table_args__ = (Index("idx_shap_prediction", "prediction_id"),)

    prediction: Mapped["Prediction"] = relationship(back_populates="shap_values")


# ----------------------------------------------------------------------------
# 10. 규칙 기반 경보
# ----------------------------------------------------------------------------
class Alert(Base):
    __tablename__ = "alert"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    plate_no: Mapped[str] = mapped_column(ForeignKey("plate.plate_no"), nullable=False)
    rule_code: Mapped[str] = mapped_column(
        Enum(
            "HSB_NOT_APPLIED",
            "ROLLING_TEMP_OVER_1000",
            "TEMP_DROP_INSUFFICIENT",
            name="alert_rule_code",
        ),
        nullable=False,
    )
    severity: Mapped[str] = mapped_column(Enum("INFO", "WARNING", "CRITICAL", name="alert_severity"), nullable=False)
    message: Mapped[str] = mapped_column(String(200), nullable=False)
    triggered_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column()
    resolved_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))

    __table_args__ = (Index("idx_alert_plate", "plate_no"),)

    plate: Mapped["Plate"] = relationship(back_populates="alerts")
    resolver: Mapped["User | None"] = relationship()


# 부분 인덱스(resolved_at IS NULL = 미해결 경보)는 테이블 확정 후 별도로 정의한다.
Index(
    "idx_alert_open",
    Alert.__table__.c.triggered_at,
    postgresql_where=Alert.__table__.c.resolved_at.is_(None),
    sqlite_where=Alert.__table__.c.resolved_at.is_(None),
)
