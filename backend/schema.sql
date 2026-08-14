-- ============================================================
-- Scale 불량 예측 시스템 — Phase 1 DB 스키마 (PostgreSQL 15+)
-- 설계 근거: 팀 공유용 설계서 2장(데이터 모델링) 참고
-- ============================================================

-- ------------------------------------------------------------
-- 0. 공통 ENUM 타입
-- ------------------------------------------------------------
CREATE TYPE user_role AS ENUM ('FIELD_ENGINEER', 'DATA_ANALYST', 'ADMIN');
CREATE TYPE defect_type AS ENUM ('없음', '압입흠', 'Scratch', '두께부족', 'Scale');
CREATE TYPE rolling_method_type AS ENUM ('TMCP', 'CR');
CREATE TYPE alert_severity AS ENUM ('INFO', 'WARNING', 'CRITICAL');
CREATE TYPE alert_rule_code AS ENUM ('HSB_NOT_APPLIED', 'ROLLING_TEMP_OVER_1000', 'TEMP_DROP_INSUFFICIENT');

-- ------------------------------------------------------------
-- 1. 사용자
-- ------------------------------------------------------------
CREATE TABLE users (
    id              BIGSERIAL PRIMARY KEY,
    name            VARCHAR(50)  NOT NULL,
    email           VARCHAR(120) NOT NULL UNIQUE,
    role            user_role    NOT NULL,
    work_group      VARCHAR(10),                       -- 소속 조 (1조~4조), 관리자는 NULL
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------
-- 2. 제품 사양 (spec_long / spec_country / steel_kind 반복값 정규화)
-- ------------------------------------------------------------
CREATE TABLE product_spec (
    spec_code       VARCHAR(20)  PRIMARY KEY,           -- 예: 'SPEC001'
    spec_long       VARCHAR(50)  NOT NULL,               -- 예: 'AB/EH32-TM'
    spec_country    VARCHAR(20)  NOT NULL,
    steel_kind      CHAR(1)      NOT NULL,               -- 'C' | 'T'
    UNIQUE (spec_long, spec_country, steel_kind)
);

-- ------------------------------------------------------------
-- 3. 코일(제품) 마스터
-- ------------------------------------------------------------
CREATE TABLE plate (
    plate_no        VARCHAR(20)  PRIMARY KEY,            -- 예: 'PLT_1001'
    spec_code       VARCHAR(20)  NOT NULL REFERENCES product_spec(spec_code),
    pt_thick        SMALLINT     NOT NULL CHECK (pt_thick > 0),    -- mm
    pt_width        SMALLINT     NOT NULL CHECK (pt_width > 0),    -- mm
    pt_length       INTEGER      NOT NULL CHECK (pt_length > 0),   -- mm
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------
-- 4. 가열로 공정 기록
-- ------------------------------------------------------------
CREATE TABLE furnace_process (
    id              BIGSERIAL PRIMARY KEY,
    plate_no        VARCHAR(20)  NOT NULL REFERENCES plate(plate_no),
    fur_no          SMALLINT     NOT NULL,               -- 1,2,3호기
    fur_input_row   SMALLINT     NOT NULL,               -- 1열, 2열
    fur_heat_temp   SMALLINT     NOT NULL,               -- 가열대 온도(℃)
    fur_heat_time   SMALLINT     NOT NULL,               -- 분
    fur_soak_temp   SMALLINT     NOT NULL,               -- 균열대 온도(℃)
    fur_soak_time   SMALLINT     NOT NULL,               -- 분
    fur_total_time  SMALLINT     NOT NULL,               -- 분
    recorded_at     TIMESTAMPTZ  NOT NULL,
    UNIQUE (plate_no, recorded_at)
);
CREATE INDEX idx_furnace_plate ON furnace_process(plate_no);
CREATE INDEX idx_furnace_recorded_at ON furnace_process(recorded_at);

-- ------------------------------------------------------------
-- 5. 압연 공정 기록
-- ------------------------------------------------------------
CREATE TABLE rolling_process (
    id              BIGSERIAL PRIMARY KEY,
    plate_no        VARCHAR(20)  NOT NULL REFERENCES plate(plate_no),
    hsb_applied     BOOLEAN      NOT NULL,               -- Hot Scale Breaker 적용 여부
    rolling_method  rolling_method_type NOT NULL,
    rolling_temp    SMALLINT,                            -- ℃, NULL 허용(센서 미기록 -> 결측)
    descaling_count SMALLINT     NOT NULL CHECK (descaling_count >= 0),
    work_group      VARCHAR(10)  NOT NULL,               -- 1조~4조
    rolling_date    TIMESTAMPTZ  NOT NULL,
    recorded_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (plate_no)
);
CREATE INDEX idx_rolling_date ON rolling_process(rolling_date);
CREATE INDEX idx_rolling_hsb ON rolling_process(hsb_applied) WHERE hsb_applied = FALSE;

-- ------------------------------------------------------------
-- 6. 육안 판정(실측 라벨) — 재학습용 Ground Truth
-- ------------------------------------------------------------
CREATE TABLE quality_inspection (
    id              BIGSERIAL PRIMARY KEY,
    plate_no        VARCHAR(20)  NOT NULL REFERENCES plate(plate_no),
    defect_type     defect_type  NOT NULL,
    inspector_id    BIGINT       NOT NULL REFERENCES users(id),
    inspected_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    note            TEXT
);
CREATE INDEX idx_inspection_plate ON quality_inspection(plate_no);

-- ------------------------------------------------------------
-- 7. 모델 레지스트리
-- ------------------------------------------------------------
CREATE TABLE model_version (
    id              BIGSERIAL PRIMARY KEY,
    name            VARCHAR(50)  NOT NULL,               -- 'scale_defect_classifier'
    version         VARCHAR(20)  NOT NULL,               -- 'v1.3.0'
    algorithm       VARCHAR(30)  NOT NULL,               -- 'XGBoost', 'RandomForest' 등
    trained_at      TIMESTAMPTZ  NOT NULL,
    train_row_count INTEGER      NOT NULL,
    metrics         JSONB        NOT NULL,               -- {"precision":0.81,"recall":0.77,"auc":0.89,...}
    threshold       NUMERIC(4,3) NOT NULL DEFAULT 0.5,
    artifact_path   VARCHAR(200) NOT NULL,               -- MinIO/S3 경로
    is_active       BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (name, version)
);
-- 활성 모델은 name당 1개만 허용
CREATE UNIQUE INDEX idx_model_active_unique
    ON model_version(name) WHERE is_active = TRUE;

-- ------------------------------------------------------------
-- 8. 예측 결과
-- ------------------------------------------------------------
CREATE TABLE prediction (
    id                  BIGSERIAL PRIMARY KEY,
    plate_no            VARCHAR(20)  NOT NULL REFERENCES plate(plate_no),
    model_version_id    BIGINT       NOT NULL REFERENCES model_version(id),
    predicted_prob      NUMERIC(5,4) NOT NULL CHECK (predicted_prob BETWEEN 0 AND 1),
    predicted_label     BOOLEAN      NOT NULL,           -- prob >= threshold
    feature_snapshot    JSONB        NOT NULL,           -- 예측 시점 파생피처 전체 (재현성용)
    predicted_at        TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX idx_prediction_plate ON prediction(plate_no);
CREATE INDEX idx_prediction_at ON prediction(predicted_at);

-- ------------------------------------------------------------
-- 9. SHAP 근거 (예측 1건당 피처 N개)
-- ------------------------------------------------------------
CREATE TABLE shap_explanation (
    id              BIGSERIAL PRIMARY KEY,
    prediction_id   BIGINT       NOT NULL REFERENCES prediction(id) ON DELETE CASCADE,
    feature_name    VARCHAR(60)  NOT NULL,
    feature_value   NUMERIC,
    shap_value      NUMERIC      NOT NULL,               -- 양수=불량 확률 증가 기여
    rank            SMALLINT     NOT NULL                -- 기여도 절대값 기준 순위 (1=가장 큰 영향)
);
CREATE INDEX idx_shap_prediction ON shap_explanation(prediction_id);

-- ------------------------------------------------------------
-- 10. 규칙 기반 경보
-- ------------------------------------------------------------
CREATE TABLE alert (
    id              BIGSERIAL PRIMARY KEY,
    plate_no        VARCHAR(20)  NOT NULL REFERENCES plate(plate_no),
    rule_code       alert_rule_code NOT NULL,
    severity        alert_severity  NOT NULL,
    message         VARCHAR(200)    NOT NULL,
    triggered_at    TIMESTAMPTZ     NOT NULL DEFAULT now(),
    resolved_at     TIMESTAMPTZ,
    resolved_by     BIGINT          REFERENCES users(id)
);
CREATE INDEX idx_alert_open ON alert(triggered_at) WHERE resolved_at IS NULL;
CREATE INDEX idx_alert_plate ON alert(plate_no);
