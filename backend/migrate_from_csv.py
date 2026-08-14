"""
Phase 1 마이그레이션: SCALE불량.csv (레거시 플랫 파일) -> 정규화된 스키마

레거시 CSV는 코일 1건당 가열로+압연 공정이 한 행에 뒤섞여 있고, 육안판정도
'양품'/'불량' 이진값만 있다. 이 스크립트는 그것을 product_spec / plate /
furnace_process / rolling_process / quality_inspection 다섯 테이블로 분해해
적재한다.

전제(이 스크립트 한정 가정, 실 운영 Ingestion API는 각 공정에서 실시간으로
개별 호출되므로 아래 가정이 필요 없다):
  - furnace_process.recorded_at 은 legacy 데이터에 별도 타임스탬프가 없으므로
    rolling_date와 동일하게 채운다(실제로는 압연보다 앞서지만, 과거 이력
    적재 목적이라 분 단위 오차는 분석에 영향 없음).
  - quality_inspection.inspector_id 는 실제 판정자가 기록돼 있지 않으므로
    "시스템 마이그레이션 계정"으로 남긴다.
  - defect_type 은 원본에 'Scale' 여부만 있으므로 불량=Scale, 양품=없음으로
    매핑한다(압입흠/Scratch/두께부족은 이번 레거시 데이터에 없음).
  - rolling_temp<=0 은 전처리 단계(preprocessing.py)에서 확인한 대로 센서
    미기록으로 보고 NULL로 적재한다(값 대치는 피처 생성 시점에서 수행).

재실행해도 안전하도록(idempotent) 이미 존재하는 plate_no는 건너뛴다.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949)에서 한글/특수문자 출력 깨짐 방지

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from csv_parsing import extract_int, load_raw_scale_csv, to_rolling_method
from models import (
    Base,
    FurnaceProcess,
    Plate,
    ProductSpec,
    QualityInspection,
    RollingProcess,
    User,
)

SYSTEM_USER_EMAIL = "system-migration@plant.local"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCALE불량.csv를 정규화된 스키마로 마이그레이션")
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).resolve().parent.parent / "data" / "raw" / "SCALE불량.csv"),
        help="원본 CSV 경로 (기본: data/raw/SCALE불량.csv)",
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL", "sqlite:///scale_system.db"),
        help="SQLAlchemy DB URL (기본: 로컬 SQLite 파일). 운영은 postgresql+psycopg2://... 지정",
    )
    return parser.parse_args()


def ensure_system_user(session: Session) -> User:
    user = session.scalar(select(User).where(User.email == SYSTEM_USER_EMAIL))
    if user is None:
        user = User(
            name="시스템 마이그레이션 계정",
            email=SYSTEM_USER_EMAIL,
            role="ADMIN",
            work_group=None,
        )
        session.add(user)
        session.flush()
    return user


def upsert_product_specs(session: Session, df: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    combos = sorted(
        {(row.spec_long, row.spec_country, row.steel_kind) for row in df.itertuples()}
    )

    existing = {
        (s.spec_long, s.spec_country, s.steel_kind): s.spec_code
        for s in session.scalars(select(ProductSpec))
    }

    spec_code_map: dict[tuple[str, str, str], str] = dict(existing)
    next_seq = len(existing) + 1
    for combo in combos:
        if combo in spec_code_map:
            continue
        spec_code = f"SPEC{next_seq:03d}"
        session.add(
            ProductSpec(spec_code=spec_code, spec_long=combo[0], spec_country=combo[1], steel_kind=combo[2])
        )
        spec_code_map[combo] = spec_code
        next_seq += 1

    session.flush()
    return spec_code_map


def migrate(csv_path: str, db_url: str) -> None:
    df = load_raw_scale_csv(csv_path)
    print(f"[로드] {csv_path} -> {len(df)}행")

    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    print(f"[스키마] 테이블 생성/확인 완료 -> {db_url}")

    with Session(engine) as session:
        system_user = ensure_system_user(session)
        spec_code_map = upsert_product_specs(session, df)
        print(f"[product_spec] {len(spec_code_map)}건 확정")

        already_migrated = {p for (p,) in session.execute(select(Plate.plate_no))}
        new_rows = df[~df["plate_no"].isin(already_migrated)]
        skipped = len(df) - len(new_rows)
        if skipped:
            print(f"[스킵] 이미 적재된 plate_no {skipped}건 제외")

        n_plate = n_furnace = n_rolling = n_inspection = 0

        for row in new_rows.itertuples():
            spec_code = spec_code_map[(row.spec_long, row.spec_country, row.steel_kind)]

            session.add(
                Plate(
                    plate_no=row.plate_no,
                    spec_code=spec_code,
                    pt_thick=int(row.pt_thick),
                    pt_width=int(row.pt_width),
                    pt_length=int(row.pt_length),
                    created_at=row.rolling_date,
                )
            )
            n_plate += 1

            session.add(
                FurnaceProcess(
                    plate_no=row.plate_no,
                    fur_no=extract_int(row.fur_no),
                    fur_input_row=extract_int(row.fur_input_row),
                    fur_heat_temp=int(row.fur_heat_temp),
                    fur_heat_time=int(row.fur_heat_time),
                    fur_soak_temp=int(row.fur_soak_temp),
                    fur_soak_time=int(row.fur_soak_time),
                    fur_total_time=int(row.fur_total_time),
                    recorded_at=row.rolling_date,  # 가정: 레거시 데이터엔 별도 타임스탬프 없음
                )
            )
            n_furnace += 1

            raw_temp = float(row.rolling_temp)
            session.add(
                RollingProcess(
                    plate_no=row.plate_no,
                    hsb_applied=(row.hsb == "적용"),
                    rolling_method=to_rolling_method(row.rolling_method),
                    rolling_temp=None if raw_temp <= 0 else int(raw_temp),
                    descaling_count=int(row.descaling_count),
                    work_group=row.work_group,
                    rolling_date=row.rolling_date,
                )
            )
            n_rolling += 1

            session.add(
                QualityInspection(
                    plate_no=row.plate_no,
                    defect_type="Scale" if row.scale == "불량" else "없음",
                    inspector_id=system_user.id,
                    inspected_at=row.rolling_date,
                    note="legacy CSV 일괄 마이그레이션",
                )
            )
            n_inspection += 1

        session.commit()

    print(
        f"[적재 완료] plate={n_plate}, furnace_process={n_furnace}, "
        f"rolling_process={n_rolling}, quality_inspection={n_inspection}"
    )


if __name__ == "__main__":
    args = parse_args()
    migrate(args.csv, args.db_url)
