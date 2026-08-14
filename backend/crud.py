"""단건 CRUD/조회 헬퍼. app.py, prediction_service.py 등에서 공용으로 사용."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from models import ProductSpec


def get_or_create_spec(session: Session, spec_long: str, spec_country: str, steel_kind: str) -> str:
    existing = session.scalar(
        select(ProductSpec).where(
            ProductSpec.spec_long == spec_long,
            ProductSpec.spec_country == spec_country,
            ProductSpec.steel_kind == steel_kind,
        )
    )
    if existing:
        return existing.spec_code

    count = session.scalar(select(func.count()).select_from(ProductSpec)) or 0
    spec_code = f"SPEC{count + 1:03d}"
    session.add(ProductSpec(spec_code=spec_code, spec_long=spec_long, spec_country=spec_country, steel_kind=steel_kind))
    session.flush()
    return spec_code
