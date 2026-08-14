"""
결정론적 규칙 엔진.

설계 문서 1장에서 밝힌 대로, 통계적으로 거의 확정적인 패턴(HSB 미적용=100% 불량 등)은
ML 추론을 거치지 않고 여기서 즉시 판정한다. 임계값은 EDA(preprocessing.py 검증) 결과를
그대로 반영한다:
  - HSB 미적용 47/47건(100%) 불량
  - HSB 적용 상태에서도 rolling_temp>=1000℃ 구간 불량률 97.7%
  - 온도낙차(균열대-사상압연) 평균: 양품 228.9℃ vs 불량 170.7℃
"""

from dataclasses import dataclass
from types import SimpleNamespace

ROLLING_TEMP_CRITICAL = 1000
TEMP_DROP_WARNING_BELOW = 190  # 양품/불량 평균의 중간값 근방


@dataclass
class RuleResult:
    rule_code: str
    severity: str
    message: str


def evaluate_rules(furnace, rolling) -> list[RuleResult]:
    """furnace: FurnaceProcess ORM 인스턴스, rolling: RollingProcess ORM 인스턴스"""
    results: list[RuleResult] = []

    if not rolling.hsb_applied:
        results.append(
            RuleResult(
                rule_code="HSB_NOT_APPLIED",
                severity="CRITICAL",
                message="Hot Scale Breaker 미적용 — 과거 데이터 기준 해당 조건 100% Scale 불량 발생",
            )
        )

    if rolling.rolling_temp is not None and rolling.rolling_temp >= ROLLING_TEMP_CRITICAL:
        results.append(
            RuleResult(
                rule_code="ROLLING_TEMP_OVER_1000",
                severity="CRITICAL",
                message=(
                    f"사상압연온도 {rolling.rolling_temp}℃ — 임계치({ROLLING_TEMP_CRITICAL}℃) 초과, "
                    "HSB 적용 상태에서도 불량 위험 97.7%"
                ),
            )
        )

    if rolling.rolling_temp is not None:
        temp_drop = furnace.fur_soak_temp - rolling.rolling_temp
        if temp_drop < TEMP_DROP_WARNING_BELOW:
            results.append(
                RuleResult(
                    rule_code="TEMP_DROP_INSUFFICIENT",
                    severity="WARNING",
                    message=(
                        f"온도낙차 {temp_drop}℃ — 관리 기준({TEMP_DROP_WARNING_BELOW}℃) 미달, "
                        "양품 평균 낙차(228.9℃) 대비 냉각 부족"
                    ),
                )
            )

    return results


def evaluate_rules_from_raw(raw: dict) -> list[RuleResult]:
    """
    ORM 인스턴스가 없는 곳(일괄 채점, 임의값 예측)에서 쓰는 편의 함수.
    raw는 최소 hsb_applied, rolling_temp, fur_soak_temp 키를 가져야 한다.
    """
    furnace = SimpleNamespace(fur_soak_temp=raw["fur_soak_temp"])
    rolling = SimpleNamespace(hsb_applied=raw["hsb_applied"], rolling_temp=raw["rolling_temp"])
    return evaluate_rules(furnace, rolling)
