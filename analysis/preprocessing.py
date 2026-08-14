"""
Scale 불량 데이터 전처리 & 피처 엔지니어링
- 입력: SCALE불량.csv (cp949 인코딩)
- 출력: scale_processed.csv, train/valid/test 분할 결과

사전 EDA로 확인된 사실 (코드 내 처리 근거):
  1) hsb == '미적용' 인 47건은 전량 불량 (100%) -> 강력한 단일 신호, 결측 아님
  2) rolling_temp == 0 인 6건은 물리적으로 불가능한 값 -> 센서 미기록으로 보고 결측 처리
  3) rolling_temp >= 1000 구간에서 불량률이 급증 (임계효과, 비선형)
  4) fur_soak_temp - rolling_temp (온도 낙차)가 양품/불량 간 뚜렷한 차이를 보임
  5) rolling_date가 존재 -> 랜덤 분할이 아닌 시간 기준 분할이 필요
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "data" / "raw" / "SCALE불량.csv"
OUT_PATH = REPO_ROOT / "data" / "processed" / "scale_processed.csv"

# ----------------------------------------------------------------------------
# 1. 데이터 로드
# ----------------------------------------------------------------------------
def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp949")
    df["rolling_date"] = pd.to_datetime(df["rolling_date"], format="%d%b%Y:%H:%M:%S")
    return df


# ----------------------------------------------------------------------------
# 2. 결측치 / 이상치 처리
# ----------------------------------------------------------------------------
def clean_missing_and_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # (a) rolling_temp == 0 : 물리적으로 불가능한 값 -> 결측(NaN)으로 치환
    #     실측 6건 전부 '양품' 라벨이라 방치하면 "온도 0=양품"이라는 허위 패턴을
    #     모델이 학습할 위험이 있음. 같은 강종(steel_kind) x 압연방식(rolling_method)
    #     그룹의 중앙값으로 대치한다.
    df["rolling_temp_was_missing"] = (df["rolling_temp"] <= 0).astype(int)
    n_bad_zero = df["rolling_temp_was_missing"].sum()
    df.loc[df["rolling_temp"] <= 0, "rolling_temp"] = np.nan

    group_median = df.groupby(["steel_kind", "rolling_method"])["rolling_temp"].transform("median")
    df["rolling_temp"] = df["rolling_temp"].fillna(group_median)
    # 그룹 내에도 결측만 있을 극단적 경우를 대비한 전체 중앙값 폴백
    df["rolling_temp"] = df["rolling_temp"].fillna(df["rolling_temp"].median())

    print(f"[클린징] rolling_temp<=0 -> 결측 처리 후 그룹 중앙값 대치: {n_bad_zero}건 (rolling_temp_was_missing 플래그로 보존)")

    # (b) 일반 결측치 점검 (수치형은 중앙값, 범주형은 최빈값)
    num_cols = [
        "pt_thick", "pt_width", "pt_length",
        "fur_heat_temp", "fur_heat_time",
        "fur_soak_temp", "fur_soak_time", "fur_total_time",
        "rolling_temp", "descaling_count",
    ]
    cat_cols = [
        "spec_long", "spec_country", "steel_kind", "hsb",
        "fur_no", "fur_input_row", "rolling_method", "work_group",
    ]

    for c in num_cols:
        n_na = df[c].isna().sum()
        if n_na > 0:
            df[c] = df[c].fillna(df[c].median())
            print(f"[클린징] {c}: 결측 {n_na}건 -> 중앙값 대치")

    for c in cat_cols:
        n_na = df[c].isna().sum()
        if n_na > 0:
            df[c] = df[c].fillna(df[c].mode().iloc[0])
            print(f"[클린징] {c}: 결측 {n_na}건 -> 최빈값 대치")

    # (c) IQR 기반 극단 이상치 점검 (온도 낙차 계산 전, 참고용 로그만 출력 -> 임의 제거는 하지 않음)
    #     Scale 불량 도메인 특성상 "이상치"가 실제로는 임계효과(1000도 이상)의 핵심 신호이므로
    #     기계적으로 잘라내지 않고 클리핑 대신 로그만 남긴다.
    for c in ["fur_heat_temp", "fur_soak_temp", "rolling_temp"]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[c] < lo) | (df[c] > hi)).sum()
        print(f"[이상치 점검] {c}: IQR 범위[{lo:.1f}, {hi:.1f}] 밖 {n_out}건 (제거하지 않음, 임계효과 신호 가능성)")

    return df


# ----------------------------------------------------------------------------
# 3. 파생 변수 (피처 엔지니어링)
# ----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # HSB 이진화 (모델 입력용)
    df["hsb_applied"] = (df["hsb"] == "적용").astype(int)

    # --- (H3) 온도 낙차: 균열대(추출온도 proxy) -> 사상압연 온도까지 얼마나 식었는가 ---
    df["temp_drop"] = df["fur_soak_temp"] - df["rolling_temp"]

    # 가열대 -> 균열대 온도차 (가열로 내부 승온 패턴)
    df["heat_soak_gap"] = df["fur_heat_temp"] - df["fur_soak_temp"]

    # --- (H2) 사상압연 온도 임계값 플래그: EDA에서 확인된 ~1000도 변곡점 ---
    df["rolling_temp_over_1000"] = (df["rolling_temp"] >= 1000).astype(int)
    # 참고용 구간화(EDA/시각화, 트리 계열 모델 해석에 유용)
    df["rolling_temp_bin"] = pd.cut(
        df["rolling_temp"],
        bins=[0, 800, 850, 900, 950, 1000, np.inf],
        labels=["<800", "800-850", "850-900", "900-950", "950-1000", ">=1000"],
    )

    # --- 교차항: HSB x 사상압연온도, HSB x 온도임계플래그, HSB x Descaling횟수 ---
    df["hsb_x_rolling_temp"] = df["hsb_applied"] * df["rolling_temp"]
    df["hsb_x_over1000"] = df["hsb_applied"] * df["rolling_temp_over_1000"]
    df["hsb_x_descaling"] = df["hsb_applied"] * df["descaling_count"]

    # --- 판두께 구간화 (후판일수록 방열이 느려 고온 노출시간이 길어질 수 있음) ---
    df["thick_bucket"] = pd.cut(
        df["pt_thick"],
        bins=[0, 20, 40, 60, np.inf],
        labels=["박(<20)", "중(20-40)", "후(40-60)", "극후(>=60)"],
    )

    # --- 가열로 체류시간 대비 온도낙차 비율 (단위시간당 냉각속도 proxy) ---
    df["cooling_rate_proxy"] = df["temp_drop"] / df["fur_total_time"].replace(0, np.nan)
    df["cooling_rate_proxy"] = df["cooling_rate_proxy"].fillna(df["cooling_rate_proxy"].median())

    # --- descaling_count 홀짝 플래그: EDA에서 홀수(5,7,9)=100% 불량 패턴 확인 ---
    #     실제 인과인지, 다른 변수와의 교란(confounding)인지는 미확정 -> 별도 플래그로
    #     남겨 모델/SHAP로 신호 강도를 검증할 수 있게 한다 (임의로 버리지 않음).
    df["descaling_is_odd"] = (df["descaling_count"] % 2 == 1).astype(int)

    # --- 시간 파생 변수: 조업 시간대(시프트) ---
    df["rolling_hour"] = df["rolling_date"].dt.hour
    df["rolling_dow"] = df["rolling_date"].dt.dayofweek

    # --- 타깃 인코딩 ---
    df["target"] = (df["scale"] == "불량").astype(int)

    return df


# ----------------------------------------------------------------------------
# 4. 데이터 분할: 시간 기준 분할 + 그룹(작업조) 누수 방지 옵션
# ----------------------------------------------------------------------------
def time_based_split(df: pd.DataFrame, valid_size: float = 0.15, test_size: float = 0.15):
    """
    rolling_date 오름차순 정렬 후 앞 구간을 train, 뒤 구간을 valid/test로 사용.
    실제 조업 환경처럼 '과거 데이터로 미래를 예측'하는 상황을 재현해
    랜덤 분할 대비 낙관적으로 부풀려진 성능 추정을 방지한다.
    """
    df_sorted = df.sort_values("rolling_date").reset_index(drop=True)
    n = len(df_sorted)
    n_test = int(n * test_size)
    n_valid = int(n * valid_size)
    n_train = n - n_valid - n_test

    train = df_sorted.iloc[:n_train]
    valid = df_sorted.iloc[n_train : n_train + n_valid]
    test = df_sorted.iloc[n_train + n_valid :]

    print(
        f"[시간 분할] train={len(train)} ({train['rolling_date'].min()} ~ {train['rolling_date'].max()}), "
        f"valid={len(valid)} ({valid['rolling_date'].min()} ~ {valid['rolling_date'].max()}), "
        f"test={len(test)} ({test['rolling_date'].min()} ~ {test['rolling_date'].max()})"
    )
    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        print(f"  - {name} 불량률: {part['target'].mean():.1%}")

    return train, valid, test


def group_split_by_work_group(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    같은 작업조(work_group)의 코일이 train/test에 동시에 섞여 들어가면
    '조 특유의 습관/버릇'이 누수(leakage)로 작용할 수 있다.
    작업조 단위로 통째로 분리하는 그룹 분할 옵션 (시간 분할과 별도로 참고용).
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df["work_group"]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    print(
        f"[그룹 분할(work_group)] train={len(train)} (조: {sorted(train['work_group'].unique())}), "
        f"test={len(test)} (조: {sorted(test['work_group'].unique())})"
    )
    return train, test


# ----------------------------------------------------------------------------
# 5. 모델 입력용 최종 컬럼 정리
# ----------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "pt_thick", "pt_width", "pt_length",
    "fur_heat_temp", "fur_heat_time",
    "fur_soak_temp", "fur_soak_time", "fur_total_time",
    "rolling_temp", "descaling_count",
    "temp_drop", "heat_soak_gap", "cooling_rate_proxy",
    "hsb_x_rolling_temp", "hsb_x_descaling",
    "rolling_hour", "rolling_dow",
]
BINARY_FEATURES = [
    "hsb_applied", "rolling_temp_over_1000", "hsb_x_over1000", "descaling_is_odd",
    "rolling_temp_was_missing",
]
CATEGORICAL_FEATURES = [
    "spec_country", "steel_kind", "fur_no", "fur_input_row",
    "rolling_method", "work_group", "thick_bucket",
]
TARGET = "target"


def to_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """수치+이진 피처는 그대로, 범주형은 원-핫 인코딩하여 모델 입력 행렬 생성."""
    cols = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    base = df[cols].copy()
    encoded = pd.get_dummies(base, columns=CATEGORICAL_FEATURES, drop_first=True)
    return encoded


# ----------------------------------------------------------------------------
# 실행 파이프라인
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_raw()
    print(f"원본 데이터: {df.shape}\n")

    df = clean_missing_and_outliers(df)
    df = engineer_features(df)

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[저장] 파생변수 포함 전체 데이터 -> {OUT_PATH} ({df.shape})")

    print("\n=== 시간 기준 분할 (모델 학습/평가에 사용 권장) ===")
    train_t, valid_t, test_t = time_based_split(df)

    print("\n=== 참고: 작업조 그룹 분할 (누수 점검용) ===")
    train_g, test_g = group_split_by_work_group(df)

    print("\n=== 모델 입력 행렬 예시 (train, 시간 기준) ===")
    X_train = to_model_matrix(train_t)
    print(X_train.shape)
    print(X_train.columns.tolist())
