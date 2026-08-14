"""
레거시 SCALE불량.csv 형식(가열로+압연이 한 행에 뒤섞인 플랫 파일) 공용 파서.
migrate_from_csv.py(DB 적재)와 batch_scoring.py(일괄 예측)가 함께 쓴다.
"""

import re

import pandas as pd


def extract_int(text) -> int:
    """'1호기', '2열' 같은 문자열에서 선행 숫자를 뽑는다."""
    match = re.match(r"\d+", str(text))
    if not match:
        raise ValueError(f"숫자를 추출할 수 없는 값: {text!r}")
    return int(match.group())


def to_rolling_method(raw: str) -> str:
    if raw.startswith("TMCP"):
        return "TMCP"
    if raw.startswith("CR"):
        return "CR"
    raise ValueError(f"알 수 없는 rolling_method 값: {raw!r}")


def load_raw_scale_csv(path_or_buffer) -> pd.DataFrame:
    """cp949 인코딩을 우선 시도하고, 실패하면 utf-8로 재시도한다."""
    try:
        df = pd.read_csv(path_or_buffer, encoding="cp949")
    except UnicodeDecodeError:
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
        df = pd.read_csv(path_or_buffer, encoding="utf-8")

    df["rolling_date"] = pd.to_datetime(df["rolling_date"], format="%d%b%Y:%H:%M:%S")
    return df
