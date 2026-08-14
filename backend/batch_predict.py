"""
CLI로 CSV 파일 전체를 활성 모델에 일괄 채점한다. 대시보드의 "일괄 처리" 버튼과
동일한 batch_scoring.score_dataframe()을 쓰므로 결과가 100% 동일하다 — 서버를
띄우지 않고 터미널에서 바로 확인하고 싶을 때, 또는 큰 파일을 결과 CSV로
남기고 싶을 때 쓴다.

사용법:
  python batch_predict.py [--csv 경로] [--out 결과경로.csv]
"""

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

import prediction_service
from batch_scoring import score_dataframe
from csv_parsing import load_raw_scale_csv
from db import engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSV 파일 전체를 활성 모델로 일괄 채점")
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).resolve().parent.parent / "data" / "raw" / "SCALE불량.csv"),
        help="채점할 CSV 경로 (기본: data/raw/SCALE불량.csv)",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "batch_predict_result.csv"),
        help="결과를 저장할 CSV 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with Session(engine) as session:
        active = prediction_service.get_active_bundle(session)
        if active is None:
            raise SystemExit("활성화된 모델이 없습니다. 먼저 train_model.py를 실행하세요.")
        model_version, bundle = active

    print(f"[모델] {model_version.name} {model_version.version} ({model_version.algorithm})")

    df = load_raw_scale_csv(args.csv)
    print(f"[로드] {args.csv} -> {len(df)}행")

    rows, summary = score_dataframe(df, model_version, bundle)

    print("\n[요약]")
    print(f"  총 {summary['total_rows']}건 중 예측 불량 {summary['predicted_defect_count']}건 "
          f"({summary['predicted_defect_rate']:.1%})")
    if summary["has_actual_labels"]:
        cm = summary["confusion_matrix"]
        print(f"  정확도={summary['accuracy']:.1%}  정밀도={summary['precision']:.1%}  재현율={summary['recall']:.1%}")
        print(f"  혼동행렬: TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']}")

    out_df = pd.DataFrame(rows)
    out_df["fired_rules"] = out_df["fired_rules"].apply(lambda r: ";".join(r))
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n[저장] 위험도 높은 순으로 정렬된 전체 결과 -> {args.out}")

    print("\n[상위 5건 - 고위험]")
    for r in rows[:5]:
        print(f"  {r['plate_no']:12s} prob={r['predicted_prob']:.3f} pred={r['predicted_label']} rules={r['fired_rules']}")


if __name__ == "__main__":
    main()
