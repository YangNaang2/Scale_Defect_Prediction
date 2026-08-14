"""
일괄 예측(batch scoring) — SCALE불량.csv와 같은 형식의 CSV 파일을 통째로 읽어
활성 모델로 전체 행을 한 번에 채점한다. 결과를 DB에 기록하는 실시간 파이프라인
(prediction_service.py)과 달리, 이 모듈은 "이 파일을 지금 모델에 돌려보면
어떤 결과가 나오나"를 확인하기 위한 것으로 DB에 아무것도 쓰지 않는다.

app.py의 POST /api/v1/batch-predict 와 batch_predict.py(CLI)가 함께 쓴다.
"""

import pandas as pd

import features
from rules import evaluate_rules_from_raw


def _resolve_rolling_temp_series(df: pd.DataFrame, bundle: dict) -> pd.Series:
    group_median = bundle["rolling_temp_group_median"]
    global_median = bundle["rolling_temp_global_median"]

    def resolve(row):
        temp = row["rolling_temp"]
        if pd.notna(temp) and temp > 0:
            return float(temp)
        key = f"{row['steel_kind']}|{row['rolling_method_norm']}"
        return group_median.get(key, global_median)

    return df.apply(resolve, axis=1)


def score_dataframe(df: pd.DataFrame, model_version, bundle: dict) -> tuple[list[dict], dict]:
    """
    df: SCALE불량.csv와 동일한 컬럼(plate_no, spec_long, spec_country, steel_kind,
        pt_thick/width/length, hsb, fur_no, fur_input_row, fur_*_temp/time,
        rolling_method, rolling_temp, descaling_count, work_group, rolling_date,
        선택적으로 scale('양품'/'불량'))을 가진 DataFrame — csv_parsing.load_raw_scale_csv()
        결과와 동일한 형태.

    반환: (행별 결과 리스트, 요약 dict)
    """
    df = df.copy()
    df["hsb_applied"] = df["hsb"] == "적용"
    df["rolling_method_norm"] = df["rolling_method"].apply(
        lambda v: "TMCP" if str(v).startswith("TMCP") else "CR"
    )
    df["rolling_temp_resolved"] = _resolve_rolling_temp_series(df, bundle)

    engineered_rows = []
    for row in df.itertuples():
        raw = {
            "pt_thick": row.pt_thick,
            "pt_width": row.pt_width,
            "pt_length": row.pt_length,
            "spec_country": row.spec_country,
            "steel_kind": row.steel_kind,
            "fur_no": row.fur_no,
            "fur_input_row": row.fur_input_row,
            "fur_heat_temp": row.fur_heat_temp,
            "fur_heat_time": row.fur_heat_time,
            "fur_soak_temp": row.fur_soak_temp,
            "fur_soak_time": row.fur_soak_time,
            "fur_total_time": row.fur_total_time,
            "hsb_applied": row.hsb_applied,
            "rolling_method": row.rolling_method_norm,
            "descaling_count": row.descaling_count,
            "work_group": row.work_group,
            "rolling_date": row.rolling_date,
            "rolling_temp": row.rolling_temp_resolved,
        }
        engineered_rows.append(features.engineer_row(raw))

    engineered_df = pd.DataFrame(engineered_rows)
    encoded = pd.get_dummies(engineered_df, columns=features.CATEGORICAL_FEATURES)
    aligned = features.align_to_model_columns(encoded, bundle["model_columns"])

    model = bundle["model"]
    probs = model.predict_proba(aligned)[:, 1]
    threshold = float(model_version.threshold)
    preds = probs >= threshold

    has_actual = "scale" in df.columns
    results = []
    tp = fp = tn = fn = 0

    for i, row in enumerate(df.itertuples()):
        fired_rules = [
            r.rule_code
            for r in evaluate_rules_from_raw(
                {
                    "fur_soak_temp": row.fur_soak_temp,
                    "hsb_applied": row.hsb_applied,
                    "rolling_temp": row.rolling_temp_resolved,
                }
            )
        ]

        entry = {
            "plate_no": row.plate_no,
            "predicted_prob": round(float(probs[i]), 4),
            "predicted_label": bool(preds[i]),
            "fired_rules": fired_rules,
        }

        if has_actual:
            actual_label = row.scale == "불량"
            entry["actual_label"] = actual_label
            entry["correct"] = bool(actual_label == preds[i])
            if actual_label and preds[i]:
                tp += 1
            elif not actual_label and preds[i]:
                fp += 1
            elif not actual_label and not preds[i]:
                tn += 1
            else:
                fn += 1

        results.append(entry)

    n = len(results)
    predicted_defect_count = int(preds.sum())
    summary = {
        "total_rows": n,
        "predicted_defect_count": predicted_defect_count,
        "predicted_defect_rate": round(predicted_defect_count / n, 4) if n else 0.0,
        "model_version": f"{model_version.name} {model_version.version}",
        "has_actual_labels": has_actual,
    }

    if has_actual:
        accuracy = (tp + tn) / n if n else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        summary.update(
            {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            }
        )

    # 위험도 높은 순으로 정렬 — 현장에서 가장 먼저 봐야 할 코일이 위로 오게
    results.sort(key=lambda r: r["predicted_prob"], reverse=True)

    return results, summary
