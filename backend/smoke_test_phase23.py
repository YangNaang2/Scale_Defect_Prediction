"""Phase 2+3 통합 스모크 테스트: FastAPI TestClient로 실제 엔드포인트를 호출해 검증한다."""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def make_plate(plate_no, thick=32, spec="AB/EH32-TM", country="미국", steel="T"):
    r = client.post(
        "/api/v1/plates",
        json={
            "plate_no": plate_no,
            "spec_long": spec,
            "spec_country": country,
            "steel_kind": steel,
            "pt_thick": thick,
            "pt_width": 3700,
            "pt_length": 15100,
        },
    )
    assert r.status_code == 201, r.text


def make_furnace(plate_no, heat_temp=1150, soak_temp=1140):
    r = client.post(
        "/api/v1/furnace-records",
        json={
            "plate_no": plate_no,
            "fur_no": 1,
            "fur_input_row": 1,
            "fur_heat_temp": heat_temp,
            "fur_heat_time": 116,
            "fur_soak_temp": soak_temp,
            "fur_soak_time": 59,
            "fur_total_time": 259,
            "recorded_at": "2026-08-14T07:00:00+09:00",
        },
    )
    assert r.status_code == 201, r.text


def make_rolling(plate_no, hsb_applied, rolling_temp, descaling=8):
    r = client.post(
        "/api/v1/rolling-records",
        json={
            "plate_no": plate_no,
            "hsb_applied": hsb_applied,
            "rolling_method": "TMCP",
            "rolling_temp": rolling_temp,
            "descaling_count": descaling,
            "work_group": "1조",
            "rolling_date": "2026-08-14T07:20:00+09:00",
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


print("=== 케이스 1: HSB 미적용 (경보+고위험 예측 기대) ===")
make_plate("TEST_HSB_OFF")
make_furnace("TEST_HSB_OFF", heat_temp=1150, soak_temp=1140)
result = make_rolling("TEST_HSB_OFF", hsb_applied=False, rolling_temp=920)
print(result)
assert any(a["rule_code"] == "HSB_NOT_APPLIED" for a in result["alerts"])
assert result["prediction"] is not None
print(f"  -> 예측확률: {result['prediction']['predicted_prob']}, 라벨: {result['prediction']['predicted_label']}")

print("\n=== 케이스 2: 사상압연온도 1000도 초과 (경보 기대) ===")
make_plate("TEST_HOT")
make_furnace("TEST_HOT", heat_temp=1150, soak_temp=1140)
result = make_rolling("TEST_HOT", hsb_applied=True, rolling_temp=1050)
print(result)
assert any(a["rule_code"] == "ROLLING_TEMP_OVER_1000" for a in result["alerts"])

print("\n=== 케이스 3: 온도낙차 부족 (경보 기대) ===")
make_plate("TEST_COOLDROP")
make_furnace("TEST_COOLDROP", heat_temp=1150, soak_temp=1140)
result = make_rolling("TEST_COOLDROP", hsb_applied=True, rolling_temp=980)  # 낙차 160도
print(result)
assert any(a["rule_code"] == "TEMP_DROP_INSUFFICIENT" for a in result["alerts"])

print("\n=== 케이스 4: 정상 조업 (경보 없음, 저위험 예측 기대) ===")
make_plate("TEST_CLEAN")
make_furnace("TEST_CLEAN", heat_temp=1150, soak_temp=1140)
result = make_rolling("TEST_CLEAN", hsb_applied=True, rolling_temp=900)  # 낙차 240도
print(result)
assert result["alerts"] == []
print(f"  -> 예측확률: {result['prediction']['predicted_prob']}, 라벨: {result['prediction']['predicted_label']}")

print("\n=== GET /predictions/{plate_no} 재조회 ===")
r = client.get("/api/v1/predictions/TEST_HSB_OFF")
assert r.status_code == 200, r.text
print(r.json())

print("\n=== GET /alerts?status=open ===")
r = client.get("/api/v1/alerts", params={"status": "open"})
assert r.status_code == 200
open_alerts = r.json()
print(f"  미해결 경보 {len(open_alerts)}건 (신규 4건 이상 포함되어야 함)")
assert len(open_alerts) >= 3  # HSB_OFF, HOT, COOLDROP 3건 이상

print("\n=== GET /dashboard/summary ===")
r = client.get("/api/v1/dashboard/summary")
assert r.status_code == 200
print(r.json())

print("\n[전체 통과] Phase 2+3 스모크 테스트 성공")
