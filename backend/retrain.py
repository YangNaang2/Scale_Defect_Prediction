"""
Phase 5: 재학습 파이프라인

quality_inspection에 새로 쌓인 실측 라벨(피드백 루프)을 포함해 train_model.py의
학습 로직을 그대로 재사용해 재학습한다. 후보 6종을 전부 새 버전으로 등록하고,
이미 활성 모델이 있는 상태(재학습)라면 전부 비활성으로 등록해 대시보드
"모델 관리"에서 사람이 성능·파라미터를 보고 직접 활성화하게 한다.
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

import train_model


def main() -> None:
    print("[재학습 시작] 최신 quality_inspection 라벨을 포함해 재학습합니다.")
    registered = train_model.main()

    print("\n[등록된 모델 버전]")
    for r in registered:
        marker = " (활성)" if r["is_active"] else ""
        print(f"  id={r['id']:<4} {r['algorithm']:<18} {r['version']}{marker}")

    if not any(r["is_active"] for r in registered):
        print("\n모두 비활성으로 등록되었습니다. 대시보드 '모델 관리'에서 성능/파라미터를 비교해 활성화하세요.")


if __name__ == "__main__":
    main()
