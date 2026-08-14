// 순수 함수 유닛 테스트 (Node로 직접 실행: node format.test.mjs). 브라우저/DOM 없이
// app.js가 쓰는 포맷팅 로직만 따로 검증한다.
import assert from "node:assert/strict";
import {
  formatPercent,
  formatProb,
  severityLabel,
  severityClass,
  ruleCodeLabel,
  shapBarWidth,
  formatDateTime,
  riskLabel,
  matchLabel,
  extractModelMetrics,
  formatParams,
  compareModelRank,
} from "./format.mjs";

assert.equal(formatPercent(0.5), "50.0%");
assert.equal(formatPercent(null), "—");
assert.equal(formatProb(0.93099), "0.931");
assert.equal(severityLabel("CRITICAL"), "긴급");
assert.equal(severityClass("WARNING"), "sev-warning");
assert.equal(ruleCodeLabel("HSB_NOT_APPLIED"), "HSB 미적용");
assert.equal(ruleCodeLabel("UNKNOWN_CODE"), "UNKNOWN_CODE");
assert.equal(shapBarWidth(0.15, 0.3), 50);
assert.equal(shapBarWidth(0.3, 0), 0);
assert.equal(riskLabel(0.93).cls, "risk-high");
assert.equal(riskLabel(0.5).cls, "risk-mid");
assert.equal(riskLabel(0.04).cls, "risk-low");
assert.equal(formatDateTime(null), "—");
assert.match(formatDateTime("2026-08-14T07:20:00+09:00"), /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$/);
assert.equal(matchLabel(true).text, "일치");
assert.equal(matchLabel(false).text, "불일치");
assert.equal(matchLabel(null).text, "—");

// 신형 metrics 모양
const newShape = { algorithm: "RandomForest", metrics: { valid: { precision: 0.98, auc: 1.0, best_params: { max_depth: 8 } }, test: { auc: 0.99 } } };
assert.equal(extractModelMetrics(newShape).valid.auc, 1.0);
assert.equal(extractModelMetrics(newShape).test.auc, 0.99);

// 구형 metrics 모양 (valid_comparison에 알고리즘명으로 중첩)
const oldShape = { algorithm: "RandomForest", metrics: { valid_comparison: { RandomForest: { precision: 0.97, auc: 1.0 } }, selected: "RandomForest", test: { auc: 1.0 } } };
assert.equal(extractModelMetrics(oldShape).valid.precision, 0.97);

// metrics 자체가 비어있어도 안 죽어야 함
assert.deepEqual(extractModelMetrics({ algorithm: "X", metrics: {} }).valid, {});

assert.equal(formatParams({ max_depth: 8, n_estimators: 100 }), "max_depth=8, n_estimators=100");
assert.equal(formatParams(null), "그리드서치 이전 버전");
assert.equal(formatParams({}), "그리드서치 이전 버전");

// AUC/F1/정밀도/재현율이 전부 동률이면 cv_best_auc로 최종 결판
const tiedButHigherCv = { valid: { auc: 1, f1: 0.99, precision: 0.98, recall: 1, cv_best_auc: 0.995 } };
const tiedLowerCv = { valid: { auc: 1, f1: 0.99, precision: 0.98, recall: 1, cv_best_auc: 0.99 } };
assert.ok(compareModelRank(tiedButHigherCv, tiedLowerCv) < 0); // 첫 인자가 더 좋으면 음수(앞으로 정렬)
const clearlyWorse = { valid: { auc: 0.9, f1: 0.8, precision: 0.7, recall: 0.9, cv_best_auc: 0.5 } };
const sorted = [clearlyWorse, tiedLowerCv, tiedButHigherCv].sort(compareModelRank);
assert.equal(sorted[0], tiedButHigherCv);
assert.equal(sorted[2], clearlyWorse);
// 레거시(cv_best_auc 없음) 모델도 죽지 않고 뒤로 밀림
const legacyNoCv = { valid: { auc: 1, f1: 0.99, precision: 0.98, recall: 1 } };
assert.equal([legacyNoCv, tiedButHigherCv].sort(compareModelRank)[0], tiedButHigherCv);

// 넷째 자리 잡음 차이로는 그리드서치 안 거친 레거시가 이기면 안 됨
// (legacy f1=0.9929, tuned f1=0.9928 — 반올림 전이면 legacy가 이기지만, 3자리 반올림 후
//  동률이 되어 cv_best_auc가 있는 tuned 쪽이 최종 승리해야 한다)
const legacyTinyEdgeNoCv = { valid: { auc: 1, f1: 0.9929, precision: 0.9859, recall: 1 } };
const tunedSlightlyLowerButHasCv = { valid: { auc: 1, f1: 0.9928, precision: 0.9857, recall: 1, cv_best_auc: 0.995 } };
assert.equal(
  [legacyTinyEdgeNoCv, tunedSlightlyLowerButHasCv].sort(compareModelRank)[0],
  tunedSlightlyLowerButHasCv
);

console.log("[전체 통과] format.mjs 유닛 테스트 성공");
