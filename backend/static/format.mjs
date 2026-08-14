// 순수 포맷팅 함수 모음. DOM에 의존하지 않아 브라우저(app.js)와 Node(format.test.mjs)
// 양쪽에서 동일하게 import해 쓸 수 있다 — 별도 빌드 도구 없이도 로직을 유닛테스트하기 위함.

export function formatPercent(value, digits = 1) {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatProb(value) {
  if (value === null || value === undefined) return "—";
  return value.toFixed(3);
}

export function severityLabel(severity) {
  const map = { CRITICAL: "긴급", WARNING: "주의", INFO: "정보" };
  return map[severity] ?? severity;
}

export function severityClass(severity) {
  const map = { CRITICAL: "sev-critical", WARNING: "sev-warning", INFO: "sev-info" };
  return map[severity] ?? "sev-info";
}

export function ruleCodeLabel(ruleCode) {
  const map = {
    HSB_NOT_APPLIED: "HSB 미적용",
    ROLLING_TEMP_OVER_1000: "사상압연온도 1000℃ 초과",
    TEMP_DROP_INSUFFICIENT: "온도낙차 부족",
  };
  return map[ruleCode] ?? ruleCode;
}

export function shapBarWidth(shapValue, maxAbs) {
  if (!maxAbs) return 0;
  return Math.min(100, Math.round((Math.abs(shapValue) / maxAbs) * 100));
}

export function formatDateTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function matchLabel(correct) {
  if (correct === null || correct === undefined) return { text: "—", cls: "" };
  return correct ? { text: "일치", cls: "risk-low" } : { text: "불일치", cls: "risk-high" };
}

// model_version.metrics는 두 가지 모양이 섞여 있다:
//  - 신형(v1.4+): {"valid": {precision,recall,f1,auc,best_params,cv_best_auc}, "test": {...}}
//  - 구형(v1.1~1.3): {"valid_comparison": {알고리즘명: {precision,...}}, "selected":.., "test": {...}}
// 화면에서는 이 차이를 신경 쓰지 않고 정규화된 값만 쓰도록 여기서 흡수한다.
export function extractModelMetrics(modelVersion) {
  const metrics = modelVersion.metrics ?? {};
  const valid = metrics.valid ?? metrics.valid_comparison?.[modelVersion.algorithm] ?? {};
  const test = metrics.test ?? null;
  return { valid, test };
}

export function formatParams(params) {
  // best_params가 없는 건 그리드서치 이전(v1.1.0/v1.2.0처럼 고정 하이퍼파라미터로
  // 학습한 구버전)이라 그런 것이지 데이터 누락이 아니므로, 이유가 드러나게 표시한다.
  if (!params || Object.keys(params).length === 0) return "그리드서치 이전 버전";
  return Object.entries(params)
    .map(([k, v]) => `${k}=${v}`)
    .join(", ");
}

// 동점 처리 순서: AUC -> F1 -> 정밀도 -> 재현율 -> 교차검증 AUC(cv_best_auc).
// F1/정밀도/재현율은 소수 3자리로 반올림해서 비교한다 — 학습 실행마다 데이터가
// 1~2행씩 달라 넷째 자리에서만 갈리는 잡음성 차이로 "최고 성능"이 뒤집히는 걸
// 막기 위함(예: 그리드서치도 안 거친 구버전이 우연히 0.0001 앞서서 1등이 되면 안 됨).
// 그 잡음 이하로도 동률이면 5-fold 교차검증 점수(cv_best_auc, 구버전엔 없어 0)로
// 최종 결판 — 그리드서치를 거친 모델이 항상 우선한다.
const round3 = (x) => Math.round((x ?? 0) * 1000) / 1000;

export function modelRankKey(modelWithMetrics) {
  const v = modelWithMetrics.valid ?? {};
  return [v.auc ?? 0, round3(v.f1), round3(v.precision), round3(v.recall), v.cv_best_auc ?? 0];
}

export function compareModelRank(a, b) {
  const ra = modelRankKey(a);
  const rb = modelRankKey(b);
  for (let i = 0; i < ra.length; i++) {
    if (rb[i] !== ra[i]) return rb[i] - ra[i];
  }
  return 0;
}

export function riskLabel(prob) {
  if (prob === null || prob === undefined) return { text: "N/A", cls: "risk-none" };
  if (prob >= 0.7) return { text: "고위험", cls: "risk-high" };
  if (prob >= 0.3) return { text: "주의", cls: "risk-mid" };
  return { text: "저위험", cls: "risk-low" };
}
