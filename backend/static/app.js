import {
  formatPercent,
  formatProb,
  formatDateTime,
  severityClass,
  ruleCodeLabel,
  shapBarWidth,
  riskLabel,
  matchLabel,
  extractModelMetrics,
  formatParams,
  compareModelRank,
} from "./format.mjs";

const BATCH_TABLE_ROW_LIMIT = 200;

const API = "/api/v1";

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${body}`);
  }
  if (res.status === 204) return null;
  return res.json();
}

// ------------------------------------------------------------------- theme
// index.html의 인라인 스크립트가 저장된 선택을 첫 페인트 전에 이미 반영해뒀으므로,
// 여기서는 버튼 라벨 동기화와 클릭 시 전환/저장만 담당한다.
const THEME_KEY = "scale-dashboard-theme";

function currentTheme() {
  const explicit = document.documentElement.getAttribute("data-theme");
  if (explicit === "light" || explicit === "dark") return explicit;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function syncToggleLabel(theme) {
  document.getElementById("theme-toggle").textContent = theme === "dark" ? "☀️ 라이트 모드" : "🌙 다크 모드";
}

// 클릭했을 때만 명시적으로 저장/고정한다 — 그냥 열어보기만 했을 때는 시스템 설정을
// 계속 따라가야 하므로(고정해버리면 OS 테마를 바꿔도 반영이 안 됨), 로드 시점엔
// 라벨만 현재 유효 테마에 맞추고 localStorage에는 쓰지 않는다.
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem(THEME_KEY, theme);
  syncToggleLabel(theme);
}

document.getElementById("theme-toggle").addEventListener("click", () => {
  applyTheme(currentTheme() === "dark" ? "light" : "dark");
});
syncToggleLabel(currentTheme());

// ------------------------------------------------------------------ models
// 지금까지 학습으로 등록된 모든 모델(승자만이 아니라 후보 전체)을 성능·파라미터와
// 함께 보여준다 — 어떤 걸 실제로 서빙할지는 사람이 여기서 직접 고른다.
async function loadModels() {
  const tbody = document.querySelector("#models-table tbody");
  try {
    const models = await fetchJSON(`${API}/models`);
    if (models.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="empty">등록된 모델이 없습니다. train_model.py를 먼저 실행하세요.</td></tr>`;
      return;
    }

    const withMetrics = models.map((m) => ({ ...m, ...extractModelMetrics(m) }));
    withMetrics.sort(compareModelRank);

    tbody.innerHTML = withMetrics
      .map((m, i) => {
        const v = m.valid ?? {};
        const isBest = i === 0;
        return `
      <tr class="${isBest ? "best-row" : ""}">
        <td class="num" title="${m.version}">${formatDateTime(m.trained_at)}</td>
        <td>${m.algorithm} ${isBest ? '<span class="pill pill-best">BEST</span>' : ""}</td>
        <td class="num">${v.precision ?? "—"}</td>
        <td class="num">${v.recall ?? "—"}</td>
        <td class="num">${v.f1 ?? "—"}</td>
        <td class="num"><strong>${v.auc ?? "—"}</strong></td>
        <td class="num truncate" style="font-size:11.5px;" title="${formatParams(v.best_params)}">${formatParams(v.best_params)}</td>
        <td>${m.is_active ? '<span class="pill sev-info">활성</span>' : '<span class="muted">비활성</span>'}</td>
        <td>${m.is_active ? "" : `<button data-model-id="${m.id}" class="activate-btn">활성화</button>`}</td>
      </tr>`;
      })
      .join("");
    tbody.querySelectorAll(".activate-btn").forEach((btn) => {
      btn.addEventListener("click", () => activateModel(btn.dataset.modelId));
    });
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="8" class="empty">모델 목록을 불러오지 못했습니다: ${e.message}</td></tr>`;
  }
}

async function activateModel(modelId) {
  try {
    await fetchJSON(`${API}/models/${modelId}/activate`, { method: "POST" });
    await loadModels();
  } catch (e) {
    alert(`활성화 실패: ${e.message}`);
  }
}

// --------------------------------------------------------------- 일괄 처리
async function submitBatch(ev) {
  ev.preventDefault();
  const fileInput = document.getElementById("batch-file");
  const resultEl = document.getElementById("batch-result");
  const file = fileInput.files[0];
  if (!file) return;

  resultEl.innerHTML = `<div class="muted">채점 중… (건수가 많으면 몇 초 걸릴 수 있습니다)</div>`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API}/batch-predict`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const data = await res.json();
    renderBatchResult(data);
  } catch (e) {
    resultEl.innerHTML = `<div class="empty">채점 실패: ${e.message}</div>`;
  }
}

function renderBatchResult(data) {
  const { summary, rows } = data;
  const resultEl = document.getElementById("batch-result");

  const metricTiles = summary.has_actual_labels
    ? `
      <div class="tile"><div class="v">${formatPercent(summary.accuracy)}</div><div class="l">정확도</div></div>
      <div class="tile"><div class="v">${formatPercent(summary.precision)}</div><div class="l">정밀도</div></div>
      <div class="tile"><div class="v">${formatPercent(summary.recall)}</div><div class="l">재현율</div></div>
      <div class="tile">
        <div class="v num" style="font-size:14px;">
          TP ${summary.confusion_matrix.tp} · FP ${summary.confusion_matrix.fp}<br>
          FN ${summary.confusion_matrix.fn} · TN ${summary.confusion_matrix.tn}
        </div>
        <div class="l">혼동행렬</div>
      </div>`
    : `<div class="tile"><div class="v muted" style="font-size:13px;">실제 판정 없음</div><div class="l">CSV에 scale 컬럼이 없어 정오 비교는 생략</div></div>`;

  const shown = rows.slice(0, BATCH_TABLE_ROW_LIMIT);
  const truncatedNote =
    rows.length > BATCH_TABLE_ROW_LIMIT
      ? `<div class="muted" style="margin-top:8px;">위험도 상위 ${BATCH_TABLE_ROW_LIMIT}건만 표시 (전체 ${rows.length}건 중)</div>`
      : "";

  resultEl.innerHTML = `
    <div class="tiles">
      <div class="tile"><div class="v">${summary.total_rows}</div><div class="l">전체 건수</div></div>
      <div class="tile"><div class="v">${summary.predicted_defect_count} (${formatPercent(summary.predicted_defect_rate)})</div><div class="l">예측 불량 건수/비율</div></div>
      ${metricTiles}
    </div>
    <div class="tbl-wrap" style="margin-top:14px;"><table>
      <thead><tr><th>코일</th><th>예측확률</th><th>예측</th>${summary.has_actual_labels ? "<th>실제</th><th>일치</th>" : ""}<th>걸린 규칙</th></tr></thead>
      <tbody>
        ${shown
          .map((r) => {
            const risk = riskLabel(r.predicted_prob);
            const actualCols = summary.has_actual_labels
              ? `<td>${r.actual_label ? "불량" : "양품"}</td><td class="${matchLabel(r.correct).cls}">${matchLabel(r.correct).text}</td>`
              : "";
            return `
          <tr>
            <td class="num">${r.plate_no}</td>
            <td class="num">${formatProb(r.predicted_prob)}</td>
            <td class="${risk.cls}">${r.predicted_label ? "불량" : "양품"}</td>
            ${actualCols}
            <td>${r.fired_rules.map(ruleCodeLabel).join(", ") || '<span class="muted">-</span>'}</td>
          </tr>`;
          })
          .join("")}
      </tbody>
    </table></div>
    ${truncatedNote}
  `;
}

// ------------------------------------------------------ 예측 결과 패널(공용)
// 코일 상세 조회와 임의값 예측(what-if)이 동일한 SHAP 막대/위험도 UI를 쓴다.
function renderPredictionPanel(prediction, alerts, { noAlertText } = {}) {
  const risk = riskLabel(prediction.predicted_prob);
  const maxAbs = Math.max(...prediction.top_features.map((f) => Math.abs(f.shap_value)), 0.0001);

  const alertsHtml = alerts.length
    ? alerts
        .map((a) => `<span class="pill ${severityClass(a.severity)}" style="margin-right:6px;">${ruleCodeLabel(a.rule_code)}</span>`)
        .join("")
    : `<span class="muted">${noAlertText ?? "해당 코일에 걸린 규칙 알림 없음"}</span>`;

  return `
    <div class="detail-summary">
      <div><div class="prob num">${formatProb(prediction.predicted_prob)}</div><div class="muted">Scale 불량 예측 확률</div></div>
      <div class="${risk.cls}">${risk.text}</div>
      <div class="muted">모델: ${prediction.model_version}</div>
    </div>
    <div style="margin:10px 0;">${alertsHtml}</div>
    <div class="muted">SHAP 상위 기여 피처 (양수=불량 확률 증가, 음수=감소)</div>
    <div class="shap-bar-wrap">
      ${prediction.top_features
        .map((f) => {
          const width = shapBarWidth(f.shap_value, maxAbs);
          const cls = f.shap_value >= 0 ? "pos" : "neg";
          return `
        <div class="shap-row">
          <span class="num">${f.feature_name}</span>
          <span class="shap-track"><span class="shap-fill ${cls}" style="width:${width}%"></span></span>
          <span class="num">${f.shap_value.toFixed(3)}</span>
        </div>`;
        })
        .join("")}
    </div>
  `;
}

// -------------------------------------------------------------- plate 상세
async function searchPlate() {
  const plateNo = document.getElementById("plate-search").value.trim();
  const el = document.getElementById("plate-detail");
  if (!plateNo) return;
  el.innerHTML = `<div class="muted">조회 중…</div>`;

  try {
    const [prediction, alerts] = await Promise.all([
      fetchJSON(`${API}/predictions/${encodeURIComponent(plateNo)}`).catch(() => null),
      fetchJSON(`${API}/alerts?status=all&plate_no=${encodeURIComponent(plateNo)}`).catch(() => []),
    ]);

    if (!prediction) {
      el.innerHTML = `<div class="empty">해당 코일의 예측 결과가 없습니다 (아직 압연 데이터가 적재되지 않았거나 활성 모델이 없습니다).</div>`;
      return;
    }

    el.innerHTML = renderPredictionPanel(prediction, alerts);
  } catch (e) {
    el.innerHTML = `<div class="empty">조회 실패: ${e.message}</div>`;
  }
}

// -------------------------------------------------------- 임의값 예측(what-if)
async function submitAdhocPredict(ev) {
  ev.preventDefault();
  const el = document.getElementById("adhoc-result");
  const form = document.getElementById("adhoc-form");
  el.innerHTML = `<div class="muted">예측 중…</div>`;

  const raw = Object.fromEntries(new FormData(form).entries());
  const rollingTemp = raw.rolling_temp.trim();

  const payload = {
    spec_country: raw.spec_country,
    steel_kind: raw.steel_kind,
    pt_thick: Number(raw.pt_thick),
    pt_width: Number(raw.pt_width),
    pt_length: Number(raw.pt_length),
    fur_no: Number(raw.fur_no),
    fur_input_row: Number(raw.fur_input_row),
    fur_heat_temp: Number(raw.fur_heat_temp),
    fur_heat_time: Number(raw.fur_heat_time),
    fur_soak_temp: Number(raw.fur_soak_temp),
    fur_soak_time: Number(raw.fur_soak_time),
    fur_total_time: Number(raw.fur_total_time),
    hsb_applied: raw.hsb_applied === "true",
    rolling_method: raw.rolling_method,
    rolling_temp: rollingTemp === "" ? null : Number(rollingTemp),
    descaling_count: Number(raw.descaling_count),
    work_group: raw.work_group,
  };

  try {
    const data = await fetchJSON(`${API}/predict-adhoc`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    el.innerHTML = renderPredictionPanel(data.prediction, data.alerts, {
      noAlertText: "걸린 규칙 없음",
    });
  } catch (e) {
    el.innerHTML = `<div class="empty">예측 실패: ${e.message}</div>`;
  }
}

// --------------------------------------------------------------------- init
document.getElementById("search-plate").addEventListener("click", searchPlate);
document.getElementById("plate-search").addEventListener("keydown", (e) => {
  if (e.key === "Enter") searchPlate();
});
document.getElementById("batch-form").addEventListener("submit", submitBatch);
document.getElementById("adhoc-form").addEventListener("submit", submitAdhocPredict);

loadModels();
