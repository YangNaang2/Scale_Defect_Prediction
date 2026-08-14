// Scale 예측 시스템 대시보드 드라이버.
// FastAPI 서버(app.py)가 이미 http://127.0.0.1:8000 에 떠 있다고 가정하고,
// (1) API로 예측 결과가 있는 코일을 하나 시드한 뒤 (2) 헤드리스 Chromium으로
// /app/ 대시보드를 열어 모델 관리(성능·파라미터 비교) / 코일 상세(SHAP) / 일괄
// 처리 / 임의값 예측(What-if) 폼까지 실제로 조작하며 스크린샷을 남긴다.
// (실시간 KPI 타일과 미해결 알림 목록 섹션은 실사용성이 낮다고 판단해 제거했다 —
//  규칙 위반 자체는 코일 상세 조회/일괄 처리 결과의 alerts에서 여전히 보인다.)
//
// 사용법: node dashboard_driver.mjs [baseUrl]
//   기본 baseUrl = http://127.0.0.1:8000
// 스크린샷은 이 스크립트와 같은 폴더의 screenshots/ 에 저장된다.

import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const BASE = process.argv[2] ?? "http://127.0.0.1:8000";
const HERE = dirname(fileURLToPath(import.meta.url));
const SHOT_DIR = join(HERE, "screenshots");
const PLATE_NO = `RUN_SKILL_${Date.now()}`;
// backend/.claude/skills/run-scale-backend/ -> 4단계 위 리포지토리 루트의 data/raw/SCALE불량.csv
const SAMPLE_CSV = join(HERE, "..", "..", "..", "..", "data", "raw", "SCALE불량.csv");

async function api(path, opts) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) throw new Error(`${opts?.method ?? "GET"} ${path} -> ${res.status}: ${await res.text()}`);
  return res.status === 204 ? null : res.json();
}

async function seedPlateWithPrediction() {
  await api("/api/v1/plates", {
    method: "POST",
    body: JSON.stringify({
      plate_no: PLATE_NO,
      spec_long: "AB/EH32-TM",
      spec_country: "미국",
      steel_kind: "T",
      pt_thick: 32,
      pt_width: 3700,
      pt_length: 15100,
    }),
  });
  await api("/api/v1/furnace-records", {
    method: "POST",
    body: JSON.stringify({
      plate_no: PLATE_NO,
      fur_no: 1,
      fur_input_row: 1,
      fur_heat_temp: 1150,
      fur_heat_time: 116,
      fur_soak_temp: 1140,
      fur_soak_time: 59,
      fur_total_time: 259,
      recorded_at: "2026-08-14T07:00:00+09:00",
    }),
  });
  // HSB 미적용으로 넣어서 규칙엔진 경보 + 고위험 예측이 함께 뜨는 걸 확인한다.
  return api("/api/v1/rolling-records", {
    method: "POST",
    body: JSON.stringify({
      plate_no: PLATE_NO,
      hsb_applied: false,
      rolling_method: "TMCP",
      rolling_temp: 920,
      descaling_count: 8,
      work_group: "1조",
      rolling_date: "2026-08-14T07:20:00+09:00",
    }),
  });
}

async function main() {
  await mkdir(SHOT_DIR, { recursive: true });

  console.log(`[seed] creating ${PLATE_NO} via API...`);
  const ingestResult = await seedPlateWithPrediction();
  console.log(`[seed] alerts=${ingestResult.alerts.length}, prediction_prob=${ingestResult.prediction?.predicted_prob}`);

  const consoleErrors = [];
  const browser = await chromium.launch();
  const page = await browser.newPage();
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  page.on("pageerror", (err) => consoleErrors.push(`pageerror: ${err.message}`));

  await page.goto(`${BASE}/app/`, { waitUntil: "networkidle" });
  await page.waitForSelector("#models-table tbody tr");
  await page.screenshot({ path: join(SHOT_DIR, "01_models.png"), fullPage: true });

  await page.fill("#plate-search", PLATE_NO);
  await page.click("#search-plate");
  await page.waitForSelector("#plate-detail .detail-summary", { timeout: 10000 });
  await page.screenshot({ path: join(SHOT_DIR, "02_plate_detail.png"), fullPage: true });

  console.log(`[batch] scoring ${SAMPLE_CSV} via the 일괄 처리 upload form...`);
  await page.setInputFiles("#batch-file", SAMPLE_CSV);
  await page.click("#batch-form button[type=submit]");
  await page.waitForSelector("#batch-result .tiles", { timeout: 30000 });
  await page.screenshot({ path: join(SHOT_DIR, "03_batch_result.png"), fullPage: true });

  console.log("[adhoc] submitting the what-if form with HSB set to 미적용...");
  await page.selectOption("#ah-hsb", "false");
  await page.click('#adhoc-form button[type=submit]');
  await page.waitForSelector("#adhoc-result .pill", { timeout: 10000 });
  await page.screenshot({ path: join(SHOT_DIR, "04_adhoc_result.png"), fullPage: true });

  await browser.close();

  console.log(`[screenshots] ${SHOT_DIR}`);
  console.log(`[console errors] ${JSON.stringify(consoleErrors)}`);
  if (consoleErrors.length > 0) {
    console.log("  (참고: 예측 이력이 없는 plate_no를 조회할 때 나는 404 하나는 정상 — app.js가 catch해서 빈 상태로 표시함)");
  }
  console.log(`[seeded plate] ${PLATE_NO} — 정리하려면 backend/scale_system.db에서 직접 삭제 (DELETE API 없음)`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
