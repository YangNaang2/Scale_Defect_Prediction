---
name: run-scale-backend
description: Build, run, and drive the Scale 불량 예측 시스템 FastAPI backend (API + rule engine + multi-model training/registry + batch CSV scoring + ad-hoc what-if prediction + static dashboard). Use when asked to start the backend, run its API/rule-engine smoke test, train/retrain/compare models, activate a model version, batch-score a CSV, run a what-if prediction, or take a screenshot of the dashboard.
---

FastAPI app (`app.py`) that serves both the REST API (`/api/v1/*`) and a
static dashboard (`/app/`). It has two independent interaction layers,
each with its own driver:

1. **Backend logic** (ingestion, rule engine, ML prediction+SHAP) — most
   of this project's real complexity lives here. Drive it directly with
   `smoke_test_phase23.py` (uses FastAPI's in-process `TestClient`, no
   server process needed).
2. **Dashboard UI** — drive it with
   `.claude/skills/run-scale-backend/dashboard_driver.mjs` (Playwright,
   headless Chromium) against a running `uvicorn` server.

All paths below are relative to `backend/` (this skill's project root),
except the driver itself which lives at
`.claude/skills/run-scale-backend/`.

## Prerequisites

Verified on Windows with a real (non-Microsoft-Store-stub) Python 3.13
and Node 24.18.0 + npm. There is no `apt-get` here — this project has no
Linux-specific dependency, but on Windows make sure `python` on PATH is
the actual interpreter, not the Microsoft Store alias (that alias exits
immediately with "Python was not found..."). If unsure, resolve it
explicitly:

```bash
where python   # pick the real one, e.g. C:\...\Python313\python.exe — NOT \WindowsApps\python.exe
```

## Setup

`requirements.txt` lives at the repo root (one level above `backend/`):

```bash
cd backend
python -m pip install -r ../requirements.txt
```

First time only — build the SQLite dev DB from the legacy CSV and train
the initial model (both are idempotent; re-running skips already-migrated
rows and just registers another model version):

```bash
python migrate_from_csv.py     # SCALE불량.csv -> scale_system.db (5 normalized tables)
python train_model.py          # GridSearchCV over 6 algorithms (LogReg/DecisionTree/RandomForest/XGBoost/LightGBM/SVM);
                                # registers ALL 6 as separate model_version rows (visible + selectable on the dashboard's
                                # "모델 관리" table). Only auto-activates the best-AUC one if no active model exists yet
                                # for this name (bootstrap) — otherwise all 6 register inactive, pick one from the dashboard.
```

## Build

No build step (interpreted Python + no-bundler static JS).

## Run (agent path)

### 1. Backend logic layer — fast, no server needed

```bash
cd backend
python smoke_test_phase23.py
```

This creates 4 throwaway test plates through the real ingestion+rule+
prediction code paths (`TestClient(app)`, in-process — no `uvicorn`
required) and asserts each rule fires correctly (HSB not applied, rolling
temp ≥1000℃, insufficient temp drop, clean case) plus prediction/SHAP and
dashboard-summary endpoints. Prints each response; exits non-zero on any
assertion failure.

### 1b. Batch CSV scoring — also no server needed

```bash
cd backend
python batch_predict.py --csv "../data/raw/SCALE불량.csv" --out batch_predict_result.csv
```

Scores every row of the CSV against the active model (no DB writes),
prints accuracy/precision/recall + confusion matrix when the file has a
`scale` column, and writes the full per-row result (sorted highest-risk
first) to `--out`. Same core function (`batch_scoring.score_dataframe`)
backs the dashboard's "일괄 처리" upload form and the
`POST /api/v1/batch-predict` endpoint — verify any of the three, you've
verified all three.

### 2. Dashboard UI layer — needs a live server

Start the server in the background and wait for readiness:

```bash
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000 > uvicorn.log 2>&1 &
timeout 30 bash -c 'until curl -sf http://127.0.0.1:8000/api/v1/models >/dev/null; do sleep 1; done'
```

Install the driver's dependency once (downloads headless Chromium,
~115MB, first time only) and run it:

```bash
cd backend/.claude/skills/run-scale-backend
npm install
node dashboard_driver.mjs                      # default: http://127.0.0.1:8000
node dashboard_driver.mjs http://localhost:9000 # or pass a different base URL
```

The driver seeds one fresh plate (`RUN_SKILL_<timestamp>`, always unique
so re-runs never 409) through `/plates` → `/furnace-records` →
`/rolling-records` with `hsb_applied:false` so a real CRITICAL alert +
high-risk prediction + SHAP breakdown are guaranteed, then opens `/app/`
in headless Chromium, drives the coil-detail search, uploads the
project's own `SCALE불량.csv` through the batch-scoring form, and
screenshots each step:

Screenshots → `.claude/skills/run-scale-backend/screenshots/`:

| file | shows |
|---|---|
| `01_models.png` | model registry table: every candidate ever trained (not just the winner), sorted by valid AUC, with precision/recall/F1/params, activate button |
| `02_plate_detail.png` | coil search result: prediction prob, risk pill, alert badges, SHAP bars |
| `03_batch_result.png` | batch upload result: summary tiles (accuracy/precision/recall/confusion matrix) + per-row table sorted by risk |
| `04_adhoc_result.png` | what-if form result: manually-entered process values scored live (no plate/DB row needed), same SHAP-bar panel as coil detail |

The script also prints `[console errors]` (JS console errors captured
during the run — should be `[]`) and the seeded plate_no.

Stop the server when done (no `lsof` on this Windows box — find the PID
via `netstat` and kill it with `taskkill`):

```bash
netstat -ano | grep ":8000" | grep LISTENING   # -> last column is the PID
taskkill //PID <pid> //F
```

## Run (human path)

```bash
cd backend
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/app/` in a browser. Ctrl-C to stop.

## Test

There is no separate pytest suite — `smoke_test_phase23.py` (above) *is*
the test: it asserts on every response and exits non-zero on failure.

---

## Gotchas

- **Batch-scoring the original `SCALE불량.csv` shows near-perfect
  accuracy (~99.7%) — that's optimistic, not a bug.** That file is the
  same data the active model was trained on, so it's not a held-out
  test. `batch_scoring.score_dataframe()` (used by `batch_predict.py`,
  `POST /api/v1/batch-predict`, and the dashboard's upload form) is
  meant for scoring *new* CSVs in the same schema — point it at
  unseen data for a meaningful accuracy read.
- **The batch endpoint needs `python-multipart` installed** (FastAPI
  raises at import/request time otherwise, not at startup) — it's in
  `requirements.txt` alongside `fastapi`.

- **`<span>` with inline `width`/`height` in CSS does nothing.** The SHAP
  bar fill (`static/styles.css` `.shap-fill`) silently failed to render
  any color until `display: block` was added — `span` is inline by
  default and ignores box-model width/height. If you add new bar/meter
  UI in `static/`, remember this.
- **Table cells wrap awkwardly without `white-space: nowrap`.** The
  alerts table's rule-name/severity-pill/button cells wrapped into 2-3
  lines at normal card width. Fixed globally on `th, td` in
  `styles.css`; `.tbl-wrap { overflow-x: auto }` handles the overflow.
- **pandas 3.x can silently produce `object`-dtype DataFrames after
  `pd.get_dummies` + `reindex`.** This crashed `shap.Explainer` with
  `TypeError: Cannot cast array data from dtype('O') to dtype('float64')`
  deep inside `_cext_dense_tree_update_weights`. Fix: `.astype("float64")`
  right after building the design matrix, both in `train_model.py`
  and in `prediction_service.py`'s live inference path — the two must
  match or SHAP's background-vs-foreground dtypes diverge again.
- **`CAST(col AS DATE)` is a silent no-op on SQLite** (no real DATE
  affinity) — `/api/v1/dashboard/summary`'s "today" filters always
  matched 0 rows until replaced with `func.date(col) == today.isoformat()`,
  which works identically on SQLite and Postgres.
- **Windows console is cp949, not UTF-8.** Any `print()`/`console.log`
  containing an em dash (—, U+2014) crashes with
  `UnicodeEncodeError: 'cp949' codec can't encode character '\u2014'`
  the moment it reaches a real terminal (TestClient/pytest capture
  hides this). Every backend script that prints starts with
  `sys.stdout.reconfigure(encoding="utf-8")`. Node's `console.log` was
  fine with Korean text in this same terminal without any fix needed —
  only Python hit this.
- **The migrated 1,000 CSV rows never went through `/rolling-records`**,
  so they have no `prediction` row — `GET /predictions/PLT_1001` 404s by
  design (the dashboard shows an empty-state message, not a bug). Use a
  freshly-ingested plate (like the driver's seeded one) to see the
  populated SHAP view.
- **`smoke_test_phase23.py` is not re-runnable without cleanup.** It
  uses fixed plate_nos (`TEST_HSB_OFF`, `TEST_HOT`, `TEST_COOLDROP`,
  `TEST_CLEAN`) and `POST /plates` 409s on a duplicate `plate_no` — a
  second run against the same `scale_system.db` fails on the first
  `assert r.status_code == 201`. Delete those 4 plates (and their
  cascaded furnace/rolling/prediction/alert rows) between runs, or run
  against a throwaway `DATABASE_URL=sqlite:///:memory:`-style fresh DB.
- **The "모델 관리" table has 9 columns and overflows its container if any cell is unbounded.** The `파라미터` column (grid-search `best_params`, e.g. `learning_rate=0.05, max_depth=5, n_estimators=100`) pushed the 활성화 button off-screen until `td.truncate { max-width:200px; overflow:hidden; text-overflow:ellipsis }` + a `title` attribute (full value on hover) was added. If you add more columns, re-check `.tbl-wrap` doesn't need horizontal scroll to reach the action button.
- **`model_version.metrics` has two incompatible shapes** depending on when it was registered: legacy rows (`v1.1.0`–`v1.3.0`, before the multi-candidate registry) nest a candidate's metrics under `metrics.valid_comparison[algorithm]`; current rows (`{algorithm}-{timestamp}` version strings, one row per trained candidate) store them flat under `metrics.valid`. `static/format.mjs`'s `extractModelMetrics()` normalizes both — always read metrics through it, never `m.metrics.valid` directly, or old rows render blank.
- **`/models/{id}/activate` must deactivate the old active version
  before activating the new one**, in that order with a `flush()`
  between — the partial unique index (`name` WHERE `is_active`) rejects
  two active rows for the same model `name` even transiently.

## Troubleshooting

- **`Python was not found; run without arguments to install from the
  Microsoft Store...`**: you invoked the WindowsApps alias stub, not a
  real interpreter. Run `where python` and use the non-`WindowsApps`
  path explicitly (or fix PATH ordering).
- **`ModuleNotFoundError: No module named 'fastapi'` (or `shap`,
  `uvicorn`)**: `pip install -r ../requirements.txt` wasn't run against the
  same `python` you're invoking — check `where python` matches.
- **`sqlite3.IntegrityError: NOT NULL constraint failed: users.id`**
  when calling `Base.metadata.create_all()` against SQLite: a
  `BigInteger` primary key isn't recognized as SQLite's rowid-alias
  autoincrement column. `models.py` defines `BigIntPK =
  BigInteger().with_variant(Integer, "sqlite")` for every PK — if you
  add a new table, use `BigIntPK`, not bare `BigInteger`, for its `id`.
- **`dashboard_driver.mjs` hangs on `page.waitForSelector(...
  detail-summary)`**: the plate you searched has no prediction (see
  Gotchas above). The driver seeds its own plate for exactly this
  reason — don't hardcode `PLT_1001` or similar migrated plate_nos.
- **`playwright install chromium` looks stuck at a progress bar**: it's
  downloading a ~115MB browser build, not frozen — first run only,
  subsequent `npm install` reuses the cached build.
