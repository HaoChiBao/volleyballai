# Decision log

Short record of locked product/engineering choices. Newest first.

---

## 2026-07-24 — Analysis jobs on Google Cloud Run Jobs

**Decision:** Video analysis generation runs as **Google Cloud Run Jobs** (one execution per analysis). The job container runs CPU work (ffmpeg, orchestration, progress) and **calls Modal** for all AI.

**Why:** Batch-friendly timeouts/retries; keeps Next.js thin; matches existing GCP job patterns; separates CPU pipeline from GPU inference.

**Implications:**
- App stores job metadata + Cloud Run execution name
- Same `worker/` Docker image for local run and Cloud Run Job
- AI never runs inside the Cloud Run Job as local weights — Modal only

See [JOBS.md](JOBS.md).

---

## 2026-07-24 — All AI models on Modal only

**Decision:** No AI model weights or inference on the laptop / local worker. SAM, ball, pose, actions, and any future neural models run on Modal.

**Why:** Local install is too heavy; Modal caches weights and scales GPUs.

**Implications:** Local worker is ffmpeg + Modal client + geometry. Every ML stage lives under `modal/`. Mock JSON allowed for UI without GPU spend.

See [AI_POLICY.md](AI_POLICY.md).

---

## 2026-07-24 — Player tracking = SAM 3.1

**Decision:** Use Meta SAM 3.1 (prefer 3.1 Object Multiplex) for player detect + segment + track.

**Why:** Quality masks, text/click prompts, multi-object video tracking; fits offline product + human repair loop.

**Does not replace:** court calibration, ball, actions/score, jersey IDs.

---

## 2026-07-24 — Local-first, then cloud

**Decision:** Build Local v0 on disk (`.data/`) + local Next.js + Modal for AI before Vercel/Supabase production wiring.

**Why:** Faster iteration; prove upload → calibrate → track → overlay loop.

**Later:** Supabase for auth/DB/storage; Vercel for web; keep same stage names and artifact schemas.

---

## 2026-07-24 — Hosting split

**Decision:** Production web on Vercel; data on Supabase; GPU/ML on Modal.

**Why:** Thin control plane; no long GPU on serverless web; Modal for batch CV.

---

## 2026-07-24 — Quality over realtime

**Decision:** Offline / batch analysis. Prefer heavier models and multi-pass cleanup over live FPS.

---

## 2026-07-24 — Calibration as first-class, human-editable

**Decision:** Court corners / keyframes are editable; camera motion handled via segments; recalibration does not re-run SAM.

---

## 2026-07-24 — Score is human-correctable early

**Decision:** Auto score is assistive; editable rallies from the start. Do not block UX on perfect auto scoring.
