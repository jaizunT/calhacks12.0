import React, { useState, useRef } from "react";

// Utility components -------------------------------------------------

function Stat({ label, value, hint }) {
  return (
    <div className="flex flex-col px-3 py-2 bg-white/60 rounded-xl border border-slate-200 shadow-sm">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="text-lg font-semibold text-slate-900 leading-tight">
        {value ?? "—"}
      </div>
      {hint && (
        <div className="text-[11px] text-slate-400 leading-snug mt-1">{hint}</div>
      )}
    </div>
  );
}

function Meter({ score = 0, max = 4 }) {
  const pct = Math.min(100, (score / max) * 100);
  return (
    <div className="w-full">
      <div className="flex justify-between text-[11px] text-slate-500 mb-1">
        <span>Hook Strength</span>
        <span>
          {score}/{max}
        </span>
      </div>
      <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-indigo-500 via-blue-500 to-cyan-400"
          style={{ width: pct + "%" }}
        />
      </div>
      <div className="text-[10px] text-slate-400 mt-1">
        High early motion, loud intro audio, instant CTA, or hard first cut →
        stronger hook.
      </div>
    </div>
  );
}

// Heuristic natural language summary --------------------------------
// This pulls from your feature dict structure returned in run.py
function InsightSummary({ ad }) {
  if (!ad) return null;

  // category guess
  const category = ad.vertical_top1 || "Uncertain category";

  // urgency from CTA timing and CTA keywords
  const earlyCTA =
    ad.cta_time_s !== null && ad.cta_time_s !== undefined && ad.cta_time_s <= 1.5;
  const ctaKeywords =
    Array.isArray(ad.asr_cta_hits) && ad.asr_cta_hits.length > 0
      ? ad.asr_cta_hits.join(", ")
      : null;

  // motion/audio punch
  const punchyMotion = ad.motion_intensity_0_1s > 0.5;
  const punchyAudio =
    ad.loudness_dbfs_0_1s !== null &&
    ad.loudness_dbfs_0_1s !== undefined &&
    ad.loudness_dbfs_0_1s > -25.0;

  // clarity of brand
  const strongBrand = ad.logo_stability >= 0.5;

  // vibe
  const smiles =
    ad.face_count > 0
      ? `${Math.round(ad.smile_rate * 100)}% of detected faces smiling`
      : "no faces detected";

  // clutter
  const clutterLevel =
    ad.clutter < 0.2
      ? "minimal clutter"
      : ad.clutter < 0.4
      ? "moderate clutter"
      : "visually busy / high clutter";

  return (
    <div className="text-sm text-slate-700 leading-relaxed">
      <p className="mb-2">
        This creative looks like{" "}
        <span className="font-semibold text-slate-900">{category}</span>. The
        opening second is{" "}
        {punchyMotion || punchyAudio ? (
          <span className="font-semibold text-indigo-600">aggressive</span>
        ) : (
          <span className="font-semibold text-slate-600">more calm</span>
        )}
        , with{" "}
        {punchyMotion
          ? "strong motion"
          : "limited motion"}{" "}
        and{" "}
        {punchyAudio
          ? "loud / attention-grabbing audio"
          : "lower-intensity audio"}
        .
      </p>
      {earlyCTA ? (
        <p className="mb-2">
          A call to action appears within ~
          {ad.cta_time_s?.toFixed(2)}s, which is extremely fast and helps stop
          scroll.{" "}
          {ctaKeywords && (
            <>
              The ad literally says{" "}
              <span className="font-semibold text-slate-900">{ctaKeywords}</span>
              , signaling immediate conversion intent.
            </>
          )}
        </p>
      ) : (
        <p className="mb-2">
          The CTA isn't surfaced in the first second; engagement may rely more
          on storytelling before conversion ask.
        </p>
      )}
      <p className="mb-2">
        Brand presence is{" "}
        {strongBrand ? (
          <span className="font-semibold text-green-600">stable</span>
        ) : (
          <span className="font-semibold text-amber-600">not strongly anchored</span>
        )}
        , and we see {smiles}, in a frame that has {clutterLevel}.
      </p>
      <p className="mb-0">
        Overall, this ad’s hook score of{" "}
        <span className="font-semibold text-slate-900">{ad.hook_score}</span>/4
        suggests it{" "}
        {ad.hook_score >= 3
          ? "grabs attention immediately (good for short-form feeds)."
          : ad.hook_score === 2
          ? "is decent at capturing interest but could punch harder upfront."
          : "leans on slower build-up and may underperform in skip-heavy environments."}
      </p>
    </div>
  );
}

// Per-ad result card -------------------------------------------------
function AdCard({ ad, previewUrl }) {
  return (
    <div className="bg-white rounded-2xl shadow-xl border border-slate-200 p-6 flex flex-col lg:flex-row gap-6">
      {/* LEFT: creative preview + hook meter */}
      <div className="w-full lg:w-1/3 flex flex-col gap-4">
        <div className="relative rounded-xl overflow-hidden border border-slate-200 bg-slate-100 aspect-video flex items-center justify-center">
          {previewUrl ? (
            // If it's a video we could thumbnail the first frame server-side;
            // here we just show the chosen previewUrl
            <img
              src={previewUrl}
              className="object-contain w-full h-full"
              alt={ad.ad_id}
            />
          ) : (
            <div className="text-slate-400 text-xs text-center px-4">
              Preview not available
            </div>
          )}
          <div className="absolute top-2 left-2 text-[10px] bg-white/80 text-slate-700 px-2 py-1 rounded-lg border border-white shadow">
            {ad.modality === "video" ? "VIDEO" : "IMAGE"}
          </div>
        </div>

        <div>
          <Meter score={ad.hook_score} max={4} />
        </div>

        <div className="grid grid-cols-2 gap-2">
          <Stat
            label="First Cut"
            value={
              ad.first_cut_time_s != null
                ? ad.first_cut_time_s.toFixed(2) + "s"
                : "none"
            }
            hint="Scene change timing"
          />
          <Stat
            label="CTA Shown"
            value={ad.cta_present ? "Yes" : "No"}
            hint={
              ad.cta_time_s != null
                ? `@ ${ad.cta_time_s.toFixed(2)}s`
                : "No early CTA"
            }
          />
          <Stat
            label="Brand Stability"
            value={(ad.logo_stability * 100).toFixed(0) + "%"}
            hint="Logo/text persistence"
          />
          <Stat
            label="Faces"
            value={ad.face_count}
            hint={
              ad.face_count > 0
                ? `${Math.round(ad.smile_rate * 100)}% smiling`
                : "—"
            }
          />
        </div>
      </div>

      {/* RIGHT: feature analysis tabs/sections */}
      <div className="w-full lg:w-2/3 flex flex-col gap-6">
        {/* Section: high-level story */}
        <section>
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">
            Ad Intelligence Summary
          </div>
          <InsightSummary ad={ad} />
        </section>

        {/* Section: Categorization */}
        <section className="bg-slate-50 rounded-xl border border-slate-200 p-4">
          <div className="flex items-start justify-between flex-wrap gap-4">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                Predicted Vertical
              </div>
              <div className="text-base font-semibold text-slate-900">
                {ad.vertical_top1 || "—"}
              </div>
              <div className="text-[11px] text-slate-500 mt-1 leading-snug">
                Top 3:
                <span className="font-medium text-slate-700 ml-1">
                  {(ad.vertical_top3 || []).join(", ")}
                </span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                Ambiguity
              </div>
              <div className="text-base font-semibold text-slate-900">
                {ad.vertical_entropy
                  ? ad.vertical_entropy.toFixed(2)
                  : "—"}
              </div>
              <div className="text-[11px] text-slate-500 mt-1 leading-snug">
                Higher entropy = less confident category match.
              </div>
            </div>
          </div>
        </section>

        {/* Section: First-second hook analytics */}
        <section className="bg-slate-50 rounded-xl border border-slate-200 p-4 grid grid-cols-2 gap-4">
          <Stat
            label="Motion @0-1s"
            value={ad.motion_intensity_0_1s?.toFixed(2)}
            hint="Optical flow magnitude (higher = fast action)"
          />
          <Stat
            label="Audio Level @0-1s"
            value={
              ad.loudness_dbfs_0_1s != null
                ? ad.loudness_dbfs_0_1s.toFixed(1) + " dBFS"
                : "—"
            }
            hint="Louder intro = stronger scroll-stop"
          />
          <Stat
            label="Cut Rate"
            value={ad.cut_rate_per_5s?.toFixed(1) + "/5s"}
            hint="Jump cuts per 5 seconds"
          />
          <Stat
            label="Camera Shake Var"
            value={ad.shake_var?.toFixed(2)}
            hint="Higher = handheld/chaotic feel"
          />
        </section>

        {/* Section: Messaging alignment */}
        <section className="bg-slate-50 rounded-xl border border-slate-200 p-4">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500 mb-2">
            Message / Visual Alignment
          </div>
          <div className="grid grid-cols-2 gap-4">
            <Stat
              label="OCR ↔ Visual"
              value={ad.align_ocr_img?.toFixed(2)}
              hint="Does on-screen text match what we show?"
            />
            <Stat
              label="Audio ↔ Visual"
              value={ad.align_asr_img?.toFixed(2)}
              hint="Does narration match what's on screen?"
            />
          </div>
          <div className="text-[11px] text-slate-500 mt-3 leading-snug">
            Higher alignment means clearer storytelling and less mismatch between
            promise (voice/text) and visuals, which tends to improve trust and
            conversion in recommendation systems.
          </div>
        </section>
      </div>
    </div>
  );
}

// Main App -----------------------------------------------------------
export default function App() {
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const dropRef = useRef(null);

  function handleFileSelect(e) {
    const incoming = Array.from(e.target.files || []);
    setFiles((prev) => [...prev, ...incoming]);
  }

  function handleDrop(e) {
    e.preventDefault();
    const incoming = Array.from(e.dataTransfer.files || []);
    setFiles((prev) => [...prev, ...incoming]);
  }

  function handleDragOver(e) {
    e.preventDefault();
  }

  async function handleAnalyze() {
    setLoading(true);

    // Build form data to send to backend
    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));

    // You'd create an /analyze route in Flask/FastAPI that:
    // 1. saves uploads
    // 2. calls process_single_ad on each file (from run.py)
    // 3. returns JSON array of feature dicts, plus a data URL thumbnail
    //
    // For now we'll just fake it client-side to show UI structure.
    //
    // const resp = await fetch("/analyze", { method: "POST", body: formData });
    // const data = await resp.json();
    // setResults(data);

    // MOCK: create demo objects using shape from run.py output
    const mock = files.map((f, idx) => ({
      ad_id: f.name,
      modality: f.type.includes("video") ? "video" : "image",

      // ---- pulled from your pipeline outputs ----
      hook_score: 3,
      first_cut_time_s: 0.18,
      cta_present: true,
      cta_time_s: 0.7,
      logo_stability: 0.6,
      face_count: 2,
      smile_rate: 0.5,
      motion_intensity_0_1s: 0.87,
      loudness_dbfs_0_1s: -18.4,
      cut_rate_per_5s: 4.2,
      shake_var: 0.13,
      vertical_top1: "mobile game / gaming app",
      vertical_top3: [
        "mobile game / gaming app",
        "finance / investing / trading",
        "brand awareness / lifestyle / no specific CTA",
      ],
      vertical_entropy: 0.62,
      align_ocr_img: 0.71,
      align_asr_img: 0.66,
      clutter: 0.21,
      warmth: 1.34,
      asr_cta_hits: ["download", "play now"],
      // -------------------------------------------
      smile_rate_pct: 50,
      previewUrl: URL.createObjectURL(f),
    }));

    setResults(mock);
    setLoading(false);
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-indigo-50 text-slate-900 flex flex-col">
      {/* Header */}
      <header className="px-6 py-5 border-b border-slate-200 bg-white/80 backdrop-blur-md sticky top-0 z-20">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <div className="text-xs font-semibold tracking-wide text-indigo-600 uppercase">
              Ad Intelligence Challenge
            </div>
            <h1 className="text-xl font-semibold text-slate-900 leading-tight">
              Creative Signal Explorer
            </h1>
            <p className="text-sm text-slate-500 leading-snug">
              Upload ad creatives (image/video). We’ll extract high-value,
              minimally overlapping features across vision, motion, audio,
              and narrative alignment — optimized for ranking models.
            </p>
          </div>
          <button
            className="px-4 py-2 rounded-xl bg-indigo-600 text-white text-sm font-semibold shadow-lg shadow-indigo-600/20 hover:bg-indigo-500 transition-colors"
            onClick={handleAnalyze}
            disabled={loading || files.length === 0}
          >
            {loading ? "Analyzing…" : "Analyze Creatives"}
          </button>
        </div>
      </header>

      <main className="flex-1 px-6 py-8">
        <div className="max-w-7xl mx-auto flex flex-col gap-8">
          {/* Upload zone */}
          <section
            ref={dropRef}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            className="border-2 border-dashed border-slate-300 rounded-2xl bg-white/70 p-6 shadow-inner flex flex-col sm:flex-row sm:items-center sm:justify-between gap-6"
          >
            <div className="flex-1">
              <div className="text-sm font-semibold text-slate-800">
                Drop your .png / .mp4 files
              </div>
              <div className="text-xs text-slate-500 leading-snug mt-1">
                We’ll sample frames (~3 fps, first 8s max), run CLIP,
                OCR, ASR (Whisper tiny), motion analysis, and produce
                structured features like CTA timing, hook strength, and
                brand stability. :contentReference[oaicite:12]{index=12} :contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14}
              </div>
            </div>
            <div className="flex flex-col items-start sm:items-end gap-3">
              <label className="cursor-pointer">
                <div className="px-4 py-2 rounded-xl border border-slate-300 bg-white text-slate-700 text-sm font-medium shadow hover:bg-slate-50 active:bg-slate-100 transition-colors">
                  Browse Files
                </div>
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,.mp4,.mov,.avi,.mkv"
                  multiple
                  className="hidden"
                  onChange={handleFileSelect}
                />
              </label>
              {files.length > 0 && (
                <div className="text-[11px] text-slate-500 leading-tight">
                  {files.length} file{files.length === 1 ? "" : "s"} selected
                </div>
              )}
            </div>
          </section>

          {/* Results list */}
          {results.length > 0 && (
            <section className="flex flex-col gap-6">
              <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                Analysis Results
              </div>

              <div className="flex flex-col gap-8">
                {results.map((ad, i) => (
                  <AdCard key={i} ad={ad} previewUrl={ad.previewUrl} />
                ))}
              </div>
            </section>
          )}
        </div>
      </main>

      <footer className="px-6 py-8 border-t border-slate-200 text-center text-[11px] text-slate-400">
        Features surfaced here are designed to be:
        distinct, scalable (8s media window, ~3fps sampling),
        and aligned with predicted engagement levers
        (attention hook, clarity of offer, trust). :contentReference[oaicite:15]{index=15}
      </footer>
    </div>
  );
}
