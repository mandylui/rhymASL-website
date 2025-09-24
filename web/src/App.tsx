import { useState } from "react";
import "./styles.css";
import { callGloss, callImage } from "./lib/api";
import type { GlossResp } from "./lib/api";
import GlossVideoPlayer from "./GlossVideoPlayer";

export default function App() {
  const [text, setText] = useState("");
  const [gloss, setGloss] = useState<GlossResp | null>(null);
  const [loading, setLoading] = useState<"" | "gloss">("");
  const [err, setErr] = useState<string | null>(null);
  const [showPics, setShowPics] = useState(true);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [selectedImg, setSelectedImg] = useState<string | null>(null);

  const doGloss = async () => {
    setErr(null);
    setLoading("gloss");
    setGloss(null);
    try {
      const g = await callGloss(text.trim());
      setGloss(g);
    } catch (e: any) {
      setErr(e?.response?.data?.detail || "Gloss failed");
    } finally {
      setLoading("");
    }
  };

  const doImage = async () => {
    setErr(null);
    setImageUrl(null);
    try {
      const res = await callImage(text.trim());
      setImageUrl(res.url);
    } catch (e: any) {
      setErr(e?.message || "Image failed");
    }
  };

  
  return (
    <div className="app">
      {/* Top bar */}
      <div className="topbar">
        <div className="brand">
          <span>rhymASL</span>
          <span className="badge">demo</span>
        </div>
      </div>

      {/* Sidebar */}
      <aside className="sidebar">
        <div className="header">Create Your Own Story</div>

        <textarea
          className="textarea"
          placeholder='Enter your story here… (e.g., "call silly dog")'
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <div className="row">
          <button className="btn" disabled={!text || !!loading} onClick={doGloss}>
            {loading === "gloss" ? <span className="spinner" /> : "🔤"}
            {loading === "gloss" ? "Translating…" : "Translate"}
          </button>

          <button className="btn secondary" disabled={!text || !!loading} onClick={doImage}>
            🖼️ Generate Image
          </button>
        </div>

        {imageUrl && (
          <div className="card" style={{ marginTop: 12 }}>
            <div className="section-head">
              <div className="section-title">Generated Image</div>
              <div className="status"><span className="dot ok" /> done</div>
            </div>
            <img className="genimg" src={imageUrl} alt="AI generated" />
          </div>
        )}

        <div className="label" style={{ marginTop: 14 }}>
          Demo images
        </div>
        <div className="row">
          <button className="btn ghost" onClick={() => setShowPics((s) => !s)}>
            {showPics ? "Hide" : "Show"}
          </button>
        </div>

        {err && (
          <p className="small warn-text" style={{ marginTop: 10 }}>
            {err}
          </p>
        )}
      </aside>

      {/* Main content */}
      <main className="main">
        <div className="container">
          {/* Gloss card */}
          <div className="card">
            <div className="section-head">
              <div className="section-title-asl-result">ASL Result</div>
              <div className="status">
                <span className={`dot ${gloss ? "ok" : "warn"}`} />
                {gloss ? "Ready" : "No Result"}
              </div>
            </div>

            {gloss ? (
              <div style={{ marginTop: 8 }}>
                <div className="eyebrow">Story</div>
                <div className="story-text">{gloss.input}</div>

                <div className="eyebrow">ASL Gloss</div>
                <div className="story-text gloss-text">{gloss.gloss}</div>

                <div className="gloss-video-slot">
                  <div className="video-heading">Playing Full Story…</div>
                  <GlossVideoPlayer files={gloss.videos ?? []} delayMs={800} />
                </div>

                {(gloss.missing?.length ?? 0) > 0 && (
                  <div className="small warn-text" style={{ marginTop: 8 }}>
                    Missing videos for: {gloss.missing.join(", ")}
                  </div>
                )}
              </div>
            ) : (
              <div className="empty">
                <div className="skeleton shimmer" />
              </div>
            )}
          </div>

          {/* Demo images */}
          {showPics && (
            <div className="card-demo">
              <div className="section-head">
                <div className="section-title">Demo Images</div>
                <div className="status"><span className="dot ok" /> Samples</div>
              </div>
              <div className="preview" style={{ marginTop: 10 }}>
                {["/bear.jpg","/call_silly_dog.jpg","/bird_call_dog.jpg","/dirty_pig.jpg"].map(src=>(
                  <div key={src} className="img-wrap" onClick={()=>setSelectedImg(src)}>
                    <img src={src} alt="" />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {selectedImg && (
          <div className="lightbox" onClick={()=>setSelectedImg(null)}>
            <span className="close">×</span>
            <img src={selectedImg} alt="enlarged" />
          </div>
        )}
      </main>
    </div>
  );
}