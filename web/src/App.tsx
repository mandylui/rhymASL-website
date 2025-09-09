import { useState } from "react";
import "./styles.css";
import { callGloss, callAnalyze } from "./lib/api";
import type { GlossResp, AnalyzeResp } from "./lib/api";

export default function App() {
  const [text, setText] = useState("");
  const [gloss, setGloss] = useState<GlossResp | null>(null);
  const [analysis, setAnalysis] = useState<AnalyzeResp | null>(null);
  const [loading, setLoading] = useState<"" | "gloss" | "analyze">("");
  const [err, setErr] = useState<string | null>(null);
  const [showPics, setShowPics] = useState(true);

  const doGloss = async () => {
    setErr(null); setLoading("gloss"); setGloss(null);
    try { setGloss(await callGloss(text.trim())); } 
    catch (e:any){ setErr(e?.response?.data?.detail || "Gloss failed"); } 
    finally { setLoading(""); }
  };
  const doAnalyze = async () => {
    setErr(null); setLoading("analyze"); setAnalysis(null);
    try { setAnalysis(await callAnalyze(text.trim())); } 
    catch (e:any){ setErr(e?.response?.data?.detail || "Analyze failed"); } 
    finally { setLoading(""); }
  };

  return (
    <div className="app">
      {/* Sidebar (Streamlit-style controls) */}
      <aside className="sidebar">
        <div className="header">rhymASL</div>

        <div className="label">Sentence</div>
        <textarea className="textarea" placeholder='e.g., "call silly dog"' value={text} onChange={e=>setText(e.target.value)} />

        <div className="row">
          <button className="btn" disabled={!text || !!loading} onClick={doGloss}>
            {loading==="gloss" ? "Translating…" : "Translate"}
          </button>
          <button className="btn secondary" disabled={!text || !!loading} onClick={doAnalyze}>
            {loading==="analyze" ? "Analyzing…" : "Analyze"}
          </button>
        </div>

        <div className="label">Demo images</div>
        <div className="row">
          <button className="btn secondary" onClick={()=>setShowPics(s=>!s)}>{showPics ? "Hide" : "Show"}</button>
        </div>

        {err && <p style={{color:"#fecaca", marginTop:10}}>{err}</p>}
        <p className="small" style={{marginTop:14}}>API: {import.meta.env.VITE_API_BASE || "http://localhost:8000"}</p>
      </aside>

      {/* Main content (results) */}
      <main className="main">
        <div className="card">
          <div className="small">Gloss</div>
          {gloss ? (
            <div style={{marginTop:8}}>
              <div className="small">Input</div>
              <div style={{margin:"4px 0 10px"}}>{gloss.input}</div>
              <div className="small">ASL Gloss</div>
              <div style={{fontWeight:800, fontSize:18}}>{gloss.gloss}</div>
            </div>
          ) : (
            <div className="small">No gloss yet.</div>
          )}
        </div>

        <div className="card">
          <div className="small">Analysis</div>
          {analysis ? (
            <>
              <div style={{marginTop:8}} className="small">Tokens</div>
              <div style={{marginBottom:8}}>{analysis.gloss_tokens.join(" · ")}</div>
              <table className="table">
                <thead><tr><th>#</th><th>Token</th><th>Entry</th><th>Lemma</th></tr></thead>
                <tbody>
                  {analysis.entry_ids.map((id, i) => (
                    <tr key={i}>
                      <td className="small">{i+1}</td>
                      <td>{analysis.gloss_tokens[i] ?? "-"}</td>
                      <td>{id ?? "-"}</td>
                      <td>{analysis.lemmas[i] ?? "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          ) : (
            <div className="small">No analysis yet.</div>
          )}
        </div>

        {showPics && (
          <div className="card">
            <div className="small">Demo Images</div>
            <div className="preview" style={{marginTop:10}}>
              <img src="/bear.jpg" alt="bear"/>
              <img src="/call_silly_dog.jpg" alt="dog"/>
              <img src="/bird_call_dog.jpg" alt="bird dog"/>
              <img src="/dirty_pig.jpg" alt="pig"/>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
