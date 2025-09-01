import { useState } from "react";
import { gloss, analyze } from "./lib/api";

export default function App() {
  const [text,setText]=useState(""); const [g,setG]=useState(""); const [res,setRes]=useState<any|null>(null);
  return (
    <main style={{maxWidth:720,margin:"40px auto"}}>
      <h1>rhymASL</h1>
      <input value={text} onChange={e=>setText(e.target.value)} placeholder="Type sentence…" style={{width:"100%",padding:8}}/>
      <div style={{display:"flex",gap:8,marginTop:8}}>
        <button onClick={async()=>setG((await gloss(text)).gloss)}>Gloss</button>
        <button onClick={async()=>setRes(await analyze(text))}>Analyze</button>
      </div>
      {g && <p><b>Gloss:</b> {g}</p>}
      {res && <pre>{JSON.stringify(res,null,2)}</pre>}
    </main>
  );
}
