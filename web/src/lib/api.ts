import axios from "axios";
const API = axios.create({ baseURL: import.meta.env.VITE_API_BASE || "http://localhost:8000" });

export type GlossResp = { input: string; gloss: string; videos?: string[]; missing?: string[]; };
  export type AnalyzeResp = { input: string; gloss_tokens: string[]; entry_ids: string[]; lemmas: string[] };

export const callGloss   = (text: string) => API.post("/gloss",   { text }).then(r => r.data as GlossResp);
export const callAnalyze = (text: string) => API.post("/analyze", { text }).then(r => r.data as AnalyzeResp);
