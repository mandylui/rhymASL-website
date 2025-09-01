import axios from "axios";
const API = axios.create({ baseURL: import.meta.env.VITE_API_BASE });
export const gloss = (text:string)=>API.post("/gloss",{text}).then(r=>r.data);
export const analyze = (text:string)=>API.post("/analyze",{text}).then(r=>r.data);
