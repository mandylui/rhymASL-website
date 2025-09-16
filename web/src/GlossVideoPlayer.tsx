import React, { useEffect, useMemo, useRef, useState } from "react";

type Props = {
  files: string[];
  delayMs?: number; // pause between clips
};

const GlossVideoPlayer: React.FC<Props> = ({ files, delayMs = 400 }) => {
  const [i, setI] = useState(0);
  const [hint, setHint] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Reset to first file when list changes
  useEffect(() => {
    setI(0);
    setHint(null);
  }, [JSON.stringify(files)]);

  if (!files || files.length === 0) return <p className="small">No playable videos.</p>;

  const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
  const src = useMemo(() => `${API_BASE}/videos/${files[i]}`, [API_BASE, files, i]);

  // Try to autoplay once the source/metadata is ready
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;

    const tryPlay = async () => {
      try {
        v.muted = true;           // autoplay requires muted
        v.playsInline = true;     // avoid fullscreen on iOS
        v.currentTime = 0;
        await v.play();
        setHint(null);
      } catch (e) {
        // Autoplay blocked: show a hint; user can press Play
        setHint("Autoplay blocked by browser. Click Play.");
      }
    };

    const onLoaded = () => void tryPlay();
    v.addEventListener("loadedmetadata", onLoaded);
    v.addEventListener("loadeddata", onLoaded);
    // also attempt immediately (some browsers allow without waiting)
    void tryPlay();

    return () => {
      v.removeEventListener("loadedmetadata", onLoaded);
      v.removeEventListener("loadeddata", onLoaded);
    };
  }, [src]);

  const next = () => {
    if (i < files.length - 1) {
      setTimeout(() => setI((x) => x + 1), delayMs);
    }
  };

  const prev = () => setI((x) => Math.max(0, x - 1));
  const manualNext = () => setI((x) => Math.min(files.length - 1, x + 1));

  return (
    <div className="flex flex-col items-center space-y-3">
      <h2 className="text-lg font-semibold">
        {i + 1}/{files.length}: {files[i]}
      </h2>

      <video
        key={files[i]}
        ref={videoRef}
        autoPlay
        muted
        playsInline
        preload="auto"
        controls
        onEnded={next}
        onError={() => setHint(`Can't load ${files[i]}.`)}
        className="w-96 max-w-full border rounded-lg shadow"
        >
        <source src={src} type="video/mp4" />
      </video>


      {hint && <p className="small" style={{ color: "#fca5a5" }}>{hint} <a href={src} target="_blank">Open video</a></p>}

      <div className="flex gap-2">
        <button className="btn secondary" onClick={() => videoRef.current?.play()}>Play</button>
        <button className="btn secondary" onClick={() => videoRef.current?.pause()}>Pause</button>
        <button className="btn" onClick={prev} disabled={i === 0}>Prev</button>
        <button className="btn" onClick={manualNext} disabled={i === files.length - 1}>Next</button>
      </div>
    </div>
  );
};

export default GlossVideoPlayer;
