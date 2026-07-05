import { useState } from "react";
import { predict } from "../api/predictApi";

const THRESHOLD = 0.5;

function ProbabilityMeter({ probability, flagged }) {
  const pct = Math.round(probability * 100);
  const fillColor = flagged ? "bg-rose-500" : "bg-teal-500";

  return (
    <div className="relative h-2.5 w-full rounded-full bg-slate-200">
      <div
        className={`h-2.5 rounded-full ${fillColor} transition-all duration-500`}
        style={{ width: `${pct}%` }}
      />
      {/* garis ambang batas 0.5 — inti dari cara model memutuskan label */}
      <div
        className="absolute top-0 h-2.5 w-px bg-slate-400"
        style={{ left: `${THRESHOLD * 100}%` }}
      />
    </div>
  );
}

function LabelResult({ name, result }) {
  const flagged = result.label === 1;
  const pct = Math.round(result.probability * 100);

  return (
    <div className="space-y-2.5 rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex items-center justify-between">
        <span className="text-base font-medium text-slate-800">{name}</span>
        <span
          className={`rounded-full px-2.5 py-1 text-sm font-medium ${
            flagged
              ? "bg-rose-100 text-rose-600"
              : "bg-teal-100 text-teal-700"
          }`}
        >
          {flagged ? "Terdeteksi" : "Tidak terdeteksi"}
        </span>
      </div>

      <ProbabilityMeter probability={result.probability} flagged={flagged} />

      <div className="flex items-center justify-between text-sm">
        <span className="font-mono text-slate-400">threshold 0.50</span>
        <span className="font-mono text-slate-600">{pct}%</span>
      </div>
    </div>
  );
}

function HistoryItem({ item, onDelete }) {
  const hsFlag = item.result.hate_speech.label === 1;
  const abFlag = item.result.abusive.label === 1;

  return (
    <div className="flex items-start justify-between gap-3 rounded-lg border border-slate-200 bg-white p-3">
      <div className="min-w-0 flex-1 space-y-1.5">
        <p className="truncate text-sm text-slate-600">{item.text}</p>
        <div className="flex gap-2">
          <span
            className={`rounded-full px-2 py-0.5 text-xs font-medium ${
              hsFlag ? "bg-rose-100 text-rose-600" : "bg-teal-100 text-teal-700"
            }`}
          >
            HS {hsFlag ? "terdeteksi" : "aman"}
          </span>
          <span
            className={`rounded-full px-2 py-0.5 text-xs font-medium ${
              abFlag ? "bg-rose-100 text-rose-600" : "bg-teal-100 text-teal-700"
            }`}
          >
            Abusive {abFlag ? "terdeteksi" : "aman"}
          </span>
        </div>
      </div>
      <button
        onClick={() => onDelete(item.id)}
        aria-label="Hapus riwayat ini"
        className="shrink-0 rounded-md px-2 py-1 text-sm text-slate-400 hover:bg-slate-100 hover:text-rose-500"
      >
        ✕
      </button>
    </div>
  );
}

export default function Classifier() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const data = await predict(text);
      setResult(data);
      setHistory((prev) => [
        { id: Date.now(), text, result: data },
        ...prev,
      ]);
    } catch (err) {
      setError("Gagal menghubungi server. Coba lagi.");
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setText("");
    setResult(null);
    setError(null);
  }

  function handleDeleteHistory(id) {
    setHistory((prev) => prev.filter((item) => item.id !== id));
  }

  function handleClearHistory() {
    setHistory([]);
  }

  const hasResult = result && !loading;

  return (
    <div className="min-h-screen bg-slate-50 px-4 py-12">
      <div className="mx-auto w-full max-w-lg space-y-8">
        <div className="space-y-6">
          <div className="space-y-1.5">
            <p className="font-mono text-xs uppercase tracking-widest text-slate-400">
              Klasifikasi Multi-Label
            </p>
            <h1 className="text-2xl font-semibold text-slate-900">
              Deteksi Hate Speech &amp; Abusive Language
            </h1>
            <p className="text-base text-slate-500">
              Analisis teks tweet berbahasa Indonesia menggunakan model
              LSTM dengan GloVe word embedding.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-3">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Tempel atau tulis teks tweet di sini..."
              rows={5}
              disabled={hasResult}
              className="w-full resize-none rounded-lg border border-slate-300 bg-white p-4 text-base text-slate-900 placeholder-slate-400 outline-none focus:border-slate-500 focus:ring-2 focus:ring-slate-200 disabled:bg-slate-100 disabled:text-slate-500"
            />

            {!hasResult ? (
              <button
                type="submit"
                disabled={loading || !text.trim()}
                className="w-full rounded-lg bg-slate-900 py-3 text-base font-medium text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
              >
                {loading ? "Menganalisis..." : "Analisis Teks"}
              </button>
            ) : (
              <button
                type="button"
                onClick={handleReset}
                className="w-full rounded-lg border border-slate-300 bg-white py-3 text-base font-medium text-slate-700 transition hover:bg-slate-100"
              >
                Analisis Ulang
              </button>
            )}
          </form>

          {error && <p className="text-sm text-rose-500">{error}</p>}

          {hasResult && (
            <div className="space-y-3">
              <LabelResult name="Hate Speech" result={result.hate_speech} />
              <LabelResult name="Abusive Language" result={result.abusive} />
            </div>
          )}
        </div>

        {history.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-medium text-slate-700">
                Riwayat Analisis
              </h2>
              <button
                onClick={handleClearHistory}
                className="text-sm text-slate-400 hover:text-rose-500"
              >
                Hapus semua
              </button>
            </div>
            <div className="space-y-2">
              {history.map((item) => (
                <HistoryItem
                  key={item.id}
                  item={item}
                  onDelete={handleDeleteHistory}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}