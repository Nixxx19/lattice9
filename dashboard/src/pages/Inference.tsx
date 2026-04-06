import { useState } from "react";

interface WorkerTrace {
  worker_id: string;
  url: string;
  phase: string;
  layers: number[];
  latency_ms: number;
}

interface InferResult {
  request_id: string;
  prompt: string;
  result: string;
  tokens_generated: number;
  total_latency_ms: number;
  worker_trace: WorkerTrace[];
}

export default function Inference() {
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState(50);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<InferResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activePhase, setActivePhase] = useState<string | null>(null);

  const runInference = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    setActivePhase("distributing");

    try {
      const res = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, max_tokens: maxTokens }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Inference failed");
      }

      const data: InferResult = await res.json();
      setResult(data);
      setActivePhase(null);
    } catch (e: any) {
      setError(e.message);
      setActivePhase(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Input section */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h2 className="text-lg font-semibold text-white mb-4">
          Submit Inference
        </h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Once upon a time in a land far away..."
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-600 focus:outline-none focus:border-indigo-500 resize-none"
              rows={3}
            />
          </div>
          <div className="flex items-end gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Max Tokens
              </label>
              <input
                type="number"
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
                min={1}
                max={200}
                className="w-32 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-indigo-500"
              />
            </div>
            <button
              onClick={runInference}
              disabled={loading || !prompt.trim()}
              className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg font-medium transition-colors"
            >
              {loading ? "Processing..." : "Run Inference"}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-700 text-red-300 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {/* Progress visualization */}
      {loading && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
            Processing
          </h3>
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 rounded-full bg-indigo-500 animate-ping" />
            <span className="text-indigo-400">
              Distributing inference across worker nodes...
            </span>
          </div>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="space-y-6">
          {/* Generated text */}
          <div className="bg-gray-900 border border-green-800 rounded-lg p-6">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
                Generated Text
              </h3>
              <span className="text-xs text-gray-500 font-mono">
                {result.request_id}
              </span>
            </div>
            <p className="text-white leading-relaxed">{result.result}</p>
            <div className="mt-4 flex gap-4 text-xs text-gray-500">
              <span>
                Tokens: <span className="text-green-400">{result.tokens_generated}</span>
              </span>
              <span>
                Latency:{" "}
                <span className="text-yellow-400">
                  {result.total_latency_ms.toFixed(0)}ms
                </span>
              </span>
            </div>
          </div>

          {/* Worker trace visualization */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
              Worker Pipeline
            </h3>
            <div className="space-y-3">
              {result.worker_trace.map((trace, idx) => {
                const total = result.total_latency_ms;
                const pct = total > 0 ? (trace.latency_ms / total) * 100 : 0;
                const phaseColors: Record<string, string> = {
                  full: "bg-indigo-600",
                  encode: "bg-blue-600",
                  middle: "bg-purple-600",
                  decode: "bg-green-600",
                };
                const barColor = phaseColors[trace.phase] || "bg-gray-600";

                return (
                  <div key={idx} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <span
                          className={`w-2 h-2 rounded-full ${barColor}`}
                        />
                        <span className="text-white font-medium">
                          {trace.worker_id}
                        </span>
                        <span className="text-gray-500 text-xs">
                          {trace.phase}
                        </span>
                        <span className="text-gray-600 text-xs">
                          layers{" "}
                          {trace.layers.length > 0
                            ? `${Math.min(...trace.layers)}-${Math.max(
                                ...trace.layers
                              )}`
                            : "N/A"}
                        </span>
                      </div>
                      <span className="text-gray-400 text-xs">
                        {trace.latency_ms.toFixed(0)}ms
                      </span>
                    </div>
                    <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${barColor} rounded-full transition-all duration-500`}
                        style={{ width: `${Math.max(pct, 2)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Timeline */}
            <div className="mt-6 pt-4 border-t border-gray-800">
              <div className="flex items-center gap-1 h-8 rounded-lg overflow-hidden">
                {result.worker_trace.map((trace, idx) => {
                  const total = result.total_latency_ms;
                  const pct =
                    total > 0 ? (trace.latency_ms / total) * 100 : 0;
                  const colors = [
                    "bg-blue-600",
                    "bg-purple-600",
                    "bg-green-600",
                    "bg-yellow-600",
                    "bg-red-600",
                  ];
                  return (
                    <div
                      key={idx}
                      className={`${colors[idx % colors.length]} h-full flex items-center justify-center text-xs text-white font-medium`}
                      style={{ width: `${Math.max(pct, 5)}%` }}
                      title={`${trace.worker_id}: ${trace.latency_ms.toFixed(0)}ms`}
                    >
                      {pct > 15 ? trace.worker_id : ""}
                    </div>
                  );
                })}
              </div>
              <div className="flex justify-between mt-1 text-xs text-gray-600">
                <span>0ms</span>
                <span>{result.total_latency_ms.toFixed(0)}ms</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
