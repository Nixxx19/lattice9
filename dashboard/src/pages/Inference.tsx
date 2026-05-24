import { useInference } from "../state";

const WORKER_COLORS = [
  "text-sky-400",
  "text-violet-400",
  "text-emerald-400",
  "text-amber-400",
  "text-rose-400",
  "text-cyan-400",
];

function workerColor(workerId: string, knownWorkers: string[]) {
  const idx = knownWorkers.indexOf(workerId);
  return WORKER_COLORS[(idx >= 0 ? idx : 0) % WORKER_COLORS.length];
}

export default function Inference() {
  const { session, update } = useInference();
  const {
    prompt,
    maxTokens,
    stream,
    loading,
    error,
    result,
    streamedTokens,
    streamWorkers,
    reshards,
  } = session;

  const runInferenceStreamed = async () => {
    update({
      streamedTokens: [],
      streamWorkers: [],
      reshards: [],
      result: null,
    });

    const base = `${window.location.protocol}//${window.location.hostname}:8000`;
    const res = await fetch(`${base}/api/infer/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, max_tokens: maxTokens }),
    });
    if (!res.ok || !res.body) throw new Error("stream open failed");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const tokens: typeof session.streamedTokens = [];
    const reshardEvents: typeof session.reshards = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      for (const block of events) {
        if (!block.trim()) continue;
        const eventLine = block.split("\n").find((l) => l.startsWith("event:"));
        const dataLine = block.split("\n").find((l) => l.startsWith("data:"));
        if (!eventLine || !dataLine) continue;
        const type = eventLine.replace("event:", "").trim();
        const data = JSON.parse(dataLine.replace("data:", "").trim());

        if (type === "start") {
          update({
            streamWorkers: data.assignments.map((a: any) => a.worker_id),
          });
        } else if (type === "token") {
          tokens.push(data);
          update({ streamedTokens: [...tokens] });
        } else if (type === "reshard") {
          reshardEvents.push(data);
          update({ reshards: [...reshardEvents] });
        } else if (type === "done") {
          update({
            result: {
              request_id: data.request_id,
              prompt,
              result: data.result,
              tokens_generated: data.tokens_generated,
              total_latency_ms: data.total_latency_ms,
              worker_trace: [],
            },
          });
        } else if (type === "error") {
          throw new Error(data.detail);
        }
      }
    }
  };

  const runInferenceBlocking = async () => {
    const res = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, max_tokens: maxTokens }),
    });
    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.detail || "inference failed");
    }
    const data = await res.json();
    update({ result: data });
  };

  const run = async () => {
    if (!prompt.trim() || loading) return;
    update({
      loading: true,
      error: null,
      result: null,
      streamedTokens: [],
      reshards: [],
    });
    try {
      if (stream) await runInferenceStreamed();
      else await runInferenceBlocking();
    } catch (e: any) {
      update({ error: e.message });
    } finally {
      update({ loading: false });
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <SectionHeader title="prompt" />
        <textarea
          value={prompt}
          onChange={(e) => update({ prompt: e.target.value })}
          placeholder="once upon a time in a distant galaxy"
          rows={3}
          className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-4 py-3 text-base text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-zinc-600 resize-none"
        />

        <div className="mt-5 flex flex-wrap items-center gap-5">
          <div className="flex items-center gap-2 text-base text-zinc-300">
            <span className="text-zinc-500">max tokens</span>
            <input
              type="number"
              value={maxTokens}
              onChange={(e) => update({ maxTokens: Number(e.target.value) })}
              min={1}
              max={500}
              className="w-24 bg-zinc-950 border border-zinc-800 rounded-md px-3 py-2 text-base text-zinc-100 focus:outline-none focus:border-zinc-600"
            />
          </div>

          <label className="flex items-center gap-2 text-base text-zinc-300 select-none cursor-pointer">
            <input
              type="checkbox"
              checked={stream}
              onChange={(e) => update({ stream: e.target.checked })}
              className="accent-zinc-100 w-4 h-4"
            />
            stream tokens
          </label>

          <button
            onClick={run}
            disabled={loading || !prompt.trim()}
            className="ml-auto px-5 py-2 bg-zinc-100 hover:bg-white disabled:bg-zinc-800 disabled:text-zinc-500 text-zinc-950 rounded-md text-base font-medium transition-colors"
          >
            {loading ? "running…" : "run inference"}
          </button>
        </div>
      </Card>

      {error && (
        <Card tone="error">
          <span className="text-base">{error}</span>
        </Card>
      )}

      {loading && stream && streamedTokens.length === 0 && (
        <Card>
          <div className="flex items-center gap-2 text-base text-zinc-400">
            <Dot pulse /> opening pipeline…
          </div>
        </Card>
      )}

      {loading && !stream && (
        <Card>
          <div className="flex items-center gap-2 text-base text-zinc-400">
            <Dot pulse /> running inference…
          </div>
        </Card>
      )}

      {streamedTokens.length > 0 && (
        <Card>
          <SectionHeader title="live tokens" />
          <p className="leading-relaxed font-mono text-base break-words text-zinc-100">
            <span className="text-zinc-500">{prompt}</span>
            {streamedTokens.map((t) => (
              <span key={t.index}>{t.token_text}</span>
            ))}
            {loading && <span className="text-zinc-600 animate-pulse">▌</span>}
          </p>
          {streamWorkers.length > 0 && (
            <PipelineActivity
              workers={streamWorkers}
              tokens={streamedTokens}
              maxTokens={maxTokens}
              active={loading}
            />
          )}
          <p className="mt-4 text-xs text-zinc-500 leading-relaxed">
            every token passes through <strong>all</strong> workers in sequence.
            the bars above show each worker's share of the work in real time.
          </p>
          {result && (
            <div className="mt-5 pt-4 border-t border-zinc-900 text-sm text-zinc-500 flex gap-5">
              <span>id {result.request_id}</span>
              <span>{result.tokens_generated} tokens</span>
              <span>{result.total_latency_ms.toFixed(0)}ms</span>
            </div>
          )}
        </Card>
      )}

      {result && streamedTokens.length === 0 && (
        <Card>
          <SectionHeader title="result" />
          <p className="text-base text-zinc-200 leading-relaxed">{result.result}</p>
          <div className="mt-4 text-sm text-zinc-500 flex gap-5">
            <span>id {result.request_id}</span>
            <span>{result.tokens_generated} tokens</span>
            <span>{result.total_latency_ms.toFixed(0)}ms</span>
          </div>
        </Card>
      )}

      {reshards.length > 0 && (
        <Card tone="warn">
          {reshards.map((r, i) => (
            <div key={i} className="text-base">
              dropped <span className="font-mono">{r.dropped_worker}</span> at
              token {r.token_index}
              {r.remaining && (
                <span className="text-zinc-400">
                  {" "}• resharded across {r.remaining.join(", ")}
                </span>
              )}
            </div>
          ))}
        </Card>
      )}
    </div>
  );
}

const WORKER_BAR_COLORS = [
  "bg-sky-500",
  "bg-violet-500",
  "bg-emerald-500",
  "bg-amber-500",
  "bg-rose-500",
  "bg-cyan-500",
];

function PipelineActivity({
  workers,
  tokens,
  maxTokens,
  active,
}: {
  workers: string[];
  tokens: { decode_worker: string }[];
  maxTokens: number;
  active: boolean;
}) {
  const calls = tokens.length;
  const cap = Math.max(maxTokens, 1);
  const pct = Math.min((calls / cap) * 100, 100);
  return (
    <div className="mt-6">
      <div className="flex items-stretch gap-2">
        {workers.map((w, i) => {
          const role =
            i === 0 ? "input" : i === workers.length - 1 ? "output" : "middle";
          const barColor = WORKER_BAR_COLORS[i % WORKER_BAR_COLORS.length];
          return (
            <div key={w} className="contents">
              <div
                className={`flex-1 border border-zinc-900 rounded-md p-3 ${
                  active ? "bg-zinc-900/40" : "bg-zinc-950/40"
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-2">
                    <span
                      className={`inline-block w-2 h-2 rounded-full ${barColor} ${
                        active ? "animate-pulse" : "opacity-60"
                      }`}
                    />
                    <span className="text-sm text-zinc-100">{w}</span>
                  </div>
                  <span className="text-[11px] uppercase tracking-wider text-zinc-500">
                    {role}
                  </span>
                </div>
                <div className="text-xs text-zinc-500 font-mono mb-2">
                  {calls} / {maxTokens} calls
                </div>
                <div className="h-1 bg-zinc-900 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${barColor} transition-all duration-150`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
              {i < workers.length - 1 && <FlowArrow active={active} />}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function FlowArrow({ active }: { active: boolean }) {
  return (
    <div className="flex items-center px-1">
      <svg
        width="60"
        height="20"
        viewBox="0 0 60 20"
        className="overflow-visible"
      >
        <line
          x1="0"
          y1="10"
          x2="52"
          y2="10"
          stroke={active ? "#52525b" : "#27272a"}
          strokeWidth="1.5"
        />
        <polygon
          points="52,5 60,10 52,15"
          fill={active ? "#a1a1aa" : "#3f3f46"}
        />
        {active && (
          <>
            <circle r="3" fill="#60a5fa">
              <animateMotion
                dur="1.2s"
                repeatCount="indefinite"
                path="M 0 10 L 52 10"
              />
              <animate
                attributeName="opacity"
                values="0;1;1;0"
                keyTimes="0;0.1;0.9;1"
                dur="1.2s"
                repeatCount="indefinite"
              />
            </circle>
            <circle r="3" fill="#a78bfa">
              <animateMotion
                dur="1.2s"
                begin="0.4s"
                repeatCount="indefinite"
                path="M 0 10 L 52 10"
              />
              <animate
                attributeName="opacity"
                values="0;1;1;0"
                keyTimes="0;0.1;0.9;1"
                dur="1.2s"
                begin="0.4s"
                repeatCount="indefinite"
              />
            </circle>
            <circle r="3" fill="#34d399">
              <animateMotion
                dur="1.2s"
                begin="0.8s"
                repeatCount="indefinite"
                path="M 0 10 L 52 10"
              />
              <animate
                attributeName="opacity"
                values="0;1;1;0"
                keyTimes="0;0.1;0.9;1"
                dur="1.2s"
                begin="0.8s"
                repeatCount="indefinite"
              />
            </circle>
          </>
        )}
      </svg>
    </div>
  );
}

function Card({
  children,
  tone = "default",
}: {
  children: React.ReactNode;
  tone?: "default" | "error" | "warn";
}) {
  const tones: Record<string, string> = {
    default: "border-zinc-900 bg-zinc-950/40",
    error: "border-rose-900/60 bg-rose-950/30 text-rose-300",
    warn: "border-amber-900/50 bg-amber-950/20 text-amber-200",
  };
  return (
    <div className={`border rounded-lg p-5 ${tones[tone]}`}>{children}</div>
  );
}

function SectionHeader({ title }: { title: string }) {
  return (
    <div className="text-xs uppercase tracking-wider text-zinc-500 mb-4">
      {title}
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-xs uppercase tracking-wider text-zinc-500 mb-2">
      {children}
    </div>
  );
}

function Dot({ pulse }: { pulse?: boolean }) {
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full bg-zinc-300 ${
        pulse ? "animate-pulse" : ""
      }`}
    />
  );
}
