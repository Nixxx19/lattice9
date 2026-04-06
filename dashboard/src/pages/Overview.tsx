import { useEffect, useState } from "react";

interface Worker {
  worker_id: string;
  url: string;
  cpu_cores: number;
  memory_mb: number;
  status: string;
  assigned_layers: number[];
  jobs_processed: number;
  avg_latency_ms: number;
  last_heartbeat: number;
}

interface Stats {
  total_jobs: number;
  avg_latency_ms: number;
  workers_active: number;
  workers_total: number;
  strategy: string;
}

interface Job {
  request_id: string;
  prompt: string;
  result: string;
  tokens_generated: number;
  total_latency_ms: number;
  worker_trace: { worker_id: string; phase: string; latency_ms: number }[];
  timestamp: number;
}

export default function Overview() {
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      const [wRes, jRes] = await Promise.all([
        fetch("/api/workers"),
        fetch("/api/jobs"),
      ]);
      if (wRes.ok) {
        const wData = await wRes.json();
        setWorkers(wData.workers);
        setStats(wData.stats);
      }
      if (jRes.ok) {
        const jData = await jRes.json();
        setJobs(jData.jobs.reverse());
      }
      setError(null);
    } catch {
      setError("Cannot connect to coordinator");
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-8">
      {error && (
        <div className="bg-red-900/30 border border-red-700 text-red-300 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {/* Stats cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <StatCard label="Active Workers" value={`${stats.workers_active}/${stats.workers_total}`} color="indigo" />
          <StatCard label="Total Jobs" value={stats.total_jobs.toString()} color="green" />
          <StatCard label="Avg Latency" value={`${stats.avg_latency_ms.toFixed(0)}ms`} color="yellow" />
          <StatCard label="Strategy" value={stats.strategy.replace("_", " ")} color="purple" />
        </div>
      )}

      {/* Worker cards */}
      <section>
        <h2 className="text-lg font-semibold text-white mb-4">Workers</h2>
        {workers.length === 0 ? (
          <div className="text-gray-500 bg-gray-900 rounded-lg p-8 text-center">
            No workers registered. Start a worker to begin.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {workers.map((w) => (
              <WorkerCard key={w.worker_id} worker={w} />
            ))}
          </div>
        )}
      </section>

      {/* Recent jobs */}
      <section>
        <h2 className="text-lg font-semibold text-white mb-4">Recent Jobs</h2>
        {jobs.length === 0 ? (
          <div className="text-gray-500 bg-gray-900 rounded-lg p-8 text-center">
            No inference jobs yet. Submit a prompt to get started.
          </div>
        ) : (
          <div className="space-y-3">
            {jobs.slice(0, 10).map((job) => (
              <div
                key={job.request_id}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-start justify-between mb-2">
                  <span className="text-xs text-gray-500 font-mono">
                    {job.request_id}
                  </span>
                  <span className="text-xs text-green-400">
                    {job.total_latency_ms.toFixed(0)}ms
                  </span>
                </div>
                <p className="text-sm text-gray-300 mb-1">
                  <span className="text-gray-500">Prompt:</span> {job.prompt}
                </p>
                <p className="text-sm text-white mb-2 line-clamp-2">
                  {job.result}
                </p>
                <div className="flex gap-2 flex-wrap">
                  {job.worker_trace.map((t, i) => (
                    <span
                      key={i}
                      className="text-xs bg-gray-800 text-gray-400 px-2 py-0.5 rounded"
                    >
                      {t.worker_id} ({t.phase}, {t.latency_ms.toFixed(0)}ms)
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function StatCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  const colorMap: Record<string, string> = {
    indigo: "border-indigo-600 text-indigo-400",
    green: "border-green-600 text-green-400",
    yellow: "border-yellow-600 text-yellow-400",
    purple: "border-purple-600 text-purple-400",
  };
  const c = colorMap[color] || colorMap.indigo;

  return (
    <div className={`bg-gray-900 border-l-4 ${c.split(" ")[0]} rounded-lg p-4`}>
      <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">
        {label}
      </div>
      <div className={`text-2xl font-bold ${c.split(" ")[1]}`}>{value}</div>
    </div>
  );
}

function WorkerCard({ worker }: { worker: Worker }) {
  const isHealthy =
    Date.now() / 1000 - worker.last_heartbeat < 30;
  const layers = worker.assigned_layers;
  const layerRange =
    layers.length > 0
      ? `${Math.min(...layers)}-${Math.max(...layers)}`
      : "none";

  return (
    <div
      className={`bg-gray-900 border rounded-lg p-5 ${
        isHealthy
          ? "border-green-800 worker-card-active"
          : "border-red-800 opacity-60"
      }`}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-white">{worker.worker_id}</h3>
        <span
          className={`text-xs px-2 py-0.5 rounded-full ${
            isHealthy
              ? "bg-green-900 text-green-300"
              : "bg-red-900 text-red-300"
          }`}
        >
          {isHealthy ? "healthy" : "offline"}
        </span>
      </div>
      <div className="text-xs text-gray-500 mb-3 font-mono">{worker.url}</div>
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <span className="text-gray-500">CPU:</span>{" "}
          <span className="text-white">{worker.cpu_cores} cores</span>
        </div>
        <div>
          <span className="text-gray-500">Memory:</span>{" "}
          <span className="text-white">{(worker.memory_mb / 1024).toFixed(1)}GB</span>
        </div>
        <div>
          <span className="text-gray-500">Layers:</span>{" "}
          <span className="text-yellow-400">{layerRange}</span>
        </div>
        <div>
          <span className="text-gray-500">Jobs:</span>{" "}
          <span className="text-green-400">{worker.jobs_processed}</span>
        </div>
      </div>
      {worker.avg_latency_ms > 0 && (
        <div className="mt-3 text-xs text-gray-500">
          Avg latency: {worker.avg_latency_ms.toFixed(1)}ms
        </div>
      )}
    </div>
  );
}
