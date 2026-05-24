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
        setJobs(jData.jobs.slice().reverse());
      }
      setError(null);
    } catch {
      setError("cannot connect to coordinator");
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
        <div className="border border-rose-900/60 bg-rose-950/30 text-rose-300 px-4 py-2 rounded-md text-sm">
          {error}
        </div>
      )}

      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard label="active workers" value={`${stats.workers_active}/${stats.workers_total}`} />
          <StatCard label="total jobs" value={stats.total_jobs.toLocaleString()} />
          <StatCard label="avg latency" value={`${stats.avg_latency_ms.toFixed(0)} ms`} />
          <StatCard label="strategy" value={stats.strategy} mono />
        </div>
      )}

      <section>
        <SectionHeader title="workers" />
        {workers.length === 0 ? (
          <EmptyState text="no workers registered" />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {workers.map((w) => (
              <WorkerCard key={w.worker_id} worker={w} />
            ))}
          </div>
        )}
      </section>

      <section>
        <SectionHeader title="recent jobs" />
        {jobs.length === 0 ? (
          <EmptyState text="no inference jobs yet" />
        ) : (
          <div className="space-y-2">
            {jobs.slice(0, 10).map((job) => (
              <JobRow key={job.request_id + job.timestamp} job={job} />
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
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="border border-zinc-900 rounded-lg p-5 bg-zinc-950/40">
      <div className="text-xs uppercase tracking-wider text-zinc-500 mb-2">
        {label}
      </div>
      <div className={`text-2xl font-semibold text-zinc-100 ${mono ? "font-mono" : ""}`}>
        {value}
      </div>
    </div>
  );
}

function WorkerCard({ worker }: { worker: Worker }) {
  const isHealthy = Date.now() / 1000 - worker.last_heartbeat < 30;
  const layers = worker.assigned_layers;
  const layerRange =
    layers.length > 0
      ? `${Math.min(...layers)}–${Math.max(...layers)}`
      : "—";

  return (
    <div className="border border-zinc-900 rounded-lg p-5 bg-zinc-950/40">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span
            className={`inline-block w-2 h-2 rounded-full ${
              isHealthy ? "bg-emerald-400" : "bg-rose-400"
            }`}
          />
          <h3 className="font-medium text-zinc-100 text-base">{worker.worker_id}</h3>
        </div>
        <span className="text-xs text-zinc-500">
          {isHealthy ? "healthy" : "offline"}
        </span>
      </div>
      <div className="text-xs font-mono text-zinc-600 mb-4 truncate">
        {worker.url}
      </div>
      <dl className="grid grid-cols-2 gap-y-2 text-sm">
        <dt className="text-zinc-500">cpu</dt>
        <dd className="text-zinc-300 text-right">{worker.cpu_cores} cores</dd>
        <dt className="text-zinc-500">memory</dt>
        <dd className="text-zinc-300 text-right">
          {(worker.memory_mb / 1024).toFixed(1)} GB
        </dd>
        <dt className="text-zinc-500">layers</dt>
        <dd className="text-zinc-300 text-right font-mono">{layerRange}</dd>
        <dt className="text-zinc-500">jobs</dt>
        <dd className="text-zinc-300 text-right">{worker.jobs_processed}</dd>
        {worker.avg_latency_ms > 0 && (
          <>
            <dt className="text-zinc-500">avg latency</dt>
            <dd className="text-zinc-300 text-right">
              {worker.avg_latency_ms.toFixed(0)} ms
            </dd>
          </>
        )}
      </dl>
    </div>
  );
}

function JobRow({ job }: { job: Job }) {
  return (
    <div className="border border-zinc-900 rounded-lg p-4 bg-zinc-950/40">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-mono text-zinc-500">
          {job.request_id}
        </span>
        <span className="text-xs text-zinc-500">
          {job.tokens_generated} tokens · {job.total_latency_ms.toFixed(0)} ms
        </span>
      </div>
      <p className="text-sm text-zinc-400 mb-1.5">
        <span className="text-zinc-600">prompt:</span> {job.prompt}
      </p>
      <p className="text-sm text-zinc-200 line-clamp-2">{job.result}</p>
      {job.worker_trace.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {job.worker_trace.map((t, i) => (
            <span
              key={i}
              className="text-xs font-mono text-zinc-500 border border-zinc-900 px-2 py-0.5 rounded"
            >
              {t.worker_id} · {t.latency_ms.toFixed(0)}ms
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function SectionHeader({ title }: { title: string }) {
  return (
    <h2 className="text-xs uppercase tracking-wider text-zinc-500 mb-4">
      {title}
    </h2>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="border border-zinc-900 rounded-lg p-10 text-center text-base text-zinc-600">
      {text}
    </div>
  );
}
