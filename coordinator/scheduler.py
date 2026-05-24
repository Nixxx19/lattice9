"""
Scheduler: assigns model layers/chunks to workers based on capacity.
Supports round-robin and capacity-based strategies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Strategy(str, Enum):
    # Equal contiguous chunks. With 12 layers and 3 workers: [0-3], [4-7], [8-11].
    UNIFORM = "uniform"
    # Contiguous chunks weighted by each worker's capacity_score.
    CAPACITY = "capacity"


@dataclass
class WorkerInfo:
    worker_id: str
    url: str
    cpu_cores: int = 1
    memory_mb: int = 1024
    status: str = "idle"
    assigned_layers: list[int] = field(default_factory=list)
    jobs_processed: int = 0
    total_latency_ms: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)

    @property
    def avg_latency_ms(self) -> float:
        if self.jobs_processed == 0:
            return 0.0
        return self.total_latency_ms / self.jobs_processed

    @property
    def capacity_score(self) -> float:
        return self.cpu_cores * 1.0 + (self.memory_mb / 1024) * 0.5

    def to_dict(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "url": self.url,
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "status": self.status,
            "assigned_layers": self.assigned_layers,
            "jobs_processed": self.jobs_processed,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "last_heartbeat": self.last_heartbeat,
        }


class Scheduler:
    def __init__(self, strategy: Strategy = Strategy.UNIFORM, total_layers: int = 12):
        self.strategy = strategy
        self.total_layers = total_layers
        self.workers: dict[str, WorkerInfo] = {}

    def register_worker(self, worker_id: str, url: str, cpu_cores: int, memory_mb: int) -> WorkerInfo:
        worker = WorkerInfo(
            worker_id=worker_id,
            url=url,
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
        )
        self.workers[worker_id] = worker
        self._reassign_layers()
        return worker

    def remove_worker(self, worker_id: str) -> None:
        self.workers.pop(worker_id, None)
        self._reassign_layers()

    def heartbeat(self, worker_id: str) -> None:
        if worker_id in self.workers:
            self.workers[worker_id].last_heartbeat = time.time()

    def get_healthy_workers(self, timeout: float = 30.0) -> list[WorkerInfo]:
        now = time.time()
        return [
            w for w in self.workers.values()
            if now - w.last_heartbeat < timeout
        ]

    def _reassign_layers(self) -> None:
        # Pipeline parallelism requires each worker to own a contiguous block of
        # layers — interleaved assignments would break the forward pass because
        # block N+1 depends on block N's output.
        healthy = self.get_healthy_workers()
        if not healthy:
            return

        for w in healthy:
            w.assigned_layers = []

        n = len(healthy)
        if self.strategy == Strategy.UNIFORM:
            counts = self._uniform_counts(n)
        else:
            counts = self._capacity_counts(healthy)

        start = 0
        for worker, count in zip(healthy, counts):
            worker.assigned_layers = list(range(start, start + count))
            start += count

    def _uniform_counts(self, n_workers: int) -> list[int]:
        base = self.total_layers // n_workers
        remainder = self.total_layers % n_workers
        return [base + (1 if i < remainder else 0) for i in range(n_workers)]

    def _capacity_counts(self, healthy: list[WorkerInfo]) -> list[int]:
        total_cap = sum(w.capacity_score for w in healthy) or 1.0
        counts: list[int] = []
        assigned = 0
        for idx, worker in enumerate(healthy):
            if idx == len(healthy) - 1:
                counts.append(self.total_layers - assigned)
            else:
                count = max(1, int(self.total_layers * (worker.capacity_score / total_cap)))
                count = min(count, self.total_layers - assigned - (len(healthy) - idx - 1))
                counts.append(count)
                assigned += count
        return counts

    def get_worker_assignments(self) -> list[dict]:
        healthy = self.get_healthy_workers()
        return [
            {
                "worker_id": w.worker_id,
                "url": w.url,
                "layers": w.assigned_layers,
            }
            for w in healthy
        ]

    def record_job(self, worker_id: str, latency_ms: float) -> None:
        if worker_id in self.workers:
            self.workers[worker_id].jobs_processed += 1
            self.workers[worker_id].total_latency_ms += latency_ms

    def get_throughput_stats(self) -> dict:
        total_jobs = sum(w.jobs_processed for w in self.workers.values())
        total_latency = sum(w.total_latency_ms for w in self.workers.values())
        return {
            "total_jobs": total_jobs,
            "avg_latency_ms": round(total_latency / total_jobs, 2) if total_jobs > 0 else 0,
            "workers_active": len(self.get_healthy_workers()),
            "workers_total": len(self.workers),
            "strategy": self.strategy.value,
        }
