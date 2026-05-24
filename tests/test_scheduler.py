import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coordinator"))

from scheduler import Scheduler, Strategy  # noqa: E402


def register(scheduler: Scheduler, n: int, cpu_cores=None, memory_mb=None):
    for i in range(n):
        scheduler.register_worker(
            worker_id=f"w{i}",
            url=f"http://w{i}",
            cpu_cores=(cpu_cores[i] if cpu_cores else 1),
            memory_mb=(memory_mb[i] if memory_mb else 1024),
        )


def assigned(scheduler: Scheduler) -> list[list[int]]:
    return [w.assigned_layers for w in scheduler.workers.values()]


@pytest.mark.parametrize("n_workers", [1, 2, 3, 4, 5, 6, 7, 12, 13])
def test_uniform_covers_every_layer_exactly_once(n_workers):
    scheduler = Scheduler(strategy=Strategy.UNIFORM, total_layers=12)
    register(scheduler, n_workers)
    all_layers = [layer for w in assigned(scheduler) for layer in w]
    assert sorted(all_layers) == list(range(12))


@pytest.mark.parametrize("n_workers", [1, 2, 3, 4, 5, 6, 7, 12])
def test_uniform_assignments_are_contiguous(n_workers):
    scheduler = Scheduler(strategy=Strategy.UNIFORM, total_layers=12)
    register(scheduler, n_workers)
    for layers in assigned(scheduler):
        if layers:
            assert layers == list(range(layers[0], layers[-1] + 1))


def test_uniform_balances_layers():
    scheduler = Scheduler(strategy=Strategy.UNIFORM, total_layers=12)
    register(scheduler, 3)
    sizes = [len(w) for w in assigned(scheduler)]
    assert sizes == [4, 4, 4]


def test_uniform_remainder_distributes_to_earliest_workers():
    scheduler = Scheduler(strategy=Strategy.UNIFORM, total_layers=12)
    register(scheduler, 5)
    sizes = [len(w) for w in assigned(scheduler)]
    assert sizes == [3, 3, 2, 2, 2]
    assert sum(sizes) == 12


def test_capacity_assignments_remain_contiguous():
    scheduler = Scheduler(strategy=Strategy.CAPACITY, total_layers=12)
    register(scheduler, 3, cpu_cores=[8, 2, 1], memory_mb=[8192, 2048, 1024])
    layers = assigned(scheduler)
    for w_layers in layers:
        assert w_layers == list(range(w_layers[0], w_layers[-1] + 1))
    all_layers = sorted(layer for w in layers for layer in w)
    assert all_layers == list(range(12))


def test_capacity_gives_bigger_workers_more_layers():
    scheduler = Scheduler(strategy=Strategy.CAPACITY, total_layers=12)
    register(scheduler, 2, cpu_cores=[8, 1], memory_mb=[8192, 1024])
    sizes = [len(w) for w in assigned(scheduler)]
    assert sizes[0] > sizes[1]


def test_reassign_after_worker_removed():
    scheduler = Scheduler(strategy=Strategy.UNIFORM, total_layers=12)
    register(scheduler, 3)
    scheduler.remove_worker("w1")
    layers = [w.assigned_layers for w in scheduler.workers.values()]
    sizes = [len(w) for w in layers]
    assert sizes == [6, 6]
    all_layers = sorted(layer for w in layers for layer in w)
    assert all_layers == list(range(12))


def test_single_worker_gets_all_layers():
    scheduler = Scheduler(strategy=Strategy.UNIFORM, total_layers=12)
    register(scheduler, 1)
    assert assigned(scheduler)[0] == list(range(12))
