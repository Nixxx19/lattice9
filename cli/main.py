"""
Hivemind CLI - interact with the distributed inference system.

Usage:
    python main.py infer --prompt "Once upon a time" --max-tokens 50
    python main.py workers
    python main.py benchmark --prompts 10
"""

from __future__ import annotations

import sys
import time

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

console = Console()

DEFAULT_COORDINATOR = "http://localhost:8000"


def get_coordinator_url(ctx: click.Context) -> str:
    return ctx.obj.get("coordinator", DEFAULT_COORDINATOR)


@click.group()
@click.option("--coordinator", "-c", default=DEFAULT_COORDINATOR, help="Coordinator URL")
@click.pass_context
def cli(ctx, coordinator: str):
    """Hivemind - Distributed AI Inference CLI"""
    ctx.ensure_object(dict)
    ctx.obj["coordinator"] = coordinator


@cli.command()
@click.option("--prompt", "-p", required=True, help="Input prompt for inference")
@click.option("--max-tokens", "-m", default=50, help="Maximum tokens to generate")
@click.pass_context
def infer(ctx, prompt: str, max_tokens: int):
    """Run distributed inference on a prompt."""
    url = f"{get_coordinator_url(ctx)}/api/infer"

    console.print(Panel(f"[bold cyan]Prompt:[/] {prompt}", title="Hivemind Inference"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Distributing inference across workers...", total=None)

        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, json={"prompt": prompt, "max_tokens": max_tokens})
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError:
            console.print(f"[red]Error: Cannot connect to coordinator at {url}[/]")
            sys.exit(1)
        except httpx.HTTPStatusError as e:
            console.print(f"[red]Error: {e.response.json().get('detail', str(e))}[/]")
            sys.exit(1)

    # Display result
    console.print()
    console.print(Panel(data["result"], title="Generated Text", border_style="green"))

    # Worker trace table
    table = Table(title="Worker Trace")
    table.add_column("Worker", style="cyan")
    table.add_column("Phase", style="magenta")
    table.add_column("Layers", style="yellow")
    table.add_column("Latency", justify="right", style="green")

    for trace in data["worker_trace"]:
        layers_str = f"{min(trace['layers'])}-{max(trace['layers'])}" if trace["layers"] else "N/A"
        table.add_row(
            trace["worker_id"],
            trace["phase"],
            layers_str,
            f"{trace['latency_ms']:.1f}ms",
        )

    console.print(table)
    console.print(f"\n[dim]Request ID: {data['request_id']}  |  "
                   f"Tokens: {data['tokens_generated']}  |  "
                   f"Total: {data['total_latency_ms']:.1f}ms[/]")


@cli.command()
@click.pass_context
def workers(ctx):
    """List active workers and their status."""
    url = f"{get_coordinator_url(ctx)}/api/workers"

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        console.print(f"[red]Error: Cannot connect to coordinator[/]")
        sys.exit(1)

    worker_list = data["workers"]
    stats = data["stats"]

    if not worker_list:
        console.print("[yellow]No workers registered.[/]")
        return

    table = Table(title=f"Hivemind Workers ({stats['workers_active']}/{stats['workers_total']} active)")
    table.add_column("ID", style="cyan")
    table.add_column("URL", style="dim")
    table.add_column("CPU", justify="right")
    table.add_column("Memory", justify="right")
    table.add_column("Layers", style="yellow")
    table.add_column("Jobs", justify="right", style="green")
    table.add_column("Avg Latency", justify="right", style="magenta")
    table.add_column("Status", style="bold")

    for w in worker_list:
        layers = w["assigned_layers"]
        layers_str = f"{min(layers)}-{max(layers)}" if layers else "none"
        status_color = "green" if w["status"] == "idle" else "yellow"
        table.add_row(
            w["worker_id"],
            w["url"],
            str(w["cpu_cores"]),
            f"{w['memory_mb']}MB",
            layers_str,
            str(w["jobs_processed"]),
            f"{w['avg_latency_ms']:.1f}ms",
            f"[{status_color}]{w['status']}[/]",
        )

    console.print(table)
    console.print(f"\n[dim]Strategy: {stats['strategy']}  |  "
                   f"Total jobs: {stats['total_jobs']}  |  "
                   f"Avg latency: {stats['avg_latency_ms']:.1f}ms[/]")


@cli.command()
@click.option("--prompts", "-n", default=5, help="Number of prompts to run")
@click.option("--max-tokens", "-m", default=30, help="Max tokens per prompt")
@click.pass_context
def benchmark(ctx, prompts: int, max_tokens: int):
    """Run a benchmark across workers."""
    url = f"{get_coordinator_url(ctx)}/api/infer"

    test_prompts = [
        "The future of artificial intelligence is",
        "In a world where robots can think,",
        "The scientist discovered a new element that",
        "Deep in the ocean, a mysterious creature",
        "The space mission to Mars encountered",
        "A revolutionary new technology allows people to",
        "The ancient library contained secrets about",
        "When the machines became self-aware,",
        "The quantum computer solved the problem by",
        "In the year 2050, humanity finally",
    ]

    console.print(Panel(
        f"[bold]Running benchmark: {prompts} prompts, {max_tokens} max tokens each[/]",
        title="Hivemind Benchmark",
    ))

    results = []
    with Progress(console=console) as progress:
        task = progress.add_task("Benchmarking...", total=prompts)

        with httpx.Client(timeout=120.0) as client:
            for i in range(prompts):
                prompt = test_prompts[i % len(test_prompts)]
                start = time.time()
                try:
                    resp = client.post(url, json={"prompt": prompt, "max_tokens": max_tokens})
                    resp.raise_for_status()
                    data = resp.json()
                    elapsed = (time.time() - start) * 1000
                    results.append({
                        "prompt": prompt[:40] + "...",
                        "tokens": data["tokens_generated"],
                        "latency_ms": elapsed,
                        "workers_used": len(data["worker_trace"]),
                        "success": True,
                    })
                except Exception as e:
                    results.append({
                        "prompt": prompt[:40] + "...",
                        "tokens": 0,
                        "latency_ms": 0,
                        "workers_used": 0,
                        "success": False,
                    })
                progress.update(task, advance=1)

    # Results table
    table = Table(title="Benchmark Results")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Prompt", style="cyan", max_width=40)
    table.add_column("Tokens", justify="right")
    table.add_column("Latency", justify="right", style="green")
    table.add_column("Workers", justify="right", style="yellow")
    table.add_column("Status")

    for idx, r in enumerate(results):
        status = "[green]OK[/]" if r["success"] else "[red]FAIL[/]"
        table.add_row(
            str(idx + 1),
            r["prompt"],
            str(r["tokens"]),
            f"{r['latency_ms']:.0f}ms",
            str(r["workers_used"]),
            status,
        )

    console.print(table)

    # Summary
    successful = [r for r in results if r["success"]]
    if successful:
        avg_latency = sum(r["latency_ms"] for r in successful) / len(successful)
        total_tokens = sum(r["tokens"] for r in successful)
        total_time = sum(r["latency_ms"] for r in successful)
        throughput = (total_tokens / total_time * 1000) if total_time > 0 else 0

        console.print(f"\n[bold]Summary:[/]")
        console.print(f"  Successful: {len(successful)}/{len(results)}")
        console.print(f"  Avg latency: {avg_latency:.0f}ms")
        console.print(f"  Total tokens: {total_tokens}")
        console.print(f"  Throughput: {throughput:.1f} tokens/sec")


if __name__ == "__main__":
    cli()
