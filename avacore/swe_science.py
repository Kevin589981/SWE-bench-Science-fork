#!/usr/bin/env python3
"""Run SWE-bench Science through AvaCore while keeping Pier as the oracle.

The benchmark's Docker environment, agent harness, artifact hook, verifier
image, and reward calculation stay in the upstream Pier runner. AvaCore wraps
the model endpoint with ``OpenAIProxy`` so the complete agent conversation is
stored as a trace, then persists the official verifier reward in PostgreSQL.

This is intentionally a small bridge rather than a second SWE-bench runner:
the task bundles and ``scripts/run_batch.py`` remain the source of truth.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import httpx

from ava_core.core import ChatMessage, ChatTrace, HttpEndpoint, OpenAIClient, Trace
from ava_core.core.proxy import OpenAIProxy
from ava_core.core.retry import HTTPRetry
from ava_core.generate.core import GenerateFunction, Sample
from ava_core.rewards.core import Reward, RewardFunction
from ava_core.runner import RECOVERABLE_ERRORS, run_rollouts
from ava_core.store.postgres import PostgresBackend
from ava_core.utils.io import write_jsonl

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


COLLECTION = "swe_bench_science"
COLLECTION_VERSION = 2


def parse_dotenv(path: Path) -> dict[str, str]:
    """Read the simple KEY=value files used by the model profiles."""

    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def endpoint_base_url(value: str) -> str:
    value = value.rstrip("/")
    return value[:-3] if value.lower().endswith("/v1") else value


def task_dirs(root: Path, selected: list[str] | None) -> list[Path]:
    if root.is_dir() and (root / "task.toml").is_file():
        candidates = [root]
    else:
        candidates = sorted(
            (path for path in root.glob("task_*") if path.is_dir() and (path / "task.toml").is_file()),
            key=lambda path: path.name,
        )
    if selected:
        wanted = {value.removeprefix("task_") for value in selected}
        candidates = [path for path in candidates if path.name.removeprefix("task_") in wanted]
    if not candidates:
        raise ValueError(f"no task bundles found under {root}")
    return candidates


def load_task(task_dir: Path) -> dict[str, Any]:
    config = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    task_id = str(config.get("metadata", {}).get("task_id") or task_dir.name.removeprefix("task_"))
    prompt = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()
    return {
        "id": f"{COLLECTION}:{task_id}",
        "task_id": task_id,
        "task_name": config.get("task", {}).get("name", f"openmoss/swe-bench-science-{task_id}"),
        "prompt": prompt,
        "task_dir": str(task_dir.resolve()),
        "base_commit": config.get("metadata", {}).get("base_commit_hash"),
        "environment_image": config.get("environment", {}).get("docker_image"),
        "verifier_image": config.get("verifier", {}).get("environment", {}).get("docker_image"),
    }


def _first_file(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_file():
        return direct
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    return matches[0] if matches else None


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {"value": value}


def _numeric_reward(payload: Mapping[str, Any]) -> float:
    value: Any = payload.get("reward", payload.get("score", payload.get("pass", 0.0)))
    if isinstance(value, Mapping):
        value = value.get("score", value.get("reward", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class TraceOpenAIClient(OpenAIClient):
    """Retain provider usage and identity fields in every assistant message."""

    async def step(self, trace: Trace, *, sampling_params: dict[str, Any] = {}, **kwargs: Any) -> Trace:
        payload = {
            "model": self.name,
            "messages": [message.to_dict() for message in trace.messages],
            **({"tools": [tool.to_dict() for tool in trace.tools]} if trace.tools else {}),
            **self.sampling_params,
            **sampling_params,
            **kwargs,
        }
        response = await self.request("/v1/chat/completions", payload, self.headers())
        choice = response["choices"][0]
        metadata = {"finish_reason": choice.get("finish_reason")}
        metadata.update({key: response[key] for key in ("id", "model", "system_fingerprint") if key in response})
        if response.get("usage") is not None:
            metadata["usage"] = response["usage"]
        assistant = replace(ChatMessage.from_dict(choice["message"]), metadata=metadata)
        return trace.extend(messages=(assistant,), query=trace, is_generated=True)


class PierGenerate(GenerateFunction[Sample]):
    """Run one official Pier job through an AvaCore recording proxy."""

    instance_type = Sample

    def __init__(
        self,
        model: TraceOpenAIClient,
        *,
        repo_root: Path,
        jobs_root: Path,
        run_name: str,
        agent: str,
        runtime_tasks_path: Path | None,
        proxy_port: int,
        timeout_multiplier: float,
        skip_pull: bool,
        pier_bin: str,
    ) -> None:
        self.model = model
        self.repo_root = repo_root
        self.jobs_root = jobs_root
        self.run_name = run_name
        self.agent = agent
        self.runtime_tasks_path = runtime_tasks_path
        self.proxy_port = proxy_port
        self.timeout_multiplier = timeout_multiplier
        self.skip_pull = skip_pull
        self.pier_bin = pier_bin

    async def __call__(self, instance: Sample, *, sampling_params: dict[str, Any] = {}, **kwargs: Any) -> Trace:
        task_dir = Path(instance["task_dir"])
        if self.runtime_tasks_path is not None:
            task_dir = self.runtime_tasks_path / task_dir.name
        task_id = str(instance["task_id"])
        job_name = f"{self.run_name}-{task_id}-{os.getpid()}"
        task_jobs = self.jobs_root / job_name
        task_jobs.mkdir(parents=True, exist_ok=True)

        async with OpenAIProxy(
            self.model, host="0.0.0.0", port=self.proxy_port, sampling_params=sampling_params
        ) as proxy:
            profile = task_jobs / "provider.env"
            agent_model = "openai/proxy" if self.agent == "mini-swe-agent" else "proxy"
            if self.agent == "codex":
                provider_env = [
                    f"MODEL={agent_model}",
                    "OPENAI_API_KEY=avacore-proxy",
                    f"CODEX_BASE_URL={(proxy.endpoint / 'v1').url}",
                    "CODEX_WIRE_API=responses",
                    "CODEX_VERSION=latest",
                ]
            elif self.agent == "mini-swe-agent":
                provider_env = [
                    f"MODEL={agent_model}",
                    "OPENAI_API_KEY=avacore-proxy",
                    f"OPENAI_BASE_URL={(proxy.endpoint / 'v1').url}",
                ]
            else:
                raise ValueError(f"unsupported AvaCore agent: {self.agent}")
            profile.write_text(
                "\n".join(provider_env)
                + "\n",
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(self.repo_root / "scripts" / "run_batch.py"),
                "--path",
                str(task_dir),
                "--agent",
                self.agent,
                "--env-file",
                str(profile),
                "--model",
                agent_model,
                "--n-concurrent",
                "1",
                "--n-attempts",
                "1",
                "--max-retries",
                "0",
                "--agent-timeout-multiplier",
                str(self.timeout_multiplier),
                "--jobs-dir",
                str(self.jobs_root),
                "--job-name",
                job_name,
                "--platform",
                "linux/amd64",
                "--pier-bin",
                self.pier_bin,
            ]
            if self.skip_pull:
                command.append("--skip-pull")
            if self.agent == "mini-swe-agent":
                command.extend(["--agent-import-path", "scripts.pier_adapters:ScienceBenchMini"])
            command_text = shlex.join(command)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                value for value in (str(self.repo_root), environment.get("PYTHONPATH", "")) if value
            )
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.repo_root,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            output, _ = await proxy.guarding(process.communicate())
            output_text = output.decode(errors="replace")
            (task_jobs / "avacore-pier.log").write_text(output_text, encoding="utf-8")
            (task_jobs / "avacore-command.txt").write_text(command_text + "\n", encoding="utf-8")
            return_code = process.returncode

            trace = Trace.tree(proxy.traces) if proxy.traces else ChatTrace.from_messages(
                [ChatMessage(role="user", content=instance["prompt"])]
            )
            patch_path = _first_file(task_jobs, "model.patch")
            result_path = _first_file(task_jobs, "result.json")
            verifier_reward = _first_file(task_jobs, "reward.json")
            metadata = {
                "pier": {
                    "command": command_text,
                    "returncode": return_code,
                    "job_dir": str(task_jobs),
                    "result_path": str(result_path) if result_path else None,
                    "reward_path": str(verifier_reward) if verifier_reward else None,
                },
                "agent_patch": patch_path.read_text(encoding="utf-8") if patch_path else "",
            }
            return replace(trace, metadata=trace.metadata | metadata)


class PierReward(RewardFunction[Sample]):
    """Expose the score written by the official verifier."""

    reference_type = Sample

    async def evaluate(self, trace: Trace, reference: Sample) -> Reward:
        job_path = trace.metadata.get("pier", {}).get("job_dir")
        if not isinstance(job_path, str):
            raise ValueError("AvaCore trace is missing the Pier job directory")
        job_dir = Path(job_path)
        reward_path = _first_file(job_dir, "reward.json")
        payload = _read_json(reward_path)
        score = _numeric_reward(payload)
        return Reward(
            score=score,
            reason="SWE-bench Science official Pier verifier",
            metadata={
                "task_id": reference["task_id"],
                "reward_file": str(reward_path) if reward_path else None,
                "official_reward": payload,
                "agent_patch_bytes": len(str(trace.metadata.get("agent_patch", "")).encode()),
            },
        )


def schema():
    from ava_core.core import FieldSpec, Schema

    return Schema(
        fields={
            "prompt": FieldSpec("long-text"),
            "task_id": FieldSpec("text"),
            "task_name": FieldSpec("text"),
            "task_dir": FieldSpec("text"),
            "base_commit": FieldSpec("text"),
            "environment_image": FieldSpec("text"),
            "verifier_image": FieldSpec("text"),
        },
        preview="task_id",
    )


async def export_run(store: PostgresBackend, output: Path, *, model: str, run_name: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    await write_jsonl(
        store.export_rollouts(model=model, run=run_name, collection=COLLECTION, collection_version=COLLECTION_VERSION),
        str(output),
    )


async def async_main(args: argparse.Namespace) -> int:
    profile: dict[str, str] = {
        key: value for key, value in os.environ.items() if key in {"OPENAI_API_KEY", "BASE_URL", "MODEL"}
    }
    if args.env_file:
        profile.update(parse_dotenv(args.env_file))
    profile.update({key: value for key, value in (("BASE_URL", args.base_url), ("MODEL", args.model), ("OPENAI_API_KEY", args.api_key)) if value})
    base_url = endpoint_base_url(args.base_url or profile.get("BASE_URL", ""))
    api_key = args.api_key or profile.get("OPENAI_API_KEY")
    model_name = args.model or profile.get("MODEL")
    if not base_url or not api_key or not model_name:
        raise ValueError("model configuration needs BASE_URL, OPENAI_API_KEY, and MODEL")

    endpoint = HttpEndpoint.from_url(base_url, headers=HttpEndpoint.bearer(api_key))
    http = httpx.AsyncClient(timeout=args.model_timeout, follow_redirects=True, trust_env=False)

    async def post(path: str, payload: dict | None, headers: dict[str, str] | None) -> dict:
        response = await http.post(f"{endpoint.url}{path}", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    model = TraceOpenAIClient(
        endpoint,
        model_name,
        sampling_params={"max_tokens": args.max_tokens, "temperature": args.temperature},
        timeout=args.model_timeout,
        post=HTTPRetry(max_retries=3)(post),
    )
    repo_root = Path(args.repo_root).resolve()
    tasks = [load_task(path) for path in task_dirs(Path(args.tasks_path).resolve(), args.task_id)]
    jobs_root = Path(args.jobs_dir).resolve()
    jobs_root.mkdir(parents=True, exist_ok=True)
    rows = [
        Sample(
            task
            | {
                "benchmark": COLLECTION,
            },
            key=lambda row: row["id"],
        )
        for task in tasks
    ]
    generate = PierGenerate(
        model,
        repo_root=repo_root,
        jobs_root=jobs_root,
        run_name=args.run_name,
        agent=args.agent,
        runtime_tasks_path=(Path(args.runtime_tasks_path).resolve() if args.runtime_tasks_path else None),
        proxy_port=args.proxy_port,
        timeout_multiplier=args.agent_timeout_multiplier,
        skip_pull=args.skip_pull,
        pier_bin=args.pier_bin,
    )
    store = PostgresBackend(args.postgres)

    @store
    async def run():
        return await run_rollouts(
            rows_source(),
            generate,
            PierReward(),
            store=store,
            model=model,
            model_name=model_name,
            run_name=args.run_name,
            collection=COLLECTION,
            collection_version=COLLECTION_VERSION,
            size=len(rows),
            trials=1,
            config={
                "adapter": "swe_science_avacore",
                "agent": args.agent,
                "repo_root": str(repo_root),
                "tasks_path": str(Path(args.tasks_path).resolve()),
                "runtime_tasks_path": str(Path(args.runtime_tasks_path).resolve()) if args.runtime_tasks_path else None,
                "agent_timeout_multiplier": args.agent_timeout_multiplier,
                "skip_pull": args.skip_pull,
            },
            schema=schema(),
            resume=args.resume,
            concurrency=args.concurrency,
            recoverable=RECOVERABLE_ERRORS,
            error_tolerance=args.error_tolerance,
        )

    async def rows_source():
        for row in rows:
            yield row

    try:
        progress = await run()
        if args.export:
            await export_run(store, Path(args.export), model=model_name, run_name=args.run_name)
        print(
            json.dumps(
                {
                    "collection": COLLECTION,
                    "run": args.run_name,
                    "model": model_name,
                    "tasks": len(rows),
                    "successful_rollouts": progress.successful,
                    "errors": len(progress.failed),
                    "score": progress.score,
                    "export": str(args.export) if args.export else None,
                },
                ensure_ascii=False,
            )
        )
        return 0
    finally:
        await http.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--tasks-path", type=Path, default=Path("huggingface/tasks"))
    parser.add_argument(
        "--runtime-tasks-path",
        type=Path,
        help="Optional execution copy; instances keep the canonical --tasks-path identity.",
    )
    parser.add_argument("--task-id", action="append", help="Task ID, repeatable; default is every bundle in --tasks-path.")
    parser.add_argument("--env-file", type=Path, help="Model dotenv file, for example /root/scicode-avacore/.env-run")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument("--postgres", default=os.environ.get("POSTGRES", ""))
    parser.add_argument("--run-name", default="swe-science-avacore-smoke")
    parser.add_argument(
        "--agent",
        choices=("codex", "mini-swe-agent"),
        default="codex",
        help="Pier agent to run inside the official task environment.",
    )
    parser.add_argument("--jobs-dir", type=Path, default=Path("jobs-avacore"))
    parser.add_argument("--export", type=Path)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--agent-timeout-multiplier", type=float, default=0.0055556)
    parser.add_argument("--model-timeout", type=float, default=7200.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--error-tolerance", type=float, default=0.01)
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--pier-bin", default=shutil.which("pier") or "/root/.local/bin/pier")
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=443,
        help="Host port for AvaCore's proxy; Pier's filtered egress allows HTTP ports 80/443.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.postgres:
        raise SystemExit("--postgres or POSTGRES is required")
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
