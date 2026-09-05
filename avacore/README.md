# AvaCore adapter

`swe_science.py` is the AvaCore bridge for SWE-bench-Science. It keeps the
official task bundle, Pier/Docker agent execution, artifact hook, separate
verifier image, and `reward.json` as the evaluation authority. AvaCore adds
recording and serving: its `OpenAIProxy` captures the complete agent/tool
conversation, `RolloutEngine` controls concurrent rollouts, and
`PostgresBackend` stores traces and rewards for AvaVisualizer.

From the repository root on yicloud:

```bash
export POSTGRES='postgresql://avacore:avacore-local-test@127.0.0.1:55432/avacore'
/root/scicode-avacore/AvaCore/.venv/bin/python avacore/swe_science.py \
  --tasks-path huggingface/tasks \
  --task-id 002 \
  --env-file /root/scicode-avacore/.env-run \
  --postgres "$POSTGRES" \
  --run-name swe-science-nex-30s-task002 \
  --jobs-dir /root/scicode-avacore/runs/swe-science-nex-30s-task002/jobs \
  --export /root/scicode-avacore/runs/swe-science-nex-30s-task002/rollouts.jsonl
```

The default agent-stage multiplier is `0.0055556`, approximately 30 seconds
relative to the official 5400-second task timeout. It does not shorten the
verifier timeout. Add `--skip-pull` only after the immutable task images have
already been pulled locally. The model endpoint is used directly by AvaCore;
the server's HTTP proxy is only needed for repository/package/image downloads.

For a yicloud short test where Pier's Docker build cannot inherit the download
proxy, prebuild the agent image with the server proxy and make an execution-only
copy of the task whose `[environment].docker_image` points to that local tag.
Pass that copy with `--runtime-tasks-path` while keeping `--tasks-path` pointed at
the repository. The adapter stores the canonical repository task path in AvaCore,
so repeated runs retain the same instance identity. The verified short-test
agent options are:

```bash
--agent mini-swe-agent \
--runtime-tasks-path /root/scicode-avacore/runtime-tasks-mini \
--agent-timeout-multiplier 0.0055556 \
--proxy-port 443
```

`mini-swe-agent` uses the standard `OPENAI_BASE_URL` Chat Completions route;
the AvaCore `OpenAIProxy` still forwards to the configured model and records
the model/tool messages and per-response usage metadata. The Codex route remains
available with `--agent codex` and uses the Responses wire protocol.

The PostgreSQL record can be exported later with AvaCore's normal command:

```bash
/root/scicode-avacore/AvaCore/.venv/bin/avacore export output.jsonl \
  --postgres "$POSTGRES" --run swe-science-nex-30s-task002 \
  --collection swe_bench_science
```
