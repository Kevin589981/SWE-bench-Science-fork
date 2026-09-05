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

The PostgreSQL record can be exported later with AvaCore's normal command:

```bash
/root/scicode-avacore/AvaCore/.venv/bin/avacore export output.jsonl \
  --postgres "$POSTGRES" --run swe-science-nex-30s-task002 \
  --collection swe_bench_science
```
