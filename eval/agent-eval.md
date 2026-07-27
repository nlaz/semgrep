# Agent-task evals (protocol)

Measures the actual product goal: does semgrep reduce the search round-trips an
agent needs to locate code?

## Setup

- ~20 tasks per corpus, each of the form *"Find where X happens; answer with
  `file:line`."* Written by sampling from `eval/data/*.jsonl` (use the
  `paraphrase` queries — they are the ones grep struggles with).
- Conditions (one search tool available per condition):
  1. `rg` only
  2. `semgrep` keyword only (sanity: should ≈ rg)
  3. `semgrep` full (hybrid default, indexed)
- Runner: `claude -p` headless with a restricted tool allowlist, e.g.
  `--allowedTools "Bash(rg *)"` vs `--allowedTools "Bash(semgrep *)"`, plus
  Read so the agent can verify candidates.

## Metrics per task

- success (returned location within ±10 lines of truth)
- number of search-tool invocations until success
- total tokens consumed (from claude CLI usage output)
- wall time

## Scoring

Report per condition: success rate, median searches-to-success, median
tokens. The headline claim to test: *hybrid search cuts searches-to-success
and tokens on paraphrase-style tasks without hurting exact-symbol tasks.*

Run this after the retrieval evals look sane — retrieval quality is the
input; this measures whether it translates to agent efficiency.
