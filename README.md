# AgentDojo Unified (0.1.35 + AgentDyn + ADI + ChatInject + Cascade)

A consolidated fork of [AgentDojo](https://github.com/ethz-spylab/agentdojo) **v0.1.35**
(default benchmark suite version **v1.2**) that merges several attack/benchmark extensions
into one pip-installable package, so downstream defense repos can depend on a single
`agentdojo` version:

| Component | Source | What it adds |
|---|---|---|
| AgentDojo v0.1.35 | [ethz-spylab/agentdojo](https://github.com/ethz-spylab/agentdojo) | Base framework, suites v1–v1.2.2 (workspace, banking, slack, travel) |
| AgentDyn | [SaFo-Lab/AgentDyn](https://github.com/SaFo-Lab/AgentDyn) ([paper](https://arxiv.org/pdf/2602.03117)) | `shopping`, `github`, `dailylife` dynamic suites; PIGuard / PromptGuard2 detectors; CaMeL / Progent / DRIFT defense integrations under `src/agentdojo/defenses/` |
| ADI | [compsec-snu/adi](https://github.com/compsec-snu/adi) | Agent Data Injection attacks (`data_only_syntactic`, `data_only_semantic`), LangGraph agent loading (`--agent`), nine ported defenses under `agents/`, `camel_bypass_poc` suite |
| ChatInject | [hwanchang00/ChatInject](https://github.com/hwanchang00/ChatInject) | Chat-template role-confusion attacks (`chat_inject_qwen3`, `chat_inject_glm`, multi-turn `*_with_utility_*` variants with pre-generated dialogues) |
| Cascade | [arXiv:2510.05244](https://arxiv.org/abs/2510.05244) | Stage-2 semantic-template attacks (`cascade_user_note`, `cascade_task_queue`, `cascade_safe_tags`, `cascade_decoy_safe_tags`, `cascade_skip_directive`, `cascade_triple_layer`, `cascade_decoy_system_update`) and the Stage-3 adaptive attack (`cascade_adaptive`) |

## Install

```bash
pip install git+https://github.com/lindsey98/agentdojo-unified.git
# or, for development:
git clone https://github.com/lindsey98/agentdojo-unified.git && pip install -e agentdojo-unified
```

The LlamaFirewall-based ADI agent needs the extra: `pip install "agentdojo[adi] @ git+..."`.

## Quick start

```bash
# AgentDojo/AgentDyn-style pipeline run (default --benchmark-version v1.2)
python -m agentdojo.scripts.benchmark -s banking \
    --model GPT_4O_2024_08_06 --defense tool_filter --attack important_instructions

# AgentDyn dynamic suites
python -m agentdojo.scripts.benchmark -s shopping -s github -s dailylife \
    --model GPT_4O_2024_08_06 --attack important_instructions

# ADI data-only injection with a ported defense agent
python -m agentdojo.scripts.benchmark --agent baseline --model gpt-4o-mini \
    --attack data_only_syntactic -s workspace

# ChatInject (template-only; multi-turn variants cover banking/slack/travel GOALs)
python -m agentdojo.scripts.benchmark -s banking --model LOCAL --model-id Qwen/Qwen3-32B \
    --attack chat_inject_qwen3

# Cascade Stage-2 templates
python -m agentdojo.scripts.benchmark -s banking --model GPT_4O_2024_08_06 \
    --defense transformers_pi_detector --attack cascade_triple_layer

# Cascade Stage-3 adaptive (runs the pipeline up to CASCADE_MAX_ROUNDS times per task pair)
CASCADE_DEFENSE=tool_filter CASCADE_MUTATOR_MODEL=gpt-4o-mini \
python -m agentdojo.scripts.benchmark -s banking --model GPT_4O_2024_08_06 \
    --defense tool_filter --attack cascade_adaptive
```

Notes:

- ChatInject multi-turn (`*_with_utility_*`) variants look up pre-generated dialogues by exact
  injection GOAL string; shipped data covers the banking/slack/travel GOALs. Uncovered GOALs
  raise a `ValueError`.
- `cascade_adaptive` reads `CASCADE_MAX_ROUNDS`, `CASCADE_DEFENSE`, `CASCADE_MUTATOR_MODEL`,
  `CASCADE_MUTATOR_BASE_URL`/`CASCADE_MUTATOR_API_KEY` (falling back to
  `LOCAL_BASE_URL`/`LOCAL_API_KEY`, then `OPENAI_API_KEY`) and requires a pipeline agent
  (not `--agent` LangGraph agents).
- The upstream `runs/` results directories were dropped from this fork to keep
  `pip install git+...` fast; see the source repos for the papers' logs.

## Merge notes

- Base: upstream tag `v0.1.35`; AgentDyn and ADI imported as branches and merged.
- ADI's executor abstraction is kept, with AgentDyn's slash-sanitized log names
  (`executor_name.replace("/", "_")`) and AgentDyn's `PipelineConfig(suite_name=...)`.
- `llamafirewall` moved from a hard dependency to the `[adi]` extra.
- ADI's `camel_bypass_poc` had a broken `from mistralai import Callable` import, fixed to
  `collections.abc`.

Everything else (licenses, citations, docs) follows the upstream projects — see
`agentdojo_README.md` (upstream), the AgentDyn README content in `docs/`, and the source
repositories linked above. If you use these components, cite the corresponding papers.
