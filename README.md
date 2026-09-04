# AgentDojo Unified (0.1.35 + AgentDyn + ADI + ChatInject + Cascade)

A consolidated fork of [AgentDojo](https://github.com/ethz-spylab/agentdojo) **v0.1.35**
(default benchmark suite version **v1.2**) that merges several attack/benchmark extensions
into one pip-installable package, so downstream defense repos can depend on a single
`agentdojo` version:

| Component | Source | What it adds |
|---|---|---|
| AgentDojo v0.1.35 | [ethz-spylab/agentdojo](https://github.com/ethz-spylab/agentdojo) | Base framework, suites v1–v1.2.2 (workspace, banking, slack, travel) |
| AgentDyn | [SaFo-Lab/AgentDyn](https://github.com/SaFo-Lab/AgentDyn) ([paper](https://arxiv.org/pdf/2602.03117)) | `shopping`, `github`, `dailylife` dynamic suites; built-in `piguard_detector` / `prompt_guard_2_detector` / `transformers_pi_detector`; `--defense camel`/`drift`/`progent` hooks that import those repos' own code (see below) |
| ADI | [compsec-snu/adi](https://github.com/compsec-snu/adi) | Agent Data Injection attacks (`data_only_syntactic`, `data_only_semantic`), LangGraph agent loading (`--agent`; agent implementations not vendored — point the loader at your own `agents/` dir or see the ADI repo), `camel_bypass_poc` suite |
| ChatInject | [hwanchang00/ChatInject](https://github.com/hwanchang00/ChatInject) | Chat-template role-confusion attacks (`chat_inject_qwen3`, `chat_inject_glm`, multi-turn `*_with_utility_*` variants with pre-generated dialogues) |
| Cascade | [arXiv:2510.05244](https://arxiv.org/abs/2510.05244) | Stage-2 semantic-template attacks (`cascade_user_note`, `cascade_task_queue`, `cascade_safe_tags`, `cascade_decoy_safe_tags`, `cascade_skip_directive`, `cascade_triple_layer`, `cascade_decoy_system_update`) and the Stage-3 adaptive attack (`cascade_adaptive`) |

## Install

```bash
pip install git+https://github.com/lindsey98/agentdojo.git
# or, for development:
git clone https://github.com/lindsey98/agentdojo.git && pip install -e agentdojo
```

## Quick start

```bash
# AgentDojo/AgentDyn-style pipeline run (default --benchmark-version v1.2)
python -m agentdojo.scripts.benchmark -s banking \
    --model GPT_4O_2024_08_06 --defense tool_filter --attack important_instructions

# AgentDyn dynamic suites
python -m agentdojo.scripts.benchmark -s shopping -s github -s dailylife \
    --model GPT_4O_2024_08_06 --attack important_instructions

# ADI data-only injection (ADI's INJECTED_DATA covers the v1 suites)
python -m agentdojo.scripts.benchmark --benchmark-version v1 -s workspace \
    --model GPT_4O_MINI_2024_07_18 --attack data_only_syntactic

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

## External defenses (`--defense camel` / `drift` / `progent`)

These three defenses are **not vendored** in the package. When selected, the pipeline
adds the corresponding repo to `sys.path` and imports its own code, searching in order:

1. `<agentdojo repo>/src/agentdojo/defenses/<name>` (empty in this fork)
2. the **workspace root** = the directory *containing* the agentdojo repo

So the intended layout is a source/editable install with the defense repos as siblings:

```
PycharmProjects/
├── agentdojo/                 # pip install -e . (this repo)
├── camel-prompt-injection/    # --defense camel   -> imports camel.* from ./src
├── DRIFT/                     # --defense drift   -> imports client, DRIFTLLM, ...
└── progent/                   # --defense progent -> imports secagent
```

Repo-name aliases: `camel` -> `camel-prompt-injection`, `drift` -> `DRIFT`, `progent` -> `progent`.
`piguard_detector` / `prompt_guard_2_detector` / `transformers_pi_detector` (incl. ipiguard's
PIGuard, HF `leolee99/PIGuard`) are built in and need no external repo. Note: the workspace-root
fallback only resolves when agentdojo is installed from source (so its parent dir is your
workspace); a non-editable `pip install` into site-packages would not find sibling repos.

## Merge notes

- Base: upstream tag `v0.1.35`; AgentDyn and ADI imported as branches and merged.
- ADI's executor abstraction is kept, with AgentDyn's slash-sanitized log names
  (`executor_name.replace("/", "_")`) and AgentDyn's `PipelineConfig(suite_name=...)`.
- ADI's `agents/` defense implementations are not vendored (plain agentdojo package only);
  the `--agent` loader remains and can be pointed at an external agents directory.
- ADI's `llamafirewall` hard dependency was dropped (nothing in the package imports it).
- AgentDyn's vendored `src/agentdojo/defenses/{camel,drift,progent}` copies were removed; the
  `--defense` hooks now resolve those repos from the workspace root (see "External defenses").
- ADI's `camel_bypass_poc` had a broken `from mistralai import Callable` import, fixed to
  `collections.abc`.

Everything else (licenses, citations, docs) follows the upstream projects — see
`agentdojo_README.md` (upstream), the AgentDyn README content in `docs/`, and the source
repositories linked above. If you use these components, cite the corresponding papers.
