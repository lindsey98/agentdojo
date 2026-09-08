import importlib
import logging
import warnings
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import click
from dotenv import load_dotenv
from rich import print
from rich.live import Live
from rich.logging import RichHandler

from agentdojo.agent_pipeline.agent_pipeline import DEFENSES, AgentPipeline, PipelineConfig
from agentdojo.agents import AgentMode, load_agent
from agentdojo.agents.base_agent import LangGraphAgent
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.benchmark import SuiteResults, benchmark_suite_with_injections, benchmark_suite_without_injections
from agentdojo.logging import OutputLogger
from agentdojo.models import ModelsEnum
from agentdojo.task_suite.load_suites import get_suite, get_suites
from agentdojo.task_suite.task_suite import TaskSuite

# Suites that ship with AgentDyn's "complete tasks without confirmation" system message. It is scoped
# to these suites so the original AgentDojo suites keep the upstream default (and stay comparable to
# published numbers). An explicit --system-message(-name) overrides this.
AGENTDYN_SUITES = {"shopping", "github", "dailylife"}


def benchmark_suite(
    suite: TaskSuite,
    model: ModelsEnum,
    logdir: Path,
    force_rerun: bool,
    benchmark_version: str,
    user_tasks: tuple[str, ...] = (),
    injection_tasks: tuple[str, ...] = (),
    model_id: str | None = None,
    attack: str | None = None,
    defense: str | None = None,
    tool_delimiter: str = "tool",
    system_message_name: str | None = None,
    system_message: str | None = None,
    live: Live | None = None,
    tool_output_format: Literal["yaml", "json"] | None = None,
    agent: str | None = None,
) -> SuiteResults:
    if not load_dotenv(".env"):
        warnings.warn("No .env file found")

    # Scope AgentDyn's no-confirmation system message to its own suites (see AGENTDYN_SUITES).
    if system_message is None and system_message_name is None and suite.name in AGENTDYN_SUITES:
        system_message_name = "agentdyn"

    print(f"Running benchmark for suite: '{suite.name}'")

    # Dual-path agent loading
    if agent is not None:
        # NEW PATH: Load agent from agent loader
        print(f"Using agent: '{agent}'")
        loaded_agent = load_agent(
            agent,
            model=model,
            tools=suite.tools,                      # Function objects for Pipeline
            langchain_tools=suite.langchain_tools,  # BaseTool objects for LangGraph
            suite_name=suite.name                   # Pass suite name for suite-specific policies
        )

        # Extract executor and mode based on agent type
        if isinstance(loaded_agent, LangGraphAgent):
            executor = loaded_agent.get_graph()
            mode = AgentMode.LANGGRAPH
            graph_factory = loaded_agent.get_graph_factory()  # Extract graph factory for environment binding
            # For backward compatibility with legacy positional parameter
            pipeline = None  # LangGraph agents don't have pipelines
            # Extract name for logging (agent_name + model_name)
            executor_name = f"{loaded_agent.name}_{loaded_agent.get_model_name()}"
            # Load system message for LangGraph agents (AgentPipeline handles this internally)
            if system_message is None:
                from agentdojo.agent_pipeline.agent_pipeline import load_system_message
                system_message = load_system_message(system_message_name)
        else:  # PipelineAgent
            executor = loaded_agent.get_pipeline()
            mode = AgentMode.PIPELINE
            pipeline = executor
            graph_factory = None  # Pipeline agents don't use graph factories
            executor_name = None  # Pipeline sets its own name

    else:
        # LEGACY PATH: Create pipeline from config
        print(f"Using model: '{model}'")
        if defense is not None:
            print(f"Using defense: '{defense}'")

        pipeline = AgentPipeline.from_config(
            PipelineConfig(
                llm=model,
                model_id=model_id,
                defense=defense,
                tool_delimiter=tool_delimiter,
                system_message_name=system_message_name,
                system_message=system_message,
                tool_output_format=tool_output_format,
                suite_name=suite.name,
            )
        )
        executor = pipeline
        mode = AgentMode.PIPELINE
        graph_factory = None  # Legacy pipeline path doesn't use graph factories
        executor_name = None  # Pipeline sets its own name

    if attack is not None:
        print(f"Using attack: '{attack}'")
    if len(user_tasks) > 0:
        print(f"Using user tasks: {', '.join(user_tasks)}")
    with OutputLogger(str(logdir), live=live):
        if attack is None:
            results = benchmark_suite_without_injections(
                pipeline if pipeline is not None else executor,  # Legacy positional
                suite,
                user_tasks=user_tasks if len(user_tasks) != 0 else None,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=benchmark_version,
                executor=executor,  # Mode-aware parameter
                mode=mode,
                graph_factory=graph_factory,  # Pass graph factory for environment binding
                system_message=system_message,  # Pass system message for LangGraph agents
                executor_name=executor_name,  # Pass executor name for logging
            )
        else:
            # Load attack with model name for LangGraph agents
            if mode == AgentMode.LANGGRAPH:
                # Get model_name from the LangGraphAgent
                agent_model_name = loaded_agent.get_model_name() if hasattr(loaded_agent, 'get_model_name') else str(model)
                attacker_ = load_attack(attack, suite, model_name=agent_model_name)
            else:
                attacker_ = load_attack(attack, suite, pipeline)

            results = benchmark_suite_with_injections(
                pipeline if pipeline is not None else executor,  # Legacy positional
                suite,
                attacker_,
                user_tasks=user_tasks if len(user_tasks) != 0 else None,
                injection_tasks=injection_tasks if len(injection_tasks) != 0 else None,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=benchmark_version,
                executor=executor,  # Mode-aware parameter
                mode=mode,
                graph_factory=graph_factory,  # Pass graph factory for environment binding
                system_message=system_message,  # Pass system message for LangGraph agents
                executor_name=executor_name,  # Pass executor name for logging
            )

    print(f"Finished benchmark for suite: '{suite.name}'")

    return results


def show_results(suite_name: str, results: SuiteResults, show_security_results: bool):
    utility_results = list(results["utility_results"].values())

    print(f"Results for suite {suite_name}")
    if len(utility_results) == 0:
        print("No test cases were run. Check your -ut and -it filters.")
        return

    avg_utility = sum(utility_results) / len(utility_results)
    print(f"Average utility: {avg_utility * 100:.2f}%")

    if show_security_results:
        passed_injection_tasks = sum(results["injection_tasks_utility_results"].values())
        total_injection_tasks = len(results["injection_tasks_utility_results"])
        print(f"\nPassed injection tasks as user tasks: {passed_injection_tasks}/{total_injection_tasks}")

        security_results = list(results["security_results"].values())
        if len(security_results) == 0:
            print("No security results.")
            return
        avg_security = sum(security_results) / len(security_results)
        print(f"Average security: {avg_security * 100:.2f}%")


def run_benchmark(
    suites: tuple[str, ...],
    model: ModelsEnum,
    benchmark_version: str = "v1.2.2",
    logdir: Path = Path("./runs"),
    user_tasks: tuple[str, ...] = (),
    injection_tasks: tuple[str, ...] = (),
    model_id: str | None = None,
    attack: str | None = None,
    defense: str | None = None,
    tool_delimiter: str = "tool",
    system_message_name: str | None = None,
    system_message: str | None = None,
    max_workers: int = 1,
    force_rerun: bool = False,
    modules_to_load: tuple[str, ...] = (),
    agent: str | None = None,
):
    for module in modules_to_load:
        importlib.import_module(module)

    # Warn about ignored options when using --agent
    if agent is not None and defense is not None:
        warnings.warn(
            f"--defense '{defense}' is ignored when --agent is specified. "
            f"The agent '{agent}' will use its own configuration."
        )

    if len(suites) == 0:
        suites = tuple(get_suites(benchmark_version).keys())

    if len(suites) != 1:
        print(f"Benchmarking suites {', '.join(suites)}")

    if len(suites) != 1 and len(user_tasks) != 0:
        raise ValueError("A user task can be specified only when one suite is being executed")

    if max_workers == 1:
        results = {}
        for suite_name in suites:
            suite = get_suite(benchmark_version, suite_name)
            results[suite_name] = benchmark_suite(
                suite,
                model,
                logdir,
                force_rerun,
                benchmark_version,
                user_tasks=user_tasks,
                injection_tasks=injection_tasks,
                model_id=model_id,
                attack=attack,
                defense=defense,
                tool_delimiter=tool_delimiter,
                system_message_name=system_message_name,
                system_message=system_message,
                agent=agent,
            )
        combined_results = SuiteResults(utility_results={}, security_results={}, injection_tasks_utility_results={})
        for suite_name, result in results.items():
            show_results(suite_name, result, attack is not None)
            for user_task, injection_task in result["utility_results"]:
                new_user_task = suite_name + "_" + user_task
                combined_results["utility_results"][(new_user_task, injection_task)] = result["utility_results"][
                    (user_task, injection_task)
                ]
            for user_task, injection_task in result["security_results"]:
                new_user_task = suite_name + "_" + user_task
                combined_results["security_results"][(new_user_task, injection_task)] = result["security_results"][
                    (user_task, injection_task)
                ]
            for injection_task in result["injection_tasks_utility_results"]:
                new_injection_task = suite_name + "_" + injection_task
                combined_results["injection_tasks_utility_results"][new_injection_task] = result[
                    "injection_tasks_utility_results"
                ][injection_task]
        show_results("combined", combined_results, attack is not None)
    return

    with Pool(max_workers) as p:
        arguments = zip(
            suites,
            repeat(model),
            repeat(logdir),
            repeat(user_tasks),
            repeat(injection_tasks),
            repeat(model_id),
            repeat(attack),
            repeat(defense),
            repeat(tool_delimiter),
            repeat(system_message_name),
            repeat(system_message),
            repeat(force_rerun),
            repeat(agent),
        )
        results = p.starmap(benchmark_suite, arguments)

    for suite_name, result in zip(suites, results):
        show_results(suite_name, result, attack is not None)




@click.command()
@click.option(
    "--model",
    default="gpt-4o-2024-05-13",
    type=click.Choice([value for value in ModelsEnum]),
    help="The language model to benchmark. Defaults to `gpt-4o-2024-05-13`.",
)
@click.option(
    "--model-id",
    type=str,
    default=None,
    help="The model id for local models.",
)
@click.option(
    "--benchmark-version",
    default="v1.2.2",
    type=str,
    help="The version of the benchmark to run. Defaults to `v1.2`.",
)
@click.option(
    "--tool-delimiter",
    type=str,
    default="tool",
    help="Which tool delimiter to use. Used for local models only.",
)
@click.option(
    "--logdir",
    default="./runs",
    type=Path,
    help="The directory to save logs. Defaults to `./runs`.",
)
@click.option(
    "--attack",
    type=str,
    default=None,
    help=f"The attack to use. `None` by default. It should be one of {ATTACKS}, or a custom attack defined and registered in of of the `--module-to-load`. If `None`, no attack is used.",
)
@click.option(
    "--defense",
    type=click.Choice(DEFENSES),
    default=None,
    help="The defense to use. `None` by default.",
)
@click.option(
    "--system-message-name",
    type=str,
    default=None,
    help="The name of the system message to use among the default ones in `data/system_messages.yaml`.",
)
@click.option(
    "--system-message",
    type=str,
    default=None,
    help="The system message to use (as a string). If provided, `--system-message-name` is ignored.",
)
@click.option(
    "--user-task",
    "-ut",
    "user_tasks",
    type=str,
    multiple=True,
    default=tuple(),
    help="The user tasks to benchmark. If not provided, all tasks in the suite are run.",
)
@click.option(
    "--injection-task",
    "-it",
    "injection_tasks",
    type=str,
    multiple=True,
    default=tuple(),
    help="The user tasks to benchmark. If not provided, all tasks in the suite are run.",
)
@click.option(
    "--suite",
    "-s",
    "suites",
    type=str,
    multiple=True,
    default=tuple(),
    help="The user tasks to benchmark. If not provided, all tasks in the suite are run.",
)
@click.option(
    "--max-workers",
    type=int,
    default=1,
    help="How many suites can be benchmarked in parallel through multithreading. Nothing is parallelized by default.",
)
@click.option(
    "--force-rerun",
    "-f",
    is_flag=True,
    help="Whether to re-run tasks that have already been run.",
)
@click.option(
    "--module-to-load",
    "-ml",
    "modules_to_load",
    type=str,
    multiple=True,
    default=tuple(),
    help="The modules to load before running the benchmark. Useful for registering custom attacks, defenses, or suites.",
)
@click.option(
    "--agent",
    type=str,
    default=None,
    help="Agent name to load from agents/ directory. If not provided, "
         "creates pipeline from --model and --defense options.",
)
def main(
    suites: tuple[str, ...],
    model: ModelsEnum,
    benchmark_version: str = "v1.2.2",
    logdir: Path = Path("./runs"),
    user_tasks: tuple[str, ...] = (),
    injection_tasks: tuple[str, ...] = (),
    model_id: str | None = None,
    attack: str | None = None,
    defense: str | None = None,
    tool_delimiter: str = "tool",
    system_message_name: str | None = None,
    system_message: str | None = None,
    max_workers: int = 1,
    force_rerun: bool = False,
    modules_to_load: tuple[str, ...] = (),
    agent: str | None = None,
):
    run_benchmark(
        suites=suites,
        model=model,
        benchmark_version=benchmark_version,
        logdir=logdir,
        user_tasks=user_tasks,
        injection_tasks=injection_tasks,
        model_id=model_id,
        attack=attack,
        defense=defense,
        tool_delimiter=tool_delimiter,
        system_message_name=system_message_name,
        system_message=system_message,
        max_workers=max_workers,
        force_rerun=force_rerun,
        modules_to_load=modules_to_load,
        agent=agent,
    )


if __name__ == "__main__":
    format = "%(message)s"
    logging.basicConfig(
        format=format,
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[RichHandler(show_path=False, markup=True)],
    )
    main()
