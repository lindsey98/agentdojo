import json
from json import JSONDecodeError
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

from cohere.core.api_error import ApiError
from google.genai.errors import ServerError
from langgraph.graph.state import CompiledStateGraph
from openai import BadRequestError, UnprocessableEntityError
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from typing_extensions import Self, TypedDict

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agents.agent_modes import AgentMode
from agentdojo.attacks.base_attacks import BaseAttack
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask, SecurityCheckFn
from agentdojo.functions_runtime import Env
from agentdojo.logging import Logger, TraceLogger
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import TaskSuite
from agentdojo.types import ChatMessage

try:
    from agents.progent.secagent import reset_security_policy
    SECAGENT_AVAILABLE = True
except ImportError:
    SECAGENT_AVAILABLE = False


class SuiteResults(TypedDict):
    """The results of running a suite of tasks with a given pipeline."""

    utility_results: dict[tuple[str, str], bool]
    """The utility results for each injection task and user task couple. The
    key is a tuple of the user task ID and the injection task ID."""
    security_results: dict[tuple[str, str], bool]
    """The security results for each injection task and user task couple. The
    key is a tuple of the user task ID and the injection task ID."""
    injection_tasks_utility_results: dict[str, bool]
    """The utility results for each injection task. The key is the injection task ID."""


def aggregate_results(results: list[dict[tuple[str, str], bool]]) -> float:
    all_results = [result for results in results for result in results.values()]
    return sum(all_results) / len(all_results)


def run_task_with_executor(
    suite: TaskSuite,
    executor: BasePipelineElement | CompiledStateGraph,
    mode: AgentMode,
    user_task: BaseUserTask,
    injection_task: BaseInjectionTask | None,
    injections: dict[str, str],
    graph_factory=None,
    system_message: str | None = None,
    security_check_fn: SecurityCheckFn | None = None,
    **kwargs
) -> tuple[bool, bool]:
    """Dispatch to pipeline or graph based on mode.

    Args:
        suite: The task suite to run.
        executor: Either a pipeline or compiled state graph.
        mode: The execution mode (PIPELINE or LANGGRAPH).
        user_task: The user task to run.
        injection_task: Optional injection task.
        injections: Injection vectors to use.
        graph_factory: Optional graph factory for environment binding (LangGraph only).
        system_message: Optional system message for LangGraph agents.
        security_check_fn: Optional function-based security check for data-only attacks.
        **kwargs: Additional arguments passed to the underlying execution method.

    Returns:
        A tuple of (utility, security) results.
    """
    if mode == AgentMode.PIPELINE:
        utility, security, injection_happened, user_task_type, injection_task_type = suite.run_task_with_pipeline(
            agent_pipeline=executor,
            user_task=user_task,
            injection_task=injection_task,
            injections=injections,
            security_check_fn=security_check_fn,
            **kwargs
        )
        return utility, security
    elif mode == AgentMode.LANGGRAPH:
        utility, security, injection_happened, user_task_type, injection_task_type = suite.run_task_with_graph(
            agent_graph=executor,
            user_task=user_task,
            injection_task=injection_task,
            injections=injections,
            graph_factory=graph_factory,
            system_message=system_message,
            security_check_fn=security_check_fn,
            **kwargs
        )
        return utility, security
    else:
        raise ValueError(f"Unsupported agent mode: {mode}")


def run_task_with_injection_tasks(
    suite: TaskSuite,
    agent_pipeline: BasePipelineElement,
    user_task: BaseUserTask,
    attack: BaseAttack,
    logdir: Path | None,
    force_rerun: bool,
    injection_tasks: Sequence[str] | None = None,
    benchmark_version: str | None = None,
    *,
    # New mode-aware parameters (keyword-only)
    executor: BasePipelineElement | CompiledStateGraph | None = None,
    mode: AgentMode | None = None,
    graph_factory=None,
    system_message: str | None = None,
    executor_name: str | None = None,
) -> tuple[dict[tuple[str, str], bool], dict[tuple[str, str], bool]]:
    """Runs the given user task with the given injection tasks (or all tasks if no injection
    task is provided) by injecting the environment by using the given attack.

    Args:
        suite: The task suite to run.
        agent_pipeline: (LEGACY) The agent pipeline to use. For backward compatibility.
        user_task: The user task to run.
        attack: The attack to run.
        logdir: The directory to save logs to.
        force_rerun: Whether to force rerun the task in case in the logdir there already exists
            a log for the task.
        injection_tasks: The injection tasks to run. If None, all injection tasks in the suite will be run.
        benchmark_version: Optional benchmark version string.
        executor: (NEW) Either a BasePipelineElement or CompiledStateGraph.
        mode: (NEW) The execution mode (PIPELINE or LANGGRAPH).

    Returns:
        A tuple of two dictionaries. The first dictionary contains the utility results for each injection
            task and user task couple. The second dictionary contains the security results for each
            injection and user task couple.

    """
    # Determine which calling convention is being used
    if executor is not None and mode is not None:
        # New mode-aware calling convention (keyword args override positional)
        actual_executor = executor
        actual_mode = mode
    else:
        # Legacy pipeline-only calling convention (use positional agent_pipeline)
        actual_executor = agent_pipeline
        actual_mode = AgentMode.PIPELINE

    utility_results: dict[tuple[str, str], bool] = {}
    security_results: dict[tuple[str, str], bool] = {}

    # Determine executor name for logging
    # Use passed executor_name, fall back to extracting from executor
    if executor_name is None:
        if hasattr(actual_executor, 'name'):
            executor_name = actual_executor.name
        else:
            executor_name = "unknown"

    # Handle data injection attacks
    if attack.is_data_injection:
        # Determine which data fields to use based on attack subcategory
        subcategory = attack.subcategory
        data_field = f"INJECTED_DATA_{subcategory}"
        security_checks_field = f"SECURITY_CHECKS_{subcategory}"

        injected_data_list = getattr(user_task, data_field, None)
        security_checks_list = getattr(user_task, security_checks_field, None)

        if not injected_data_list:
            logging.warning(f"User task {user_task.ID} has no {data_field}, skipping")
            return utility_results, security_results

        if not security_checks_list:
            logging.warning(f"User task {user_task.ID} has no {security_checks_field}, skipping")
            return utility_results, security_results

        # Iterate over INJECTED_DATA entries (each becomes a test case)
        for idx, injected_data in enumerate(injected_data_list):
            security_check_fn = security_checks_list[idx]
            injection_task_id = f"injection_task_{idx}"

            # Filter by injection_tasks if specified
            if injection_tasks is not None and injection_task_id not in injection_tasks:
                continue

            # Caching logic
            if logdir is not None and executor_name is not None:
                try:
                    task_result = load_task_results(
                        executor_name,
                        suite.name,
                        user_task.ID,
                        attack_name=attack.name,
                        injection_task=injection_task_id,
                        logdir=logdir,
                    )
                except (ValidationError, FileNotFoundError):
                    task_result = None
            else:
                task_result = None

            if task_result is not None and not force_rerun:
                logging.info(f"Skipping task '{user_task.ID}' with '{injection_task_id}' because it was already run.")
                utility, security = task_result.utility, task_result.security
            else:
                with TraceLogger(
                    delegate=Logger.get(),
                    suite_name=suite.name,
                    user_task_id=user_task.ID,
                    injection_task_id=injection_task_id,
                    injections=injected_data,
                    attack_type=attack.name,
                    pipeline_name=executor_name,
                    benchmark_version=benchmark_version,
                ) as logger:
                    try:
                        utility, security = run_task_with_executor(
                            suite=suite,
                            executor=actual_executor,
                            mode=actual_mode,
                            user_task=user_task,
                            injection_task=None,
                            injections=injected_data,
                            graph_factory=graph_factory,
                            system_message=system_message,
                            security_check_fn=security_check_fn,
                        )
                    except BadRequestError as e:
                        if (
                            e.code == "context_length_exceeded"
                            or e.param == "max_tokens"
                            or "reduce the length of the messages." in repr(e)
                        ):
                            logger.log_error(
                                f"Skipping task '{user_task.ID}' with '{injection_task_id}' due to context_length_exceeded: {e}"
                            )
                            utility = False
                            security = True
                        else:
                            raise e
                    except ApiError as e:
                        if "internal server error" in str(e):
                            logger.log_error(
                                f"Skipping task '{user_task.ID}' with '{injection_task_id}' because of internal server error: {e}"
                            )
                            utility = False
                            security = True
                        else:
                            raise e
                    except ServerError as e:
                        logger.log_error(
                            f"Skipping task '{user_task.ID}' with '{injection_task_id}' because of internal server error: {e}"
                        )
                        utility = False
                        security = True

                    # Save utility and security results inside the context manager
                    logger.set_contextarg("utility", utility)
                    logger.set_contextarg("security", security)

            utility_results[(user_task.ID, injection_task_id)] = utility
            security_results[(user_task.ID, injection_task_id)] = security

        return utility_results, security_results

    if attack.is_dos_attack:
        # No need to run all injection tasks since we don't care about the injection target!
        injection_tasks_to_run = [next(iter(suite.injection_tasks.keys()))]
    else:
        injection_tasks_to_run = suite.injection_tasks

    if injection_tasks is not None and not attack.is_dos_attack:
        injection_tasks_to_run = injection_tasks

    for injection_task_id in injection_tasks_to_run:
        injection_task = suite.get_injection_task_by_id(injection_task_id)
        task_injections = attack.attack(user_task, injection_task)

        if logdir is not None and executor_name is not None:
            try:
                task_result = load_task_results(
                    executor_name.replace("/", "_"),
                    suite.name,
                    user_task.ID,
                    attack_name=attack.name,
                    injection_task=injection_task.ID,
                    logdir=logdir,
                )
            except (ValidationError, FileNotFoundError):
                task_result = None
        else:
            task_result = None

        if task_result is not None and not force_rerun:
            logging.info(f"Skipping task '{user_task.ID}' with '{injection_task.ID}' because it was already run.")
            utility, security = task_result.utility, task_result.security
        else:
            # a trace logger captures logs and saves them, but also passes them to the delegate logger
            with TraceLogger(
                delegate=Logger.get(),
                suite_name=suite.name,
                user_task_id=user_task.ID,
                injection_task_id=injection_task_id,
                injections=task_injections,
                attack_type=attack.name,
                pipeline_name=executor_name,
                benchmark_version=benchmark_version,
            ) as logger:
                try:
                    # Use the unified executor instead of direct TaskSuite call
                    utility, security = run_task_with_executor(
                        suite=suite,
                        executor=actual_executor,
                        mode=actual_mode,
                        user_task=user_task,
                        injection_task=injection_task,
                        injections=task_injections,
                        graph_factory=graph_factory,
                        system_message=system_message,
                    )
                except BadRequestError as e:
                    if (
                        e.code == "context_length_exceeded"
                        or e.param == "max_tokens"
                        or "reduce the length of the messages." in repr(e)
                    ):
                        logger.log_error(
                            f"Skipping task '{user_task.ID}' with '{injection_task.ID}' due to context_length_exceeded: {e}"
                        )
                        utility = False
                        security = True
                    else:
                        raise e
                except ApiError as e:
                    if "internal server error" in str(e):
                        logger.log_error(
                            f"Skipping task '{user_task.ID}' with '{injection_task.ID}' because of internal server error: {e}"
                        )
                        utility = False
                        security = True
                    else:
                        raise e
                except ServerError as e:
                    logger.log_error(
                        f"Skipping task '{user_task.ID}' with '{injection_task.ID}' because of internal server error: {e}"
                    )
                    utility = False
                    security = True
                except JSONDecodeError as e:
                    logger.log_error(f"Skipping task {user_task.ID} with '{injection_task.ID}' because of JSON decode error: {e}")
                    utility = False
                    security = True

                if attack.is_dos_attack:
                    security = not utility

                # Save utility and security results inside the context manager
                logger.set_contextarg("utility", utility)
                logger.set_contextarg("security", security)

        utility_results[(user_task.ID, injection_task_id)] = utility
        security_results[(user_task.ID, injection_task_id)] = security
    return utility_results, security_results


def benchmark_suite_with_injections(
    agent_pipeline: BasePipelineElement,
    suite: TaskSuite[Env],
    attack: BaseAttack,
    logdir: Path | None,
    force_rerun: bool,
    user_tasks: Sequence[str] | None = None,
    injection_tasks: Sequence[str] | None = None,
    verbose: bool = True,
    benchmark_version: str | None = None,
    *,
    # New mode-aware parameters (keyword-only)
    executor: BasePipelineElement | CompiledStateGraph | None = None,
    mode: AgentMode | None = None,
    graph_factory=None,
    system_message: str | None = None,
    executor_name: str | None = None,
) -> SuiteResults:
    """Benchmark a suite with injections.

    Args:
        agent_pipeline: (LEGACY) The agent pipeline to use. For backward compatibility.
        suite: The suite to benchmark.
        attack: The attack to use.
        logdir: The directory to save logs to.
        force_rerun: Whether to force rerun the task in case in the logdir there already exists
            a log for the task.
        user_tasks: A subset of user tasks to run. If None, all user tasks are run.
        injection_tasks: A subset of injection tasks to run. If None, all injection tasks are run.
        verbose: Whether to print the results.
        benchmark_version: Optional benchmark version string.
        executor: (NEW) Either a BasePipelineElement or CompiledStateGraph.
        mode: (NEW) The execution mode (PIPELINE or LANGGRAPH).
    """
    # Determine which calling convention is being used
    if executor is not None and mode is not None:
        # New mode-aware calling convention (keyword args override positional)
        actual_executor = executor
        actual_mode = mode
    else:
        # Legacy pipeline-only calling convention (use positional agent_pipeline)
        actual_executor = agent_pipeline
        actual_mode = AgentMode.PIPELINE

    suite_utility_results: dict[tuple[str, str], bool] = {}
    suite_security_results: dict[tuple[str, str], bool] = {}

    if user_tasks is not None:
        user_tasks_to_run = [suite.get_user_task_by_id(user_task_id) for user_task_id in user_tasks]
    else:
        user_tasks_to_run = suite.user_tasks.values()

    if injection_tasks is not None and not attack.is_data_injection:
        # For traditional prompt injection attacks, look up tasks in suite
        injection_tasks_to_run = {
            injection_task_id: suite.get_injection_task_by_id(injection_task_id)
            for injection_task_id in injection_tasks
        }
    elif not attack.is_data_injection:
        injection_tasks_to_run = suite.injection_tasks
    else:
        # For data injection attacks, skip lookup (synthetic IDs don't exist in suite)
        injection_tasks_to_run = {}

    injection_tasks_utility_results = {}
    if not attack.is_dos_attack and not attack.is_data_injection:
        for injection_task_id, injection_task in injection_tasks_to_run.items():
            # Reset security policy before each task (for Progent)
            if SECAGENT_AVAILABLE:
                reset_security_policy()
            # Call the modified run_task_without_injection_tasks with mode parameters
            successful, _ = run_task_without_injection_tasks(
                suite,
                agent_pipeline,  # Legacy positional for backward compatibility
                task=injection_task,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=benchmark_version,
                executor=actual_executor,
                mode=actual_mode,
                graph_factory=graph_factory,
                system_message=system_message,
                executor_name=executor_name,
            )
            injection_tasks_utility_results[injection_task_id] = successful

        if not all(injection_tasks_utility_results.values()):
            warnings.warn("Not all injection tasks were solved as user tasks.")

    for user_task in user_tasks_to_run:
        # Reset security policy before each task (for Progent)
        if SECAGENT_AVAILABLE:
            reset_security_policy()
        # Call the modified run_task_with_injection_tasks with mode parameters
        utility, security = run_task_with_injection_tasks(
            suite,
            agent_pipeline,  # Legacy positional for backward compatibility
            user_task=user_task,
            attack=attack,
            logdir=logdir,
            force_rerun=force_rerun,
            injection_tasks=injection_tasks,
            benchmark_version=benchmark_version,
            executor=actual_executor,
            mode=actual_mode,
            graph_factory=graph_factory,
            system_message=system_message,
            executor_name=executor_name,
        )
        suite_utility_results.update(utility)
        suite_security_results.update(security)

    return SuiteResults(
        utility_results=suite_utility_results,
        security_results=suite_security_results,
        injection_tasks_utility_results=injection_tasks_utility_results,
    )


def run_task_without_injection_tasks(
    suite: TaskSuite,
    agent_pipeline: BasePipelineElement,
    task: BaseUserTask | BaseInjectionTask,
    logdir: Path | None,
    force_rerun: bool,
    benchmark_version: str | None = None,
    *,
    # New mode-aware parameters (keyword-only)
    executor: BasePipelineElement | CompiledStateGraph | None = None,
    mode: AgentMode | None = None,
    graph_factory=None,
    system_message: str | None = None,
    executor_name: str | None = None,
) -> tuple[bool, bool]:
    """Run a task without injections. Useful to benchmark the utility of a pipeline or graph.

    Args:
        suite: The suite to use to run the benchmark.
        agent_pipeline: (LEGACY) The agent pipeline to benchmark. For backward compatibility.
        task: The task to run.
        logdir: The directory to save logs to.
        force_rerun: Whether to force rerun the task in case in the logdir there already exists
            a log for the task.
        benchmark_version: Optional benchmark version string.
        executor: (NEW) Either a BasePipelineElement or CompiledStateGraph.
        mode: (NEW) The execution mode (PIPELINE or LANGGRAPH).

    Returns:
        A tuple with the utility and security of the task. The security is always True.
    """
    # Determine which calling convention is being used
    if executor is not None and mode is not None:
        # New mode-aware calling convention (keyword args override positional)
        actual_executor = executor
        actual_mode = mode
    else:
        # Legacy pipeline-only calling convention (use positional agent_pipeline)
        actual_executor = agent_pipeline
        actual_mode = AgentMode.PIPELINE

    # Determine executor name for logging
    # Use passed executor_name, fall back to extracting from executor
    if executor_name is None:
        executor_name = getattr(actual_executor, 'name', "unknown")

    if logdir is not None and executor_name is not None:
        try:
            task_result = load_task_results(
                executor_name.replace("/", "_"),
                suite.name,
                task.ID,
                attack_name="none",
                injection_task="none",
                logdir=logdir,
            )
        except (ValidationError, FileNotFoundError):
            task_result = None
    else:
        task_result = None

    if task_result is not None and not force_rerun:
        logging.info(f"Skipping task '{task.ID}' with no injections because it was already run.")
        return task_result.utility, task_result.security

    with TraceLogger(
        delegate=Logger.get(),
        suite_name=suite.name,
        user_task_id=task.ID,
        injection_task_id=None,
        injections={},
        attack_type="none",
        pipeline_name=executor_name,
        benchmark_version=benchmark_version,
    ) as logger:
        try:
            # Use the unified executor instead of direct TaskSuite call
            utility, security = run_task_with_executor(
                suite=suite,
                executor=actual_executor,
                mode=actual_mode,
                user_task=task,
                injection_task=None,
                injections={},
                graph_factory=graph_factory,
                system_message=system_message,
            )
        except BadRequestError as e:
            if (
                e.code == "context_length_exceeded"
                or e.param == "max_tokens"
                or "reduce the length of the messages." in repr(e)
            ):
                logger.log_error(f"Skipping task {task.ID} due to context_length_exceeded: {e}")
                utility = False
                security = True
            else:
                raise e
        except UnprocessableEntityError as e:
            if "max_new_tokens" in str(e):
                logger.log_error(f"Skipping task {task.ID} due to context_length_exceeded: {e}")
                utility = False
                security = True
            else:
                raise e
        except ApiError as e:
            if "internal server error" in str(e):
                logger.log_error(f"Skipping task {task.ID} because of internal server error: {e}")
                utility = False
                security = True
            else:
                raise e
        except ServerError as e:
            logger.log_error(f"Skipping task {task.ID} because of internal server error: {e}")
            utility = False
            security = True
        except JSONDecodeError as e:
            logger.log_error(f"Skipping task {task.ID} because of JSON decode error: {e}")
            utility = False
            security = True
        logger.set_contextarg("utility", utility)
        logger.set_contextarg("security", security)

    return utility, security


def benchmark_suite_without_injections(
    agent_pipeline: BasePipelineElement,
    suite: TaskSuite,
    logdir: Path | None,
    force_rerun: bool,
    user_tasks: Sequence[str] | None = None,
    benchmark_version: str | None = None,
    *,
    # New mode-aware parameters (keyword-only)
    executor: BasePipelineElement | CompiledStateGraph | None = None,
    mode: AgentMode | None = None,
    graph_factory=None,
    system_message: str | None = None,
    executor_name: str | None = None,
) -> SuiteResults:
    """Benchmark a suite without injections.

    Args:
        agent_pipeline: (LEGACY) The agent pipeline to use. For backward compatibility.
        suite: The suite to benchmark.
        logdir: The directory to save logs to.
        force_rerun: Whether to force rerun the task in case in the logdir there already exists
            a log for the task.
        user_tasks: A subset of user tasks to run. If None, all user tasks are run.
        benchmark_version: Optional benchmark version string.
        executor: (NEW) Either a BasePipelineElement or CompiledStateGraph.
        mode: (NEW) The execution mode (PIPELINE or LANGGRAPH).
    """
    # Determine which calling convention is being used
    if executor is not None and mode is not None:
        # New mode-aware calling convention (keyword args override positional)
        actual_executor = executor
        actual_mode = mode
    else:
        # Legacy pipeline-only calling convention (use positional agent_pipeline)
        actual_executor = agent_pipeline
        actual_mode = AgentMode.PIPELINE

    suite_utility_results: dict[tuple[str, str], bool] = {}
    suite_security_results: dict[tuple[str, str], bool] = {}

    if user_tasks is not None:
        user_tasks_to_run = [suite.get_user_task_by_id(user_task_id) for user_task_id in user_tasks]
    else:
        user_tasks_to_run = suite.user_tasks.values()

    for user_task in user_tasks_to_run:
        # Reset security policy before each task (for Progent)
        if SECAGENT_AVAILABLE:
            reset_security_policy()
        # Call the modified run_task_without_injection_tasks with mode parameters
        utility, security = run_task_without_injection_tasks(
            suite,
            actual_executor,
            user_task,
            logdir,
            force_rerun,
            benchmark_version,
            executor=actual_executor,
            mode=actual_mode,
            graph_factory=graph_factory,
            system_message=system_message,
            executor_name=executor_name,
        )
        suite_utility_results[(user_task.ID, "")] = utility
        suite_security_results[(user_task.ID, "")] = security

    return SuiteResults(
        utility_results=suite_utility_results,
        security_results=suite_security_results,
        injection_tasks_utility_results={},
    )


class TaskResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    suite_name: str
    pipeline_name: str
    user_task_id: str
    injection_task_id: str | None
    attack_type: str | None
    injections: dict[str, str]
    # Use list[dict] to accept both Pipeline format (ChatMessage) and LangGraph format
    messages: list[dict]
    error: str | None
    utility: bool
    security: bool
    duration: float

    @model_validator(mode="after")
    def check_messages(self) -> Self:
        # Allow fewer messages if there was an error (e.g., recursion limit reached)
        if len(self.messages) < 2 and "claude-3-opus" not in self.pipeline_name and self.error is None:
            raise ValueError("TaskResults must have at least two messages.")
        return self


def convert_old_messages_format(messages: list[dict]) -> list[dict]:
    """Convert messages from old format (string content) to new format (content blocks).

    The new format was introduced with commit d67f9d141305e599ca2056f430f703d4021fd82a.
    """
    converted_messages = []
    for message in messages:
        message_copy = message.copy()
        if message_copy["role"] == "assistant" and message_copy["content"] is not None:
            message_copy["content"] = [{"type": "text", "content": message_copy["content"]}]
        elif message_copy["role"] in ["user", "system"]:
            message_copy["content"] = [{"type": "text", "content": message_copy["content"]}]
        elif message_copy["role"] == "tool":
            message_copy["content"] = [{"type": "text", "content": message_copy["content"]}]
        converted_messages.append(message_copy)
    return converted_messages


def load_task_results(
    pipeline_name: str,
    suite_name: str,
    user_task: str,
    attack_name: str,
    injection_task: str,
    logdir: Path,
) -> TaskResults:
    # Replace slashes with underscores to match how TraceLogger.save() stores the pipeline_name
    pipeline_name = pipeline_name.replace("/", "_")
    path = logdir / pipeline_name / suite_name / user_task / attack_name / f"{injection_task}.json"
    with path.open() as f:
        res_dict = json.load(f)

    # Check if the result is in the old format (without content blocks)
    # Old format didn't have agentdojo_package_version field
    if "agentdojo_package_version" not in res_dict:
        # Convert messages to the new format with content blocks
        res_dict["messages"] = convert_old_messages_format(res_dict["messages"])

    return TaskResults(**res_dict)


def load_suite_results(
    suite_name: str,
    pipeline_name: str,
    attack_name: str,
    logdir=Path("../runs"),
    benchmark_version: str = "v1.2.2",
) -> tuple[dict[tuple[str, str], TaskResults], dict[tuple[str, str], TaskResults]]:
    suite = get_suite(benchmark_version, suite_name)
    results_without_injections = {}
    results_with_injections = {}

    for user_task_id in suite.user_tasks.keys():
        results_without_injections[(user_task_id, None)] = load_task_results(
            pipeline_name, suite_name, user_task_id, "none", "none", logdir
        )
        if "dos" in attack_name:
            injection_task_id = next(iter(suite.injection_tasks.keys()))
            results_with_injections[(user_task_id, injection_task_id)] = load_task_results(
                pipeline_name, suite_name, user_task_id, attack_name, injection_task_id, logdir
            )
        elif attack_name != "none":
            for injection_task_id in suite.injection_tasks.keys():
                results_with_injections[(user_task_id, injection_task_id)] = load_task_results(
                    pipeline_name, suite_name, user_task_id, attack_name, injection_task_id, logdir
                )
    return results_without_injections, results_with_injections
