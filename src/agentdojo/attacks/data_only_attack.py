from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.attacks.attack_registry import register_attack
from agentdojo.attacks.base_attacks import BaseAttack
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.task_suite.task_suite import TaskSuite


class BaseDataOnlyAttack(BaseAttack):
    """Base class for data injection attacks.

    Data injection attacks test if injected data leaks into agent output.
    They inject data into the environment and check if that data appears
    in the agent's tool call arguments or final output. They differ from
    prompt injection attacks in that there is no explicit malicious
    instruction - only data that the agent may incorrectly propagate.

    Subclasses must define:
        name: str - The attack name for CLI usage
        subcategory: str - The subcategory suffix (e.g., "SYNTACTIC" or "SEMANTIC")

    User tasks must define per-subcategory fields:
        INJECTED_DATA_{subcategory}: list[dict[str, str]] - List of injection vectors to data mappings
        SECURITY_CHECKS_{subcategory}: list[Callable] - List of security check functions for each injection
    """

    is_data_injection = True
    subcategory: str  # Subclasses must set this (e.g., "SYNTACTIC" or "SEMANTIC")

    def __init__(
        self,
        task_suite: TaskSuite,
        target_pipeline: BasePipelineElement | None = None,
        *,
        model_name: str | None = None,
    ):
        super().__init__(task_suite, target_pipeline, model_name=model_name)

    def attack(self, user_task: BaseUserTask, injection_task: BaseInjectionTask) -> dict[str, str]:
        """For data-only attacks, this method is not used in the traditional sense.

        The benchmark loop handles INJECTED_DATA directly from the user task.
        This method returns an empty dict as a placeholder.
        """
        return {}


@register_attack
class DataOnlySyntacticAttack(BaseDataOnlyAttack):
    """Syntactic data injection attack - uses structural syntax to override fields via parsing confusion.

    User tasks must define:
        INJECTED_DATA_SYNTACTIC: list[dict[str, str]] - List of injection vectors to data mappings
        SECURITY_CHECKS_SYNTACTIC: list[Callable] - List of security check functions for each injection
    """

    name = "data_only_syntactic"
    subcategory = "SYNTACTIC"


@register_attack
class DataOnlySemanticAttack(BaseDataOnlyAttack):
    """Semantic data injection attack - uses authoritative text to override fields via interpretation confusion.

    User tasks must define:
        INJECTED_DATA_SEMANTIC: list[dict[str, str]] - List of injection vectors to data mappings
        SECURITY_CHECKS_SEMANTIC: list[Callable] - List of security check functions for each injection
    """

    name = "data_only_semantic"
    subcategory = "SEMANTIC"
