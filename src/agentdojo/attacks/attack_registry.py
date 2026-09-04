from typing import TypeVar

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.attacks.base_attacks import BaseAttack
from agentdojo.task_suite.task_suite import TaskSuite

ATTACKS: dict[str, type[BaseAttack]] = {}


A = TypeVar("A", bound=BaseAttack)


def register_attack(attack: type[A]) -> type[A]:
    """Adds an attack to the registry.

    It can be used both as a function call or as a decorator.

    For example:
    ```python
    @register_attack
    class MyAttack(BaseAttack): ...
    ```

    Or:
    ```python
    class MyAttack(BaseAttack): ...


    register_attack(MyAttack)
    ```

    Args:
        attack: The attack class to register.

    """
    ATTACKS[attack.name] = attack
    return attack


def load_attack(
    attack_name: str,
    task_suite: TaskSuite,
    target_pipeline: BasePipelineElement | None = None,
    *,
    model_name: str | None = None,
) -> BaseAttack:
    """Gets and instantiates an attack from the registry.

    To use your own attack, you must first register it using the [`register_attack`][agentdojo.attacks.attack_registry.register_attack] function.

    Args:
        attack_name: The name of the attack to load.
        task_suite: The task suite to use for the attack.
        target_pipeline: The target pipeline to attack (for pipeline agents).
        model_name: The model name string (for LangGraph agents).

    Note:
        Either target_pipeline or model_name should be provided.
        For pipeline agents, use target_pipeline.
        For LangGraph agents, use model_name.
    """
    return ATTACKS[attack_name](task_suite, target_pipeline, model_name=model_name)
