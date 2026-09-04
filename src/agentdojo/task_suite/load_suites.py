from collections import defaultdict

from agentdojo.default_suites.v1.banking import banking_task_suite
from agentdojo.default_suites.v1.slack import slack_task_suite
from agentdojo.default_suites.v1.travel import travel_task_suite
from agentdojo.default_suites.v1.workspace import workspace_task_suite
from agentdojo.task_suite.task_suite import TaskSuite


def _get_camel_bypass_poc_suite():
    from agentdojo.default_suites.v1.camel_bypass_poc.task_suite import task_suite
    return task_suite


_V1_SUITES: dict[str, TaskSuite] = {
    "workspace": workspace_task_suite,
    "travel": travel_task_suite,
    "banking": banking_task_suite,
    "slack": slack_task_suite,
}

_V1_1_SUITES: dict[str, TaskSuite] = {
    "workspace": workspace_task_suite,
    "travel": travel_task_suite.get_new_version((1, 1, 0)),
    "banking": banking_task_suite,
    "slack": slack_task_suite,
}

_V1_1_1_SUITES: dict[str, TaskSuite] = {
    "workspace": workspace_task_suite.get_new_version((1, 1, 1)),
    "travel": travel_task_suite.get_new_version((1, 1, 1)),
    "banking": banking_task_suite.get_new_version((1, 1, 1)),
    "slack": slack_task_suite.get_new_version((1, 1, 1)),
}

_V1_1_2_SUITES: dict[str, TaskSuite] = {
    "workspace": workspace_task_suite.get_new_version((1, 1, 2)),
    "travel": travel_task_suite.get_new_version((1, 1, 2)),
    "banking": banking_task_suite.get_new_version((1, 1, 1)),
    "slack": slack_task_suite.get_new_version((1, 1, 1)),
}

_V1_2_SUITES: dict[str, TaskSuite] = {
    "workspace": workspace_task_suite.get_new_version((1, 2, 0)),
    "travel": travel_task_suite.get_new_version((1, 2, 0)),
    "banking": banking_task_suite.get_new_version((1, 2, 0)),
    "slack": slack_task_suite.get_new_version((1, 2, 0)),
}

_V1_2_1_SUITES: dict[str, TaskSuite] = {
    "workspace": workspace_task_suite.get_new_version((1, 2, 1)),
    "travel": travel_task_suite.get_new_version((1, 2, 0)),
    "banking": banking_task_suite.get_new_version((1, 2, 0)),
    "slack": slack_task_suite.get_new_version((1, 2, 0)),
}

_V1_2_2_SUITES: dict[str, TaskSuite] = {
    "workspace": workspace_task_suite.get_new_version((1, 2, 2)),
    "travel": travel_task_suite.get_new_version((1, 2, 0)),
    "banking": banking_task_suite.get_new_version((1, 2, 2)),
    "slack": slack_task_suite.get_new_version((1, 2, 0)),
}

_SUITES: defaultdict[str, dict[str, TaskSuite]] = defaultdict(
    dict,
    {
        "v1": _V1_SUITES,
        "v1.1": _V1_1_SUITES,
        "v1.1.1": _V1_1_1_SUITES,
        "v1.1.2": _V1_1_2_SUITES,
        "v1.2": _V1_2_SUITES,
        "v1.2.1": _V1_2_1_SUITES,
        "v1.2.2": _V1_2_2_SUITES,
    },
)


def register_suite(suite: TaskSuite, version: str) -> None:
    _SUITES[version][suite.name] = suite


def get_suites(benchmark_version: str) -> dict[str, TaskSuite]:
    suites = _SUITES[benchmark_version].copy()
    # Lazy load camel_bypass_poc to avoid circular import
    if benchmark_version in ("v1", "v1.2.2"):
        suites["camel_bypass_poc"] = _get_camel_bypass_poc_suite()
    return suites


def get_suite(benchmark_version: str, suite_name: str) -> TaskSuite:
    # Lazy load camel_bypass_poc to avoid circular import
    if suite_name == "camel_bypass_poc" and benchmark_version in ("v1", "v1.2.2"):
        return _get_camel_bypass_poc_suite()
    return _SUITES[benchmark_version][suite_name]
