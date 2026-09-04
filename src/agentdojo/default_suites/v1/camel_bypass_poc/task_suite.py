from functools import partial

from deepdiff.diff import DeepDiff

from agentdojo.default_suites.v1.camel_bypass_poc.tools import (
    Inbox,
    get_received_emails,
    send_email,
)
from agentdojo.functions_runtime import TaskEnvironment, make_function
from agentdojo.task_suite.task_suite import TaskSuite
from langchain_core.tools import tool


class CamelBypassEnvironment(TaskEnvironment):
    inbox: Inbox


TOOLS = [
    get_received_emails,
    send_email,
]

deepdiff_paths_to_exclude = [
    "root.inbox.sent",
    "root.inbox.received",
    "root.responses",
    "root.model_fields_set",
    "root.inbox.initial_emails",
]

CamelBypassDeepDiff = partial(DeepDiff, exclude_paths=deepdiff_paths_to_exclude)

task_suite = TaskSuite[CamelBypassEnvironment](
    "camel_bypass_poc",
    CamelBypassEnvironment,
    tools=[make_function(func) for func in TOOLS],
    langchain_tools=[tool(func) for func in TOOLS],
)
