"""The OTP verification-gate registry (default_suites/v1/tools/verification_gates.py) must stay in sync with
the tool implementations. Every gate calls ``send_otp_to_inbox`` in its body, so we find the gated tools
straight from the source and assert the registry lists exactly those -- if a gate is added or removed in the
code and not here (or vice versa), this test fails."""

import ast
import inspect

# Import the suite registry first, so importing the dailylife task_suite submodule below does not trip the
# package's import-order circular dependency.
import agentdojo.task_suite.load_suites  # noqa: F401
from agentdojo.default_suites.v1.dailylife import task_suite as dailylife_suite
from agentdojo.default_suites.v1.tools import github_client, shopping_client
from agentdojo.default_suites.v1.tools.verification_gates import VERIFICATION_GATES, gated_tools

# The modules that define OTP-gated tools.
_GATE_MODULES = (shopping_client, github_client, dailylife_suite)


def _functions_calling_send_otp(module) -> set[str]:
    """Names of the functions in `module` whose body calls ``send_otp_to_inbox`` -- i.e. the OTP gates. The
    ``send_otp_to_inbox`` helper itself calls ``send_receive_email``, not itself, so it is not matched."""
    tree = ast.parse(inspect.getsource(module))
    gated = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "send_otp_to_inbox":
                gated.add(node.name)
    return gated


def test_registry_matches_the_gated_tool_implementations():
    actual = set().union(*(_functions_calling_send_otp(m) for m in _GATE_MODULES))
    assert actual == gated_tools(), (
        "verification_gates registry is out of sync with the tool bodies; "
        f"only in code: {sorted(actual - gated_tools())}, only in registry: {sorted(gated_tools() - actual)}"
    )


def test_every_gate_names_an_existing_verify_and_inbox_tool():
    names = set().union(*(set(dir(m)) for m in _GATE_MODULES))
    for gate in VERIFICATION_GATES:
        assert gate.verify_tool in names, f"{gate.tool}: verify_tool {gate.verify_tool!r} not found"
        assert gate.inbox_tool in names, f"{gate.tool}: inbox_tool {gate.inbox_tool!r} not found"
