"""The gate registry (default_suites/v1/tools/verification_gates.py) must stay in sync with the tool
implementations. Each gate kind has a reliable marker in the tool bodies, so we find the gated tools straight
from the source and assert the registry lists exactly those -- if a gate is added or removed in the code and
not in the registry (or vice versa), a test here fails."""

import ast
import inspect

# Import the suite registry first, so importing the dailylife task_suite submodule below does not trip the
# package's import-order circular dependency.
import agentdojo.task_suite.load_suites  # noqa: F401
from agentdojo.default_suites.v1.dailylife import task_suite as dailylife_suite
from agentdojo.default_suites.v1.tools import github_client, shopping_client
from agentdojo.default_suites.v1.tools.verification_gates import (
    CONFLICT_GATES,
    PRECONDITION_GATES,
    VERIFICATION_GATES,
    gated_tools,
)

# The modules that define gated tools.
_GATE_MODULES = (shopping_client, github_client, dailylife_suite)


def _top_level_sources(module) -> dict[str, str]:
    """{name: source} for every MODULE-LEVEL function (nested helpers excluded, so a marker inside a nested
    function is attributed to the top-level tool that owns it)."""
    src = inspect.getsource(module)
    tree = ast.parse(src)
    return {n.name: ast.get_source_segment(src, n) or "" for n in tree.body if isinstance(n, ast.FunctionDef)}


def _funcs_calling(module, callee: str) -> set[str]:
    """Top-level function names whose body calls `callee` (by name). Excludes `callee`'s own definition."""
    tree = ast.parse(inspect.getsource(module))
    out = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == callee:
                out.add(node.name)
    return out


def _funcs_containing(module, needle: str) -> set[str]:
    """Top-level function names whose source contains `needle`."""
    return {name for name, src in _top_level_sources(module).items() if needle in src}


def test_verification_gates_match_the_code():
    # Every OTP gate calls send_otp_to_inbox; the registry must list exactly those tools.
    actual = set().union(*(_funcs_calling(m, "send_otp_to_inbox") for m in _GATE_MODULES))
    assert actual == gated_tools(), (
        f"VERIFICATION_GATES out of sync; only in code: {sorted(actual - gated_tools())}, "
        f"only in registry: {sorted(gated_tools() - actual)}"
    )


def test_verification_gates_name_existing_verify_and_inbox_tools():
    names = set().union(*(set(dir(m)) for m in _GATE_MODULES))
    for gate in VERIFICATION_GATES:
        assert gate.verify_tool in names, f"{gate.tool}: verify_tool {gate.verify_tool!r} not found"
        assert gate.inbox_tool in names, f"{gate.tool}: inbox_tool {gate.inbox_tool!r} not found"


def test_conflict_gates_match_the_code():
    # git_push / git_pull are the only tools that raise "Conflict detected".
    actual = set().union(*(_funcs_containing(m, "Conflict detected") for m in _GATE_MODULES))
    declared = {g.tool for g in CONFLICT_GATES}
    assert actual == declared, (
        f"CONFLICT_GATES out of sync; only in code: {sorted(actual - declared)}, "
        f"only in registry: {sorted(declared - actual)}"
    )


def test_precondition_gate_messages_are_in_the_code():
    # Forward check: each declared precondition's refusal message really appears in its tool's source.
    sources = {}
    for m in _GATE_MODULES:
        sources.update(_top_level_sources(m))
    for gate in PRECONDITION_GATES:
        assert gate.tool in sources, f"{gate.tool} not found among the tools"
        assert gate.message in sources[gate.tool], (
            f"{gate.tool}: precondition message {gate.message!r} not found in its source"
        )
