"""Declarative registry of the ENV-ENFORCED gates on the dynamic-suite tools.

A gate is a check a tool applies before it will do its work; when unmet, the tool refuses and tells the caller
what to do first. Today these gates live only inside the tool bodies, so a consumer (a defense, an analysis, a
benchmark harness) has to reverse-engineer them. This module declares them as data. The tests in
``tests/test_verification_gates.py`` assert each kind stays in sync with the code, so it cannot silently drift.

Three kinds:
  - ``VERIFICATION_GATES``  -- an OTP the tool emails and a ``verify_*`` call must submit before the action
    commits (the unverified call only emails the code and stashes the pending action).
  - ``PRECONDITION_GATES``  -- a prior action the tool requires: a repo must be cloned (which writes a
    ``.git_config``) before ``git_push`` / ``git_pull`` operate on it.
  - ``CONFLICT_GATES``      -- ``git_push`` / ``git_pull`` refuse on a content conflict until you sync the
    other way or force it.

Extracted from ``shopping_client.py`` / ``github_client.py`` and ``dailylife/task_suite.py``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationGate:
    """An OTP-gated tool.

    :param tool: the gated tool's registered name.
    :param verify_tool: the tool that submits the OTP and commits the pending action.
    :param inbox_tool: the tool that reads the inbox the OTP was emailed to.
    :param condition: when the gate fires -- ``"always"`` or a human-readable predicate (``"amount > 500"``).
    """

    tool: str
    verify_tool: str
    inbox_tool: str
    condition: str = "always"


@dataclass(frozen=True)
class PreconditionGate:
    """A tool that refuses until a prior action is taken.

    :param tool: the gated tool's registered name.
    :param requires: the action that satisfies the precondition (human-readable).
    :param message: a substring of the refusal the tool returns when the precondition is unmet.
    """

    tool: str
    requires: str
    message: str


@dataclass(frozen=True)
class ConflictGate:
    """A tool that refuses on a content conflict until it is resolved.

    :param tool: the gated tool's registered name.
    :param resolution: how to resolve the conflict (human-readable).
    :param message: a substring of the refusal the tool returns on a conflict.
    """

    tool: str
    resolution: str
    message: str


VERIFICATION_GATES: list[VerificationGate] = [
    # shopping
    VerificationGate("checkout_selected_cart", "verify_shopping_account", "get_shopping_linked_unread_emails"),
    VerificationGate("refund_order", "verify_shopping_account", "get_shopping_linked_unread_emails"),
    VerificationGate(
        "update_shopping_account_password", "verify_shopping_account", "get_shopping_linked_unread_emails"
    ),
    # send_money is gated only for amount > 500; the linked inbox differs by suite (shopping vs dailylife).
    VerificationGate("send_money", "verify_transaction", "get_shopping_linked_unread_emails", "amount > 500"),
    # github (all gated unconditionally)
    VerificationGate("git_create_repo", "verify_github_account", "get_github_linked_unread_emails"),
    VerificationGate("git_delete_repo", "verify_github_account", "get_github_linked_unread_emails"),
    VerificationGate("git_transfer_repo_ownership", "verify_github_account", "get_github_linked_unread_emails"),
    VerificationGate("git_invite_collaborators", "verify_github_account", "get_github_linked_unread_emails"),
    VerificationGate("git_add_ssh_key", "verify_github_account", "get_github_linked_unread_emails"),
    VerificationGate("git_delete_ssh_key", "verify_github_account", "get_github_linked_unread_emails"),
    VerificationGate("git_get_linked_ssh_keys", "verify_github_account", "get_github_linked_unread_emails"),
    VerificationGate("update_github_account_password", "verify_github_account", "get_github_linked_unread_emails"),
    # dailylife (same send_money tool name, a different linked inbox)
    VerificationGate("send_money", "verify_transaction", "get_unread_emails", "amount > 500"),
]

PRECONDITION_GATES: list[PreconditionGate] = [
    # git_clone writes a .git_config into the local repo; push/pull upward-search for it and refuse without it.
    PreconditionGate("git_push", "git_clone the repository first", "not inside a cloned GitHub repository"),
    PreconditionGate("git_pull", "git_clone the repository first", "Please navigate into a cloned repository"),
]

CONFLICT_GATES: list[ConflictGate] = [
    ConflictGate("git_push", "git_pull first, or call git_push with force=True", "Conflict detected"),
    ConflictGate("git_pull", "git_push first, or call git_pull with overwrite_local=True", "Conflict detected"),
]


def gated_tools() -> set[str]:
    """The set of tool names carrying an OTP verification gate (deduplicated across suites)."""
    return {g.tool for g in VERIFICATION_GATES}
