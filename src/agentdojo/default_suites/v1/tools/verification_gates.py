"""Declarative registry of the OTP verification gates enforced by the dynamic-suite tools.

Several state-changing tools refuse to execute until the account is verified: the unverified call emails a
one-time password (OTP), stashes the pending action, and returns "please verify"; a ``verify_*`` call with
that code then commits it. Today these gates live only inside the tool bodies -- each calls
``send_otp_to_inbox`` and stashes into ``verification_stack[otp]`` -- so a consumer (a defense, an analysis,
a benchmark harness) has to reverse-engineer the implementations to learn which tools are gated.

This module declares them as data, so they can be enumerated directly. ``tests/test_verification_gates.py``
asserts the registry stays in sync with the code, so it cannot silently drift.

Extracted from ``shopping_client.py`` / ``github_client.py`` and ``dailylife/task_suite.py``. ``send_money``
appears twice: it is defined in both the shopping and dailylife suites, which email the OTP to different
inboxes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationGate:
    """One OTP-gated tool.

    :param tool: the gated tool's registered name.
    :param verify_tool: the tool that submits the OTP and commits the pending action.
    :param inbox_tool: the tool that reads the inbox the OTP was emailed to.
    :param condition: when the gate fires -- ``"always"`` or a human-readable predicate (``"amount > 500"``).
    """

    tool: str
    verify_tool: str
    inbox_tool: str
    condition: str = "always"


VERIFICATION_GATES: list[VerificationGate] = [
    # shopping
    VerificationGate("checkout_selected_cart", "verify_shopping_account", "get_shopping_linked_unread_emails"),
    VerificationGate("refund_order", "verify_shopping_account", "get_shopping_linked_unread_emails"),
    VerificationGate(
        "update_shopping_account_password", "verify_shopping_account", "get_shopping_linked_unread_emails"
    ),
    VerificationGate("send_money", "verify_transaction", "get_shopping_linked_unread_emails", "amount > 500"),
    # github
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


def gated_tools() -> set[str]:
    """The set of tool names that are OTP-gated (deduplicated across suites)."""
    return {g.tool for g in VERIFICATION_GATES}
