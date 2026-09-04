# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from agents.camel.utils import security_policy
from agents.camel.capabilities import can_readers_read_value, get_all_readers, is_trusted
from agents.camel.interpreter.value import CaMeLStr, CaMeLValue
from agents.camel.security_policies.agentdojo_security_policies import (
    AgentDojoSecurityPolicyEngine,
)
from agents.camel.utils.security_policy import Allowed, Denied, SecurityPolicyResult

if TYPE_CHECKING:
    from agentdojo.default_suites.v1.camel_bypass_poc.task_suite import CamelBypassEnvironment

camel_bypass_no_side_effect_tools = {
    "get_received_emails",
}


class CamelBypassSecurityPolicyEngine(AgentDojoSecurityPolicyEngine["CamelBypassEnvironment"]):
    def __init__(self, env: CamelBypassEnvironment) -> None:
        super().__init__(env)
        self.policies = [
            ("send_email", self.send_email_policy),
        ]
        self.no_side_effect_tools = security_policy.NO_SIDE_EFFECT_TOOLS | camel_bypass_no_side_effect_tools

    def send_email_policy(self, tool_name: str, kwargs: Mapping[str, CaMeLValue]) -> SecurityPolicyResult:
        recipient: CaMeLStr = kwargs["recipient"]  # type: ignore

        # Email address comes directly from the user
        if is_trusted(recipient):
            return Allowed()
        recipient_set = {recipient.raw}

        # Check body
        body: CaMeLStr = kwargs["body"]  # type: ignore
        body_readers, _ = get_all_readers(body)
        if not can_readers_read_value(recipient_set, body):
            return Denied(
                f"Body is not public or shared with the recipient: recipient={recipient_set}, readers={body_readers}."
            )

        # Check subject
        subject: CaMeLStr = kwargs["subject"]  # type: ignore
        subject_readers, _ = get_all_readers(subject)
        if not can_readers_read_value(recipient_set, subject):
            return Denied(
                f"Subject is not public or shared with the recipient: recipient={recipient_set}, readers={subject_readers}."
            )

        return Allowed()
