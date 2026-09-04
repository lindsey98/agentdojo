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

# Note: To avoid circular imports, modules in this package should be imported
# directly from their files rather than from this __init__.py. For example:
#
#   from agents.camel.utils.security_policy import SecurityPolicyEngine
#   from agents.camel.utils.quarantined_llm import query_quarantined_llm
#   from agents.camel.utils.conditional_cache import conditional_lru_cache
#
# This is because security_policy imports from interpreter, which imports
# from utils, creating a circular dependency if we eagerly import here.
