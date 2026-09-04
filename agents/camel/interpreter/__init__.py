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

"""CaMeL interpreter module.

This module provides the core interpreter functionality for CaMeL.
"""

from agents.camel.interpreter.result import Error, Ok, Result
from agents.camel.interpreter.namespace import Namespace
from agents.camel.interpreter.value import (
    CaMeLValue,
    CaMeLCallable,
    CaMeLNone,
    CaMeLTrue,
    CaMeLFalse,
    CaMeLInt,
    CaMeLFloat,
    CaMeLStr,
    CaMeLList,
    CaMeLDict,
    CaMeLTuple,
    CaMeLSet,
    CaMeLClass,
    CaMeLClassInstance,
    CaMeLFunction,
    CaMeLBuiltin,
    value_from_raw,
    make_camel_builtin,
)
from agents.camel.interpreter.interpreter import (
    camel_eval,
    parse_and_interpret_code,
    extract_code_block,
    EvalResult,
    EvalArgs,
    MetadataEvalMode,
    FunctionCall,
    CaMeLException,
    InvalidOutputError,
)

__all__ = [
    # Result types
    "Ok",
    "Error",
    "Result",
    # Namespace
    "Namespace",
    # Value types
    "CaMeLValue",
    "CaMeLCallable",
    "CaMeLNone",
    "CaMeLTrue",
    "CaMeLFalse",
    "CaMeLInt",
    "CaMeLFloat",
    "CaMeLStr",
    "CaMeLList",
    "CaMeLDict",
    "CaMeLTuple",
    "CaMeLSet",
    "CaMeLClass",
    "CaMeLClassInstance",
    "CaMeLFunction",
    "CaMeLBuiltin",
    "value_from_raw",
    "make_camel_builtin",
    # Interpreter
    "camel_eval",
    "parse_and_interpret_code",
    "extract_code_block",
    "EvalResult",
    "EvalArgs",
    "MetadataEvalMode",
    "FunctionCall",
    "CaMeLException",
    "InvalidOutputError",
]
