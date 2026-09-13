#!/usr/bin/env python3
"""Extract AgentDojo tool response schemas to CSV."""

import csv
import json
import sys
import types
from typing import Any, Union, get_origin, get_args

from pydantic import BaseModel

from agentdojo.task_suite.load_suites import get_suites


def is_union_type(tp: Any) -> bool:
    """Check if type is a Union (typing.Union or X | Y syntax)."""
    origin = get_origin(tp)
    if origin is Union:
        return True
    if isinstance(tp, types.UnionType):
        return True
    return False


def simplify_schema(schema: dict, defs: dict | None = None) -> dict | str | list:
    """
    Convert JSON Schema to a simple {field: type} representation.
    """
    if not isinstance(schema, dict):
        return schema

    # Merge defs from current schema with passed defs
    current_defs = schema.get("$defs", {})
    if defs is None:
        defs = current_defs
    else:
        defs = {**defs, **current_defs}

    # Resolve $ref first
    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        if ref_name in defs:
            return simplify_schema(defs[ref_name], defs)

    # Handle anyOf/oneOf (union types)
    if "anyOf" in schema:
        types = [simplify_schema(s, defs) for s in schema["anyOf"]]
        # Filter out null and simplify
        non_null = [t for t in types if t != "null"]
        if len(non_null) == 1:
            return non_null[0]
        # Simplify union representation
        type_names = []
        for t in types:
            if isinstance(t, str):
                type_names.append(t)
            elif isinstance(t, dict):
                type_names.append("object")
            elif isinstance(t, list):
                type_names.append("array")
            else:
                type_names.append(str(t))
        return " | ".join(type_names)

    # Get the type
    schema_type = schema.get("type")

    # Handle type arrays like ["string", "null"]
    if isinstance(schema_type, list):
        non_null = [t for t in schema_type if t != "null"]
        if len(non_null) == 1:
            schema_type = non_null[0]
        else:
            return " | ".join(schema_type)

    # Handle enums
    if "enum" in schema:
        return "string"  # enums are typically strings

    # Handle primitives
    if schema_type in ("string", "integer", "number", "boolean", "null"):
        return schema_type

    # Handle arrays
    if schema_type == "array":
        items = schema.get("items", {})
        item_type = simplify_schema(items, defs)
        return [item_type]

    # Handle objects with properties (structured objects)
    if "properties" in schema:
        result = {}
        for prop_name, prop_schema in schema["properties"].items():
            result[prop_name] = simplify_schema(prop_schema, defs)
        return result

    # Handle dict types (additionalProperties)
    if schema_type == "object" and "additionalProperties" in schema:
        value_type = simplify_schema(schema["additionalProperties"], defs)
        return {"<key>": value_type}

    # Handle plain object
    if schema_type == "object":
        return "object"

    return "unknown"


def return_type_to_json_schema(return_type: Any) -> dict:
    """
    Convert a Python return type annotation to JSON Schema.

    Handles:
    - Pydantic BaseModel subclasses -> model_json_schema()
    - list[T] -> {"type": "array", "items": ...}
    - dict[K, V] -> {"type": "object", "additionalProperties": ...}
    - Primitive types (str, int, float, bool)
    - None/NoneType -> {"type": "null"}
    - Union types (T | None) -> {"anyOf": [...]}
    """
    if return_type is None:
        return {"type": "null"}

    if return_type is type(None):
        return {"type": "null"}

    # Check if it's a Pydantic BaseModel subclass
    try:
        if isinstance(return_type, type) and issubclass(return_type, BaseModel):
            return return_type.model_json_schema()
    except TypeError:
        pass

    origin = get_origin(return_type)
    args = get_args(return_type)

    # Handle Union types (including T | None via types.UnionType)
    if is_union_type(return_type):
        if isinstance(return_type, types.UnionType):
            args = return_type.__args__
        schemas = [return_type_to_json_schema(arg) for arg in args]
        non_null = [s for s in schemas if s.get("type") != "null"]
        has_null = any(s.get("type") == "null" for s in schemas)
        if len(non_null) == 1 and has_null:
            result = non_null[0].copy()
            if "type" in result and isinstance(result["type"], str):
                result["type"] = [result["type"], "null"]
            return result
        return {"anyOf": schemas}

    # Handle list[T]
    if origin is list:
        if args:
            item_schema = return_type_to_json_schema(args[0])
            return {"type": "array", "items": item_schema}
        else:
            return {"type": "array"}

    # Handle dict[K, V]
    if origin is dict:
        if len(args) >= 2:
            value_schema = return_type_to_json_schema(args[1])
            return {"type": "object", "additionalProperties": value_schema}
        else:
            return {"type": "object"}

    # Handle primitive types
    primitive_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }

    if return_type in primitive_map:
        return primitive_map[return_type]

    return {"type": "unknown", "python_type": str(return_type)}


def get_all_suite_tools() -> list[dict]:
    """Get all tools from all suites with their metadata."""
    suites = get_suites("v1")

    all_tools = []
    for suite_name, suite in suites.items():
        for func in suite.tools:
            all_tools.append({
                "suite_name": suite_name,
                "tool_name": func.name,
                "return_type": func.return_type,
            })

    return all_tools


def export_to_csv(output_path: str) -> None:
    """Export all tool response schemas to CSV."""
    tools = get_all_suite_tools()

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["suite_name", "tool_name", "response_schema"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for tool in tools:
            schema = return_type_to_json_schema(tool["return_type"])
            simplified = simplify_schema(schema)
            writer.writerow({
                "suite_name": tool["suite_name"],
                "tool_name": tool["tool_name"],
                "response_schema": json.dumps(simplified, ensure_ascii=False, indent=2),
            })

    print(f"Exported {len(tools)} tools to {output_path}")


if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else "tool_response_schemas.csv"
    export_to_csv(output_path)
