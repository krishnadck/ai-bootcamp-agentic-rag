import inspect
import json
import re
from typing import Any, Callable, Dict, List, get_type_hints
from pydantic import BaseModel, TypeAdapter, create_model, Field
from pydantic.fields import FieldInfo
from langchain_core.messages import AIMessage, HumanMessage

# #### FORMAT AI MESSAGE (Cleaner approach) ####
def format_ai_message(response, *, name: str = "") -> AIMessage:
    """
    Standardizes the response. 
    ``name`` is attached to the AIMessage so downstream agents
    (e.g. the coordinator) can see which agent produced it.
    """
    tool_calls = []

    def _get_field(obj, field, default=None):
        if isinstance(obj, dict):
            return obj.get(field, default)
        return getattr(obj, field, default)

    # Check if tool_calls exists and is not None/Empty
    if getattr(response, "tool_calls", None):
        for i, tc in enumerate(response.tool_calls):
            tc_name = _get_field(tc, "name")
            tc_id = _get_field(tc, "id", f"call_{i}")
            tc_args = _get_field(tc, "args")
            if tc_args is None:
                tc_args = _get_field(tc, "arguments")
            if isinstance(tc_args, str):
                try:
                    tc_args = json.loads(tc_args)
                except json.JSONDecodeError:
                    tc_args = {"raw": tc_args}

            tool_calls.append(
                {
                    "id": tc_id,
                    "name": tc_name,
                    "args": tc_args or {},
                    "type": "tool_call",
                }
            )

    content = ""
    for field in ("content", "answer"):
        val = getattr(response, field, None)
        if val:
            content = val
            break

    return AIMessage(
        content=content,
        tool_calls=tool_calls,
        name=name or "",
    )


def _extract_last_user_query(messages) -> str:
    """Walk backwards through messages to find the last human query."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
        if isinstance(msg, dict) and msg.get("type") == "human":
            return msg.get("content", "")
        if hasattr(msg, "type") and getattr(msg, "type", None) == "human":
            return getattr(msg, "content", "")
    return ""


def _extract_product_ids_from_messages(messages) -> list:
    """Scan tool output messages for Product IDs (parent_asin pattern)."""
    product_ids = []
    for msg in messages:
        content = ""
        if hasattr(msg, "content"):
            content = msg.content or ""
        elif isinstance(msg, dict):
            content = msg.get("content", "")
        for match in re.findall(r"Product ID:\s*(B[A-Z0-9]+)", content):
            if match not in product_ids:
                product_ids.append(match)
    return product_ids


def patch_empty_tool_arguments(response, messages, *, user_id: str = "", cart_id: str = ""):
    """Fix tool calls where the LLM omitted the arguments (common with
    instructor TOOLS mode + generic dict fields at temperature=0).
    Mutates ``response.tool_calls`` in-place."""
    user_query = _extract_last_user_query(messages)

    for tc in response.tool_calls:
        args = getattr(tc, "arguments", None) or {}
        if args:
            continue

        if tc.name == "retrieve_products":
            tc.arguments = {"query": user_query}

        elif tc.name == "get_formatted_reviews_context":
            product_ids = _extract_product_ids_from_messages(messages)
            tc.arguments = {
                "query": user_query,
                "item_list": product_ids,
                "top_k": 20,
            }

        elif tc.name in ("add_to_shopping_cart", "read_shopping_cart", "remove_item_from_cart"):
            tc.arguments = {"user_id": user_id, "cart_id": cart_id}

        elif tc.name == "check_warehouse_availability":
            tc.arguments = {"items": []}

        elif tc.name == "reserve_warehouse_items":
            tc.arguments = {"reservations": []}


import inspect
from typing import Any, Callable, Dict, List
from pydantic import create_model, Field
from docstring_parser import parse

# --- CORE LOGIC (The New Engine) ---
def _generate_openai_schema(func: Callable) -> Dict[str, Any]:
    """Internal robust generator using Pydantic + Inspect."""
    doc = parse(inspect.getdoc(func) or "")
    param_docs = {p.arg_name: p.description for p in doc.params}
    return_description = doc.returns.description if doc.returns else ""
    
    fields = {}
    for name, param in inspect.signature(func).parameters.items():
        if name in ('self', 'cls'): continue
        fields[name] = (
            param.annotation if param.annotation != inspect.Parameter.empty else Any,
            Field(
                default=param.default if param.default != inspect.Parameter.empty else ...,
                description=param_docs.get(name, "")
            )
        )
        
    model = create_model(func.__name__, **fields)
    schema = model.model_json_schema()

    return_annotation = inspect.signature(func).return_annotation
    returns_schema = None
    if return_annotation != inspect.Signature.empty:
        try:
            return_schema = TypeAdapter(return_annotation).json_schema()
        except Exception:
            return_schema = {"type": "object"}
        return_schema.pop("title", None)
        returns_schema = {
            "description": return_description,
            "schema": return_schema,
        }
    
    return {
        "name": func.__name__,
        "description": doc.short_description or "",
        "parameters": {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", [])
        },
        "returns": returns_schema,
    }

# --- YOUR ORIGINAL INTERFACE (The Wrapper) ---

def parse_function_definition(function_def: Callable) -> Dict[str, Any]:
    """
    Legacy Wrapper: Accepts a FUNCTION OBJECT instead of a string.
    NOTE: You must pass the actual function now, not 'inspect.getsource(func)'.
    """
    # We ignore the 'function_def' string input if you were passing strings before.
    # This adapter assumes you update the call site to pass the function object.
    return _generate_openai_schema(function_def)

def get_tool_descriptions(function_list: List[Callable]) -> List[Dict[str, Any]]:
    """Your original function name, now using the robust engine."""
    return [parse_function_definition(func) for func in function_list]

def normalize_tool_calls(ai_msg: AIMessage) -> AIMessage:
    """
    If tool_calls exist, strip namespace prefixes like 'functions.'
    from tool_call names. Otherwise return original message.
    """

    tool_calls = getattr(ai_msg, "tool_calls", None)
    if not tool_calls:
        return ai_msg  # no tool calls, nothing to normalize

    changed = False
    fixed_calls = []

    for tc in tool_calls:
        name = tc.get("name", "")
        normalized_name = name.split(".")[-1].strip()

        if normalized_name != name:
            changed = True

        fixed_calls.append({**tc, "name": normalized_name})

    if not changed:
        return ai_msg  # no changes needed

    # Return new AIMessage with corrected tool_calls
    return AIMessage(
        content=ai_msg.content or "",
        tool_calls=fixed_calls,
        additional_kwargs=ai_msg.additional_kwargs,
        response_metadata=ai_msg.response_metadata,
    )