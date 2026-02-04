import inspect
import json
from typing import Any, Callable, Dict, List
from pydantic import TypeAdapter, create_model, Field
from langchain_core.messages import AIMessage
from docstring_parser import parse

# #### FORMAT AI MESSAGE (Cleaner approach) ####
def format_ai_message(response) -> AIMessage:
    """
    Standardizes the response. 
    Note: Modern frameworks usually handle object conversion, 
    but this keeps your logic explicit.
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

    return AIMessage(
        content=response.content if hasattr(response, "content") else "",
        tool_calls=tool_calls,
    )

# --- CORE LOGIC (The New Engine) ---
def _generate_openai_schema(func: Callable) -> Dict[str, Any]:
    """Internal robust generator using Pydantic + Inspect."""
    doc = parse(inspect.getdoc(func) or "")
    param_docs = {p.arg_name: p.description for p in doc.params}
    return_description = doc.returns.description if doc.returns else ""
    
    fields = {}
    for name, param in inspect.signature(func).parameters.items():
        if name in ('self', 'cls'): 
            continue
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