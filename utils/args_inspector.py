import inspect
import re
from typing import Dict, Any, Callable, List


def parse_docstring(docstring: str, drop_keys: list = ["additional_list_args"]) -> Dict[str, Dict[str, Any]]:
  parsed = {}
  if not docstring:
    return parsed

  lines = docstring.split("\n")
  param_pattern = re.compile(r"\s*(\w+) \(([^)]+)\): (.+)")

  for line in lines:
    match = param_pattern.match(line.strip())
    if match:
      arg_name, arg_type, desc = match.groups()
      if arg_name in drop_keys:
        continue
      # Remove 'optional' from type
      arg_type = arg_type.replace(", optional", "")
      try:
          parsed[arg_name] = {"desc": desc, "type": eval(
              arg_type, {"List": List}), "default": None}
      except NameError:
          parsed[arg_name] = {"desc": desc, "type": Any, "default": None}  # Handle unknown types

  return parsed


def inspect_function_args(func: Callable, drop_keys: list = ["additional_list_args"]) -> Dict[str, Dict[str, Dict[str, Any]]]:
  signature = inspect.signature(func)
  docstring = inspect.getdoc(func) or ""
  parsed_docs = parse_docstring(docstring, drop_keys)

  required_args = {}
  optional_args = {}

  for param_name, param in signature.parameters.items():
    if param_name in drop_keys:
      continue
    arg_info = parsed_docs.get(
        param_name, {"desc": "", "type": Any, "default": None})

    if param.default is inspect.Parameter.empty:
      required_args[param_name] = arg_info
    else:
      arg_info["default"] = param.default
      optional_args[param_name] = arg_info

  return {"required_args": required_args, "optional_args": optional_args}


if __name__ == "__main__":
  import os
  import sys
  sys.path.append(os.path.join(os.path.dirname(__file__), "../template/code/1/"))
  from openai_server_starter import OpenAI_APIServer
  
  out = inspect_function_args(OpenAI_APIServer.from_llamacpp_backend, drop_keys=[])
  print(out)