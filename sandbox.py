import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import multiprocessing
import traceback

BANNED = ["import os", "import sys", "subprocess", "eval(", "exec(", "open(", "__import__"]

# Safe builtins re-added since exec() below strips __builtins__ for security.
SAFE_BUILTINS = {
    "len": len, "range": range, "list": list, "dict": dict, "set": set,
    "str": str, "int": int, "float": float, "bool": bool, "sum": sum,
    "min": min, "max": max, "sorted": sorted, "enumerate": enumerate,
    "zip": zip, "abs": abs, "round": round, "print": print,
}


def _worker(code, df, return_dict):
    try:
        local_env = {"df": df, "pd": pd, "np": np, "px": px, "go": go}
        exec(code, {"__builtins__": SAFE_BUILTINS}, local_env)
        return_dict["result"] = local_env.get("result")
        return_dict["fig"] = local_env.get("fig")
        return_dict["error"] = None
    except Exception:
        return_dict["result"] = None
        return_dict["fig"] = None
        return_dict["error"] = traceback.format_exc()


def run_code_safely(code: str, df: pd.DataFrame, timeout: int = 15) -> dict:
    for banned in BANNED:
        if banned in code:
            return {"result": None, "fig": None, "error": f"Blocked unsafe pattern: {banned}"}

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=_worker, args=(code, df, return_dict))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        return {"result": None, "fig": None, "error": f"Execution timed out after {timeout}s"}

    return dict(return_dict)
