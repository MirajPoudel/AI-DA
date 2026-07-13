import pandas as pd
import multiprocessing
import builtins
import traceback

BANNED = ["import os", "import sys", "subprocess", "eval(", "exec(", "open(", "__import__", "importlib"]

# Modules the sandboxed code is allowed to import.
ALLOWED_MODULES = {
    "pandas", "numpy", "plotly", "plotly.express", "plotly.graph_objects",
    "math", "statistics", "datetime",
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Restricted __import__: only allows whitelisted modules."""
    top_level = name.split(".")[0]
    allowed_top_levels = {m.split(".")[0] for m in ALLOWED_MODULES}
    if top_level not in allowed_top_levels:
        raise ImportError(f"Import of '{name}' is not allowed in the sandbox")
    return builtins.__import__(name, globals, locals, fromlist, level)


# Safe builtins re-added since exec() below strips __builtins__ for security.
SAFE_BUILTINS = {
    "len": len, "range": range, "list": list, "dict": dict, "set": set,
    "str": str, "int": int, "float": float, "bool": bool, "sum": sum,
    "min": min, "max": max, "sorted": sorted, "enumerate": enumerate,
    "zip": zip, "abs": abs, "round": round, "print": print,
    "__import__": _safe_import,
}


def _worker(code, df, queue):
    try:
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
        local_env = {"df": df, "pd": pd, "np": np, "px": px, "go": go}
        exec(code, {"__builtins__": SAFE_BUILTINS}, local_env)
        queue.put({"result": local_env.get("result"), "fig": local_env.get("fig"), "error": None})
    except Exception:
        queue.put({"result": None, "fig": None, "error": traceback.format_exc()})


def run_code_safely(code: str, df: pd.DataFrame, timeout: int = 15) -> dict:
    for banned in BANNED:
        if banned in code:
            return {"result": None, "fig": None, "error": f"Blocked unsafe pattern: {banned}"}

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(code, df, queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return {"result": None, "fig": None, "error": f"Execution timed out after {timeout}s"}

    if p.exitcode != 0:
        return {"result": None, "fig": None, "error": f"Sandbox process exited unexpectedly (code {p.exitcode})"}

    if queue.empty():
        return {"result": None, "fig": None, "error": "Sandbox produced no output"}

    return queue.get_nowait()
