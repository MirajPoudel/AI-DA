from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, END
import pandas as pd

from agents.profiler import profile_dataset
from agents.query_agent import plan_query
from agents.code_agent import generate_code
from agents.insight_agent import generate_insight
from sandbox import run_code_safely


class GraphState(TypedDict):
    df: pd.DataFrame
    user_query: str
    llm: Any
    profile: Optional[dict]
    plan: Optional[dict]
    code: Optional[str]
    result: Optional[Any]
    fig: Optional[Any]
    error: Optional[str]
    answer: Optional[str]
    description: Optional[str]


def profile_node(state: GraphState) -> GraphState:
    state["profile"] = profile_dataset(state["df"])
    return state


def plan_node(state: GraphState) -> GraphState:
    state["plan"] = plan_query(state["profile"], state["user_query"], state["llm"])
    return state


def code_node(state: GraphState) -> GraphState:
    state["code"] = generate_code(state["plan"], state["profile"], state["user_query"], state["llm"])
    return state


def execute_node(state: GraphState) -> GraphState:
    output = run_code_safely(state["code"], state["df"])
    state["result"] = output["result"]
    state["fig"] = output["fig"]
    state["error"] = output["error"]
    return state


def insight_node(state: GraphState) -> GraphState:
    if state["error"]:
        state["answer"] = "Execution failed."
        state["description"] = state["error"]
    else:
        insight = generate_insight(state["user_query"], state["result"], state["llm"])
        state["answer"] = insight["answer"]
        state["description"] = insight["description"]
    return state


def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("profile", profile_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("code", code_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("insight", insight_node)

    workflow.set_entry_point("profile")
    workflow.add_edge("profile", "plan")
    workflow.add_edge("plan", "code")
    workflow.add_edge("code", "execute")
    workflow.add_edge("execute", "insight")
    workflow.add_edge("insight", END)

    return workflow.compile()
