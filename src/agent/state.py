"""
LangGraph state definition for the Churn Retention Agent.
Defines the TypedDict that flows between all agent nodes.
Will be fully implemented in the agent implementation prompt.
"""

from typing import TypedDict, Optional


class ChurnAgentState(TypedDict):
    customer_data: dict  # Raw customer feature dict from the Streamlit form
    churn_probability: float  # Output from M1 ML model (0.0 to 1.0)
    risk_level: str  # Derived: "low", "medium", or "high"
    risk_drivers: list  # Key features contributing to churn risk
    retrieval_query: str  # Query string sent to RAG
    retrieved_strategies: list  # List of text chunks from Chroma
    llm_reasoning: str  # Raw LLM output from the planning node
    retention_report: dict  # Final structured report dict
    error: Optional[str]  # Populated if any node fails
