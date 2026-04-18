"""
LangGraph node functions for the Churn Retention Agent.
Each function represents one step in the reasoning pipeline.
Will be fully implemented in the agent implementation prompt.
"""

from src.agent.state import ChurnAgentState


def assess_risk(state: ChurnAgentState) -> ChurnAgentState:
    """
    Node 1: Risk Assessment
    Reads churn_probability and customer_data from state.
    Classifies risk_level as low / medium / high.
    Extracts top risk_drivers from the customer feature values.
    Returns updated state with risk_level and risk_drivers populated.
    """
    raise NotImplementedError("To be implemented in agent implementation prompt.")


def retrieve_strategies(state: ChurnAgentState) -> ChurnAgentState:
    """
    Node 2: RAG Retrieval
    Builds a retrieval_query from risk_level and risk_drivers.
    Calls retriever.retrieve_strategies() to fetch relevant chunks.
    Returns updated state with retrieval_query and retrieved_strategies populated.
    """
    raise NotImplementedError("To be implemented in agent implementation prompt.")


def plan_intervention(state: ChurnAgentState) -> ChurnAgentState:
    """
    Node 3: LLM-Based Planning
    Constructs a prompt from customer profile + retrieved strategy chunks.
    Calls Groq LLM via LangChain to generate retention recommendations.
    Returns updated state with llm_reasoning populated.
    """
    raise NotImplementedError("To be implemented in agent implementation prompt.")


def generate_report(state: ChurnAgentState) -> ChurnAgentState:
    """
    Node 4: Report Generation
    Structures llm_reasoning into a clean retention_report dict.
    Format: { risk_summary, recommended_actions, sources, ethical_disclaimer }
    Returns updated state with retention_report populated.
    """
    raise NotImplementedError("To be implemented in agent implementation prompt.")
