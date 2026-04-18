"""
LangGraph node functions for the Churn Retention Agent.
Each function represents one step in the reasoning pipeline.
"""

import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from src.agent.state import ChurnAgentState
from src.agent.prompts import RETENTION_SYSTEM_PROMPT, build_retention_user_prompt
from src.agent.retriever import retrieve_strategies as rag_retrieve


def assess_risk(state: ChurnAgentState) -> ChurnAgentState:
    """
    Node 1: Risk Assessment
    Classifies churn_probability into risk_level (low/medium/high).
    Extracts human-readable risk_drivers from customer_data field values.
    """
    prob = state["churn_probability"]
    data = state["customer_data"]

    # Classify risk tier
    if prob >= 0.65:
        risk_level = "high"
    elif prob >= 0.35:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Map raw customer features to human-readable risk signals
    drivers = []

    contract = data.get("Contract", "")
    if contract == "Month-to-month":
        drivers.append("month-to-month contract (no long-term commitment)")
    elif contract == "One year":
        drivers.append("one-year contract (moderate commitment)")

    tenure = data.get("tenure", 0)
    if tenure < 12:
        drivers.append(f"very short tenure ({tenure} months)")
    elif tenure < 24:
        drivers.append(f"short tenure ({tenure} months)")

    internet = data.get("InternetService", "")
    if internet == "Fiber optic":
        drivers.append("fiber optic service (higher churn segment)")

    security = data.get("OnlineSecurity", "")
    if security == "No":
        drivers.append("no online security add-on")

    tech = data.get("TechSupport", "")
    if tech == "No":
        drivers.append("no tech support subscription")

    payment = data.get("PaymentMethod", "")
    if payment == "Electronic check":
        drivers.append("electronic check payment (lower engagement signal)")

    monthly = data.get("MonthlyCharges", 0)
    if monthly > 70:
        drivers.append(f"high monthly charges (${monthly:.0f})")

    senior = data.get("SeniorCitizen", 0)
    if senior == 1:
        drivers.append("senior citizen (higher churn demographic)")

    partner = data.get("Partner", "No")
    dependents = data.get("Dependents", "No")
    if partner == "No" and dependents == "No":
        drivers.append("no partner or dependents (lower switching cost)")

    paperless = data.get("PaperlessBilling", "No")
    if paperless == "Yes":
        drivers.append("paperless billing enabled")

    # Guarantee at least one driver even for edge cases
    if not drivers:
        drivers.append(f"churn probability of {prob:.1%} detected by ML model")

    return {
        **state,
        "risk_level": risk_level,
        "risk_drivers": drivers,
    }


def retrieve_strategies(state: ChurnAgentState) -> ChurnAgentState:
    """
    Node 2: RAG Retrieval
    Builds a targeted retrieval query from the risk profile.
    Calls the Chroma vector store to fetch relevant strategy chunks.
    Falls back gracefully if the vector store is unavailable.
    """
    risk_level = state["risk_level"]
    drivers = state["risk_drivers"]
    data = state["customer_data"]

    # Build a rich, specific query for semantic search
    contract = data.get("Contract", "")
    tenure = data.get("tenure", 0)
    internet = data.get("InternetService", "")

    query = (
        f"customer retention strategy for {risk_level} churn risk customer "
        f"with {contract.lower()} contract, {tenure} months tenure"
    )
    if internet:
        query += f", {internet.lower()} internet service"
    if drivers:
        # Add the top 2 drivers as additional context
        query += f". Key issues: {', '.join(drivers[:2])}"

    # Attempt RAG retrieval with graceful fallback
    try:
        use_cloud = os.getenv("USE_CHROMA_CLOUD", "false").lower() == "true"
        chunks = rag_retrieve(query, k=4, use_cloud=use_cloud)
    except Exception as e:
        print(f"[retrieve_strategies] RAG retrieval failed: {e}")
        chunks = []

    return {
        **state,
        "retrieval_query": query,
        "retrieved_strategies": chunks,
    }


def plan_intervention(state: ChurnAgentState) -> ChurnAgentState:
    """
    Node 3: LLM-Based Planning
    Sends the customer risk profile and retrieved knowledge to Groq LLM.
    Instructs the LLM to respond in structured JSON format.
    Stores the raw LLM response string in llm_reasoning.
    """
    # Resolve API key: try Streamlit secrets first, fall back to env var
    groq_api_key = None
    try:
        import streamlit as st

        groq_api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass
    if not groq_api_key:
        groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        return {
            **state,
            "llm_reasoning": "",
            "error": (
                "GROQ_API_KEY is not set. Add it to your .env file locally "
                "or to Streamlit Secrets for deployment."
            ),
        }

    # Build prompts
    user_prompt = build_retention_user_prompt(
        churn_probability=state["churn_probability"],
        risk_level=state["risk_level"],
        risk_drivers=state["risk_drivers"],
        customer_data=state["customer_data"],
        retrieved_chunks=state["retrieved_strategies"],
    )

    # Call Groq
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=groq_api_key,
            temperature=0.3,  # Low temp for consistent, factual output
            max_tokens=1024,
        )
        messages = [
            SystemMessage(content=RETENTION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        raw_output = response.content.strip()
    except Exception as e:
        return {
            **state,
            "llm_reasoning": "",
            "error": f"LLM call failed: {str(e)}",
        }

    return {
        **state,
        "llm_reasoning": raw_output,
    }


def generate_report(state: ChurnAgentState) -> ChurnAgentState:
    """
    Node 4: Report Generation
    Parses the LLM JSON output into a structured retention_report dict.
    Appends source references and a mandatory ethical disclaimer.
    Falls back to a safe structured response if JSON parsing fails.
    """
    import json

    # If a prior node set an error, propagate a safe fallback report
    if state.get("error"):
        report = {
            "risk_summary": f"Risk level assessed as {state.get('risk_level', 'unknown')} "
            f"with churn probability of {state.get('churn_probability', 0):.1%}.",
            "recommended_actions": [
                "Contact customer proactively via their preferred channel.",
                "Offer a contract upgrade incentive tailored to their usage.",
                "Schedule a customer success check-in call.",
            ],
            "reasoning": "Recommendations are based on general best practices as the AI "
            "reasoning step encountered an error.",
            "sources": [],
            "ethical_disclaimer": _ethical_disclaimer(),
            "agent_error": state.get("error"),
        }
        return {**state, "retention_report": report}

    raw = state.get("llm_reasoning", "")

    # Attempt JSON parse - strip markdown fences if the LLM added them
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract a JSON object with a simple heuristic
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass

    if parsed is None:
        # Full fallback: LLM returned something but it wasn't parseable JSON
        parsed = {
            "risk_summary": raw[:300] if raw else "Risk analysis completed.",
            "recommended_actions": [
                "Review customer account for upgrade opportunities.",
                "Reach out with a personalised retention offer.",
                "Consider a loyalty discount based on tenure.",
            ],
            "reasoning": "Structured output parsing failed; actions are standard best-practice fallbacks.",
        }

    # Build sources list from retrieved chunk previews
    chunks = state.get("retrieved_strategies", [])
    sources = []
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk[:120].replace("\n", " ").strip()
        sources.append(f'Knowledge Base Source {i + 1}: "{preview}..."')

    report = {
        "risk_summary": parsed.get("risk_summary", ""),
        "recommended_actions": parsed.get("recommended_actions", []),
        "reasoning": parsed.get("reasoning", ""),
        "sources": sources,
        "ethical_disclaimer": _ethical_disclaimer(),
    }

    return {**state, "retention_report": report}


def _ethical_disclaimer() -> str:
    return (
        "⚠️ Ethical Disclaimer: This retention strategy was generated by an AI system "
        "and is intended as decision-support for human agents, not as an autonomous "
        "decision-making tool. Recommendations should be reviewed by a qualified "
        "customer success professional before any action is taken. Customer data used "
        "in this analysis must be handled in accordance with applicable data protection "
        "regulations (e.g., GDPR, CCPA). The probability score is a statistical estimate "
        "and does not guarantee future customer behaviour."
    )
