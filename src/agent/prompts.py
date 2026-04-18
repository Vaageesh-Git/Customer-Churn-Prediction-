"""
Prompt templates for the Churn Retention Agent.
All LLM prompts are defined here to keep nodes.py clean and prompts easy to iterate.
"""

RETENTION_SYSTEM_PROMPT = """You are an expert customer retention strategist for a telecom company.
Your role is to analyze a customer's churn risk profile and recommend specific, personalized, actionable retention strategies.

You have access to retention research and best practices retrieved from industry knowledge bases.
Use this knowledge to ground your recommendations in proven strategies.

Rules you must follow:
1. Be specific - reference the customer's actual situation (their contract, tenure, services).
2. Be actionable - every recommendation must be something a retention team can execute today.
3. Be honest - if a customer has very low churn risk, say so and recommend lighter-touch actions.
4. Avoid hallucinating offers or policies not grounded in the retrieved knowledge.
5. You MUST respond with a single valid JSON object and nothing else - no preamble, no markdown fences, no trailing text.

The JSON must have exactly these keys:
- "risk_summary": string, 2-3 sentences explaining why this customer is at risk
- "recommended_actions": list of 3-5 strings, each a concrete retention action
- "reasoning": string, 2-3 sentences explaining why these actions fit this specific customer
"""


def build_retention_user_prompt(
    churn_probability: float,
    risk_level: str,
    risk_drivers: list,
    customer_data: dict,
    retrieved_chunks: list,
) -> str:
    """Build the user-turn prompt sent to the LLM for the plan_intervention node."""

    # Format customer details into readable lines
    customer_lines = []
    field_labels = {
        "gender": "Gender",
        "SeniorCitizen": "Senior Citizen",
        "Partner": "Has Partner",
        "Dependents": "Has Dependents",
        "tenure": "Tenure (months)",
        "InternetService": "Internet Service",
        "Contract": "Contract Type",
        "PaymentMethod": "Payment Method",
        "MonthlyCharges": "Monthly Charges ($)",
        "TotalCharges": "Total Charges ($)",
        "TechSupport": "Tech Support",
        "OnlineSecurity": "Online Security",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Movies",
        "PhoneService": "Phone Service",
    }
    for key, label in field_labels.items():
        val = customer_data.get(key, "N/A")
        if key == "SeniorCitizen":
            val = "Yes" if val == 1 else "No"
        customer_lines.append(f"  - {label}: {val}")

    customer_profile_text = "\n".join(customer_lines)
    drivers_text = "\n".join(f"  - {d}" for d in risk_drivers)

    # Format retrieved knowledge chunks
    if retrieved_chunks:
        chunks_text = "\n\n".join(
            f"[Source {i + 1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)
        )
    else:
        chunks_text = (
            "No specific knowledge retrieved. Use general retention best practices."
        )

    return f"""Customer Risk Profile:
- Churn Probability: {churn_probability:.1%}
- Risk Level: {risk_level.upper()}
- Key Risk Drivers:
{drivers_text}

Customer Details:
{customer_profile_text}

Relevant Retention Knowledge (retrieved from research knowledge base):
{chunks_text}

Based on the above, provide your retention strategy recommendation as a JSON object.
Remember: respond with ONLY the JSON object, no other text."""
