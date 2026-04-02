# Project 5: Customer Churn Prediction & Agentic Retention Strategy

## From Predictive Analytics to Intelligent Intervention

### Project Overview

This project involves the design and implementation of an **AI-driven customer analytics system** that predicts customer churn and evolves into an agentic AI retention strategist.

- **Milestone 1:** Classical machine learning techniques applied to historical customer behavior data to predict churn risk and identify key drivers of disengagement.
- **Milestone 2:** Extension into an agent-based AI application that autonomously reasons about churn risk, retrieves retention best practices (RAG), and plans intervention strategies.

---

### Technology Stack

| Component | Technology |
| :--- | :--- |
| **ML Models (M1)** | Logistic Regression, Decision Trees, Random Forest, Scikit-Learn |
| **Agent Framework (M2)** | LangGraph, Chroma/FAISS (RAG) |
| **UI Framework** | Streamlit |
| **LLMs (M2)** | Open-source models or Free-tier APIs |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly |

---

### Milestones & Deliverables

#### Milestone 1: ML-Based Churn Prediction (Mid-Sem)

**Objective:** Identify customers at risk using historical behavioral data focus on classical ML pipelines *without LLMs*.

**Key Deliverables:**

**Problem Understanding & Business Context**
- Customer churn is a critical business challenge affecting revenue and growth
- Telco industry dataset with 19+ customer features
- Predictive analytics to enable proactive retention strategies

**System Architecture Diagram**
```
Data Input → Preprocessing → Feature Engineering → ML Models → Prediction → Visualization
```

**Working Local Application with UI (Streamlit)**
- Interactive web application with 4-page flow:
  1. Introduction page with project overview
  2. Customer data input form (19 features)
  3. Model selection (Logistic Regression, Decision Tree, Random Forest)
  4. Results page with predictions and visualizations

**Model Performance Evaluation Report**
- Three ML models implemented and evaluated
- Performance metrics: Accuracy, Precision, Recall, F1-Score
- Feature importance analysis
- Interactive visualizations with Plotly

**Current Status:** **COMPLETED**

---

#### Milestone 2: Agentic AI Retention Assistant (End-Sem)

**Objective:** Extend the system into an agentic strategist that reasons about risk and retrieves best practices to generate structured recommendations.

**Key Deliverables:**

**Publicly Deployed Application** (Link required)
- Deployment URL: [To be added]
- Platform: Streamlit Cloud

**Agent Workflow Documentation** (States & Nodes)
- LangGraph-based agent architecture
- RAG implementation for retention strategies
- State management and reasoning flow

**Structured Retention Report Generation**
- Automated recommendation engine
- Personalized intervention strategies
- Risk-based action plans

**GitHub Repository & Complete Codebase**
- Repository: [Current Repository]
- Well-documented code with comments
- Modular architecture

**Demo Video** (Max 5 mins)
- Application walkthrough
- Feature demonstration
- Use case scenarios

**Current Status:** 🔄 **IN PROGRESS**

---

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd genAI_capstone_project
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

---

## Milestone 1 Implementation Details

### Dataset
- **Source:** Telco Customer Churn Dataset
- **Records:** 7,043 customers
- **Features:** 19 customer attributes including:
  - Demographics (gender, senior citizen, partner, dependents)
  - Services (phone, internet, online security, backup, etc.)
  - Account information (tenure, contract, payment method, charges)

### Feature Engineering
- Categorical encoding (one-hot encoding)
- Feature scaling (StandardScaler)
- Handling missing values
- Feature importance analysis

### ML Models Implemented

1. **Logistic Regression**
   - Linear model for baseline performance
   - Interpretable coefficients
   - Fast training and prediction

2. **Decision Tree**
   - Non-linear decision boundaries
   - Feature interaction capture
   - Visual decision rules

3. **Random Forest**
   - Ensemble of decision trees
   - Robust to overfitting
   - High accuracy

### Model Evaluation Metrics
- Accuracy Score
- Precision, Recall, F1-Score
- ROC-AUC Curve
- Confusion Matrix
- Feature Importance Rankings

### UI Features
- **Modern Design:** Gradient backgrounds, smooth animations
- **Interactive Forms:** 19 input fields organized by category
- **Model Selection:** Choose from 3 ML algorithms
- **Visualizations:** 
  - Churn probability gauge
  - Feature importance bar chart
  - Probability distribution
  - Risk level indicators

---

## 📁 Project Structure

```
genAI_capstone_project/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
├── data/
│   └── Telco-Customer-Churn.csv   # Dataset
├── models/
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   └── model_columns.pkl
├── src/
│   ├── preprocessing.py        # Data preprocessing
│   ├── model_training.py       # Model loading
│   └── evaluation.py           # Model evaluation
└── notebook/
    └── Telco_Customer_Churn.ipynb  # EDA & Training
```

---

## Application Flow

1. **Introduction Page**
   - Project overview and key features
   - Business context explanation
   - Call-to-action to start prediction

2. **Data Input Page**
   - Comprehensive form with 19 customer features
   - Organized by categories (Demographics, Services, Billing)
   - Input validation and user-friendly interface

3. **Model Selection Page**
   - Three model cards with descriptions
   - Visual icons and hover effects
   - Easy model selection

4. **Results Page**
   - Large churn risk indicator (High/Low)
   - Probability percentage display
   - Interactive visualizations:
     - Churn probability gauge
     - Top 10 feature importance chart
     - Probability distribution bar chart
   - Action buttons (Try another model, New prediction, Home)

---

## Key Achievements (Milestone 1)

- Successfully implemented 3 ML models with high accuracy  
- Created intuitive, industry-standard UI with Streamlit  
- Developed comprehensive data preprocessing pipeline  
- Implemented interactive visualizations with Plotly  
- Feature importance analysis for business insights  
- Real-time prediction with probability scores  
- Responsive design for all screen sizes  
- Modular, maintainable code architecture  

---

## Future Work (Milestone 2)

- [ ] Implement LangGraph-based agentic AI system
- [ ] Integrate RAG for retention strategy retrieval
- [ ] Add reasoning and planning capabilities
- [ ] Generate structured retention reports
- [ ] Deploy to Streamlit Cloud
- [ ] Create demo video
- [ ] Complete documentation

---

## Team Members

- Shreyash Golhani
- Gokul VKS
- Vaageesh Kumar SIngh
- Mohammad Affan Anas

---

## License

This project is developed for educational purposes as part of the Intro to GenAI course.

---

## Contact

For questions or feedback, please contact [Add contact information]

---

**Last Updated:** February 2026
