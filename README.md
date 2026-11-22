
# Autonomous Supply Chain Intelligence and Optimization Platform

An end-to-end supply chain analytics and optimization platform that combines Machine Learning, Operations Research, simulation, and AI-driven decision support to improve demand forecasting, inventory planning, supplier allocation, and logistics operations.

## Project Overview

This project demonstrates how data analytics, machine learning, and mathematical optimization can be integrated into a supply chain management system.

It uses synthetic supply chain data to simulate business operations involving suppliers, warehouses, retail stores, products, and historical sales.

The platform covers the complete workflow, from demand forecasting to inventory optimization, supplier allocation, vehicle routing, disruption simulation, and AI-assisted recommendations.

## Key Features

- Demand forecasting using XGBoost.
- Inventory optimization using Economic Order Quantity (EOQ).
- Safety stock and Reorder Point (ROP) calculation.
- Supplier allocation using mathematical optimization.
- Vehicle routing using Google OR-Tools.
- Supply chain disruption simulation using SimPy.
- AI-assisted decision support using LangGraph and GroqCloud.
- Interactive dashboard using Streamlit and Plotly.

## Project Workflow

1. Data generation and preprocessing.
2. Exploratory Data Analysis.
3. Demand forecasting.
4. Inventory optimization.
5. Supplier allocation and capacity planning.
6. Warehouse-to-store assignment.
7. Vehicle routing optimization.
8. Supply chain disruption simulation.
9. AI-based decision support.
10. Interactive dashboard visualization.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Optimization | Pyomo, HiGHS |
| Vehicle Routing | Google OR-Tools |
| Simulation | SimPy |
| AI Workflow | LangGraph, GroqCloud |
| Dashboard | Streamlit, Plotly |
| Version Control | Git, GitHub |

## Dataset Description

The project uses synthetically generated supply chain data.

| Dataset | Records |
|---|---:|
| Suppliers | 3 |
| Warehouses | 10 |
| Retail Stores | 20 |
| Products (SKUs) | 100 |
| Historical Sales | 360,000 |
| Inventory Records | 1,000 |

The historical sales dataset contains 180 days of sales information.

## 1. Demand Forecasting

An XGBoost regression model is used to forecast product demand.

### Features Used

- Lag 1 demand
- Lag 7 demand
- 7-day rolling mean
- Day of the week
- Month
- Day of the year

### Model Configuration

- Algorithm: XGBoost Regressor
- Number of estimators: 300
- Maximum depth: 6
- Learning rate: 0.05
- Chronological train-test split: 80:20

### Model Performance

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Lag-7 Baseline | 4.2610 | 5.3997 | 20.29% |
| 7-Day Moving Average | 3.3450 | 4.2925 | 15.93% |
| XGBoost | 3.1223 | 3.9461 | 14.87% |

The XGBoost model achieved a WAPE of approximately 14.87%, outperforming the evaluated baseline models.

## 2. Inventory Optimization

Inventory planning is performed using Operations Research techniques.

### Methods Used

- Economic Order Quantity (EOQ)
- Safety Stock
- Reorder Point
- Target Stock Calculation
- Replenishment Planning

### Key Results

- Total products analyzed: 100
- Average EOQ: 1,393.14 units
- Planned replenishment: 348,922 units
- Network stock: 69,797 units
- Average inventory coverage: 2.14 days

The model uses configurable assumptions for ordering cost, lead time, and service level.

## 3. Supplier Allocation Optimization

Supplier allocation considers supplier capacity and allocation priorities.

Pyomo and the HiGHS solver are used for cost and risk-adjusted supplier assignment.

### Recorded Results

| Metric | Result |
|---|---:|
| Planned Quantity | 348,922 units |
| Available Capacity | 34,126 units |
| Allocated Quantity | 34,126 units |
| Unallocated Quantity | 314,796 units |
| Capacity Utilization | 100% |
| Base Procurement Cost | ₹16,902,890 |
| Risk Premium | ₹2,208,763.63 |
| Total Cost | ₹19,111,653.63 |

The available supplier capacity is insufficient to satisfy the complete planned requirement. The model therefore identifies the unallocated quantity instead of assuming that all demand can be fulfilled.

## 4. Vehicle Routing Optimization

Google OR-Tools is used to solve Capacitated Vehicle Routing Problems (CVRP).

The logistics module assigns stores to warehouses using geographic distance and optimizes vehicle routes.

### Results

- Retail stores: 20
- Vehicles: 14
- Vehicle capacity: 5,000 units
- Total route distance: 6,209.72 km

The current implementation uses straight-line geographic distance for routing calculations.

## 5. Supply Chain Disruption Simulation

SimPy is used to simulate supply chain disruptions over a 30-day period.

### Disruption Scenarios

- Reduced supplier reliability.
- Supplier delivery delays of 2–5 days.
- Demand increase of 30% during selected days.

### Recorded Results

| Metric | Baseline | Disruption |
|---|---:|---:|
| Service Level | 7.45% | 6.57% |

The disruption scenario also resulted in an increase of 79,294 lost sales units.

These results represent the project's synthetic simulation scenario and are not real-world operational measurements.

## 6. AI-Based Decision Support

The project integrates LangGraph and GroqCloud to create an AI-assisted decision-support workflow.

The workflow uses structured supply chain information to generate recommendations and evaluate them through a critic stage.

The final recommendations are intended for human review and approval before implementation.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Saurabhv15/Autonomous_Supply_Chain_Optimization.git
cd Autonomous_Supply_Chain_Optimization
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root and add the required API key:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not upload your actual API key or `.env` file to GitHub.

### 5. Run the Application

If the Streamlit entry point in your repository is `app.py`, run:

```bash
streamlit run app.py
```

Use the actual dashboard entry-point filename if it is different.

## Project Structure

```text
Autonomous_Supply_Chain_Optimization/
│
├── agents/          # AI agents and decision-support workflows
├── app/             # Streamlit application and dashboard
├── data/            # Supply chain datasets
├── figures/         # Charts and visualizations
├── models/          # Trained machine learning models
├── notebooks/       # Jupyter notebooks and experiments
├── reports/         # Analysis reports and results
├── src/             # Core project source code
│
├── config.py        # Project configuration
├── requirements.txt # Python dependencies
├── .gitignore       # Git ignore rules
└── README.md        # Project documentation
```


## Limitations

- The project uses synthetic data rather than real company data.
- Forecasting performance depends on the generated dataset and selected features.
- Supplier capacity is limited in the current scenario.
- Geographic routing uses straight-line distances rather than live road-network distances.
- Disruption simulation is based on configurable assumptions.
- AI-generated recommendations require human validation.

## Future Improvements

- Integrate real-world supply chain datasets.
- Add stochastic and robust optimization models.
- Improve supplier risk assessment.
- Integrate real road-network distances and traffic information.
- Implement real-time inventory monitoring.
- Add automated model retraining and forecasting.
- Improve multi-warehouse and multi-period optimization.
- Add detailed cost and service-level sensitivity analysis.

## Author

**Saurabh Verma**

M.Tech in Operations Research  
National Institute of Technology, Durgapur

- GitHub: https://github.com/Saurabhv15
- Portfolio: https://saurabhv15.github.io/
