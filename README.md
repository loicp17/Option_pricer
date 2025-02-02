# Option Pricer - Streamlit App

This project is a **Streamlit-based Option Pricer**, which allows users to calculate **European and American option prices** using **Black-Scholes** and **Binomial Tree models**. Additionally, it computes **Greeks** such as Delta, Gamma, Vega, Theta, and visualizes how option prices change with different parameters.

## Features

✅ **Option Pricing**  
- Black-Scholes model for **European Call and Put Options**  
- Binomial Tree model for **American Put Options**  

✅ **Greek Calculations**  
- Delta, Gamma, Vega, Theta for both Call and Put options  

✅ **Interactive Parameter Adjustment**  
- Modify **Stock Price, Strike Price, Volatility, Risk-Free Rate, and Maturity** dynamically  
- **Visualize** the impact of each parameter on option pricing using **matplotlib**  

✅ **User-Friendly UI**  
- Built with **Streamlit** for an interactive and easy-to-use experience  

## Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/option-pricer.git
cd option-pricer
```

### 2️⃣ Install Dependencies
Make sure you have Python installed, then install required libraries:
```bash
pip install streamlit numpy scipy matplotlib
```

### 3️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

## Usage

1. Enter values for **Stock Price, Strike Price, Volatility, Risk-Free Rate, and Maturity**.
2. View computed **European and American option prices**.
3. Click buttons to **visualize** how changes in variables affect option prices.
4. Check the **Greeks** for risk assessment.
