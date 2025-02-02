import streamlit as st 
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


st.title("Option Pricer")

#variables


S0 = st.number_input("Stock Price", min_value=0.0, value=0.0, step=0.01)
K = st.number_input("Strike Price", min_value=0.0, value= 1.0, step=0.01)
sigma = st.number_input("Volatility in %", min_value=0.0, value=00.0, step=0.01)
r = st.number_input("Risk Free Rate in %", min_value=0.0, value=0.0, step=0.01)
T = st.number_input("Maturity in years", min_value=0.0, value=0.0, step=0.01)

    
sigma = sigma/100
r = r/100

N = 1000

def d1(S0, K, sigma, r, T):
    return (np.log(S0/K)+(r+(sigma*sigma)/2)*T) / (sigma*np.sqrt(T))

def d2(S0, K, sigma, r, T):  
    return (np.log(S0/K)+(r-(sigma*sigma)/2)*T) / (sigma*np.sqrt(T))

def call_price(S0, K, sigma, r, T):
    d1_value = (np.log(S0/K)+(r+(sigma*sigma)/2)*T) / (sigma*np.sqrt(T))
    d2_value = (np.log(S0/K)+(r-(sigma*sigma)/2)*T) / (sigma*np.sqrt(T))
    C = S0 * sp.stats.norm.cdf(d1_value)  - K* np.exp(-r*T)*  sp.stats.norm.cdf(d2_value)
    return C

def put_price(S0, K, sigma, r, T):
    d1_value = (np.log(S0/K)+(r+(sigma*sigma)/2)*T) / (sigma*np.sqrt(T))
    d2_value = (np.log(S0/K)+(r-(sigma*sigma)/2)*T) / (sigma*np.sqrt(T))
    P = K* np.exp(-r*T)*  sp.stats.norm.cdf(-d2_value) - S0 * sp.stats.norm.cdf(-d1_value)
    return P

def binomial_tree_american_put(S0, K, sigma, r, T, N):
    
    dt = T/N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(- sigma * np.sqrt(dt))
    q = (np.exp(r * dt) - d) / (u - d)
    
    S = [0]* (N+1)
    
    for j in range(0, N+1):
        S[j] = S0 * (d ** (N - j)) * (u ** j) 
    
    
    C = [0] * (N+1)
    for j in range(0,N+1):
        C[j] = max(0, K - S[j])
        
    for i in range(N-1, -1, -1):
        for j in range(0,i+1):
            S[j] = S0 * (d ** (i -j)) * (u ** j)
            C[j] = max( max(K-S[j], 0), np.exp(-r * dt) * ((q * C[j+1]) + (1 - q) * C[j]))
     
    
    return C[0]


st.write("The price of a American & European Call is : ", round(call_price(S0, K, sigma, r, T),2))
st.write("The price of a European Put is : ", round(put_price(S0, K, sigma, r, T),2))
st.write("The price of a American Put is : ", round(binomial_tree_american_put(S0, K, sigma, r, T, N),2))

# Create buttons inside columns
st.write("Select variable you want to change : ")

# Create 5 columns for buttons
col1, col2, col3, col4, col5 = st.columns(5)

# Define buttons in each column
with col1:
    S0_clicked = st.button("Stock Price")
with col2:
    K_clicked = st.button("Strike Price")
with col3:
    sigma_clicked = st.button("Volatility")
with col4:
    r_clicked = st.button("Risk Free Rate")
with col5:
    T_clicked = st.button("Maturity")
    
# Button to trigger action
# Logic for handling the buttons
if S0_clicked:
    st.write("You clicked the **Stock Price** button!")
    
    # Initialize the array with the first value
    price_array = [round(S0) - 10]
    
    # Populate the array with 21 values
    for i in range(1, 21):
        price_array.append(price_array[i-1] + 1)
    
    # Call and Put price arrays
    call_price_array = [call_price(price, K, sigma, r, T) for price in price_array]
    put_price_array = [put_price(price, K, sigma, r, T) for price in price_array]
    
    # Plot the results using matplotlib
    fig, ax = plt.subplots()
    ax.plot(price_array, call_price_array, label="Call Price", color='blue')
    ax.plot(price_array, put_price_array, label="Put Price", color='red')
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Option Price")
    ax.set_title("Option Prices for Different Stock Prices")
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)

elif K_clicked:
    st.write("You clicked the **Strike Price** button!")
    
    # Initialize the array with the first value
    strike_price_array = [round(K) - 5]
    
    # Populate the array with 11 values
    for i in range(1, 21):
        strike_price_array.append(strike_price_array[i-1] + 1)
    
    # Call and Put price arrays
    call_price_array = [call_price(S0, strike, sigma, r, T) for strike in strike_price_array]
    put_price_array = [put_price(S0, strike, sigma, r, T) for strike in strike_price_array]
    
    # Plot the results using matplotlib
    fig, ax = plt.subplots()
    ax.plot(strike_price_array, call_price_array, label="Call Price", color='blue')
    ax.plot(strike_price_array, put_price_array, label="Put Price", color='red')
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Option Price")
    ax.set_title("Option Prices for Different Strike Prices")
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)

elif sigma_clicked:
    st.write("You clicked the **Volatility** button!")
    
    # Initialize the array with the first value
    volatility_array = [round(sigma * 100) - 5]
    
    # Populate the array with 11 values
    for i in range(1, 21):
        volatility_array.append(volatility_array[i-1] + 1)
    
    # Call and Put price arrays
    call_price_array = [call_price(S0, K, vol / 100, r, T) for vol in volatility_array]
    put_price_array = [put_price(S0, K, vol / 100, r, T) for vol in volatility_array]
    
    # Plot the results using matplotlib
    fig, ax = plt.subplots()
    ax.plot(volatility_array, call_price_array, label="Call Price", color='blue')
    ax.plot(volatility_array, put_price_array, label="Put Price", color='red')
    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Option Price")
    ax.set_title("Option Prices for Different Volatilities")
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)

elif r_clicked:
    st.write("You clicked the **Risk Free Rate** button!")
    
    # Initialize the array with the first value
    risk_free_array = [round(r * 100) - 1]
    
    # Populate the array with 11 values
    for i in range(1, 21):
        risk_free_array.append(risk_free_array[i-1] + 0.1)
    
    # Call and Put price arrays
    call_price_array = [call_price(S0, K, sigma, rate / 100, T) for rate in risk_free_array]
    put_price_array = [put_price(S0, K, sigma, rate / 100, T) for rate in risk_free_array]
    
    # Plot the results using matplotlib
    fig, ax = plt.subplots()
    ax.plot(risk_free_array, call_price_array, label="Call Price", color='blue')
    ax.plot(risk_free_array, put_price_array, label="Put Price", color='red')
    ax.set_xlabel("Risk Free Rate (%)")
    ax.set_ylabel("Option Price")
    ax.set_title("Option Prices for Different Risk Free Rates")
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)

elif T_clicked:
    st.write("You clicked the **Maturity** button!")
    
    # Initialize the array with the first value
    maturity_array = [round(T) - 1]
    
    # Populate the array with 11 values
    for i in range(1, 21):
        maturity_array.append(maturity_array[i-1] + 0.1)
    
    # Call and Put price arrays
    call_price_array = [call_price(S0, K, sigma, r, maturity) for maturity in maturity_array]
    put_price_array = [put_price(S0, K, sigma, r, maturity) for maturity in maturity_array]
    
    # Plot the results using matplotlib
    fig, ax = plt.subplots()
    ax.plot(maturity_array, call_price_array, label="Call Price", color='blue')
    ax.plot(maturity_array, put_price_array, label="Put Price", color='red')
    ax.set_xlabel("Maturity (Years)")
    ax.set_ylabel("Option Price")
    ax.set_title("Option Prices for Different Maturities")
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)

#delta calculation
def delta_call(d1) : 
    return sp.stats.norm.cdf(d1)

def delta_put(d1):
    return  -sp.stats.norm.cdf(-d1)

st.write("Delta Call : ", delta_call(d1(S0, K, sigma, r, T)))
st.write("Delta Put : ", delta_put(d1(S0, K, sigma, r, T)))

#gamma calculation
def gamma(S0, K, sigma, r, T):
    return (sp.stats.norm.pdf(d1(S0, K, sigma, r, T),0,1))/(S0*sigma*np.sqrt(T))

st.write("Gamma : ", gamma(S0, K, sigma, r, T))

#vega Calculation
def vega(S0, K, sigma, r, T):
    return S0 * sp.stats.norm.pdf(d1(S0, K, sigma, r, T), 0, 1) * np.sqrt(T)*0.01
st.write("Vega : ", vega(S0, K, sigma, r, T))


#Theta Calculation
theta_call = (-((S0 * sp.stats.norm.pdf(d1(S0, K, sigma, r, T)) * sigma) / (2 * np.sqrt(T)) 
               - r * K * np.exp(-r * T) * sp.stats.norm.cdf(d2(S0, K, sigma, r, T))))/252
theta_put = (-((S0 * sp.stats.norm.pdf(d1(S0, K, sigma, r, T)) * sigma) / (2 * np.sqrt(T)) 
               + r * K * np.exp(-r * T) * sp.stats.norm.cdf(-d2(S0, K, sigma, r, T))))/252


st.write("Theta Call : ", theta_call)
st.write("Theta Put : ", theta_put)


# Create 5 columns for buttons
col1, col2, col3, col4, col5 = st.columns(5)

# Define buttons in each column
with col1:
    delta_clicked = st.button("Delta")
with col2:
    gamma_clicked = st.button("Gamma")
with col3:
    vega_clicked = st.button("Vega")
with col4:
    rho_clicked = st.button("Rho")
with col5:
    theta = st.button("Theta")
    
if delta_clicked:
    delta_price_array = []
    #start value = S0 * -80%
    #end value = S0+ * 80%
    
    delta_price_array.append(S0*0.2)
    start_value = round(S0*0.2)
    #print(start_value)
    end_value = round(S0*1.8)
    #print(end_value)
    
    nb_it = round((end_value - start_value)/0.1)
    #print(nb_it)
        
    for i in range(1,nb_it+1):
        delta_price_array.append(delta_price_array[0]+0.1*i)
        
    #print(delta_price_array[-1])
          
        
    d1_array = []
    delta_call_array = []
    delta_put_array = []

    for price in delta_price_array: 
        d1_val = d1(price, K, sigma, r, T)
        d1_array.append(d1_val)
        
    for element in d1_array:
        delta_call_array.append(delta_call(element))
        delta_put_array.append(delta_put(element))
        
    fig, ax = plt.subplots()
    ax.plot(delta_price_array, delta_call_array, label="Delta Call", color = 'blue')
    #ax.plot(delta_price_array, delta_put_array, label = 'Delta Put', color = 'red')
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Delta")
    ax.set_title("Delta vs. Stock Price")
    ax.legend()

    st.pyplot(fig)
    
elif gamma_clicked:
    
    gamma_price_array = []
    #start value = S0 * -80%
    #end value = S0+ * 80%
    
    gamma_price_array.append(S0*0.2)
    start_value = round(S0*0.2)
    #print(start_value)
    end_value = round(S0*1.8)
    #print(end_value)
    
    nb_it = round((end_value - start_value)/0.1)
    #print(nb_it)
        
    for i in range(1,nb_it+1):
        gamma_price_array.append(gamma_price_array[0]+0.1*i)
        
    gamma_values_array = []    
    for element in gamma_price_array:
        gamma_values_array.append(gamma(element, K, sigma, r, T))
        
    fig, ax = plt.subplots()
    ax.plot(gamma_price_array, gamma_values_array, label = "Gamma", color = "blue")
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Gamma")
    ax.set_title("Gamma vs. Stock Price")
    ax.legend()
    
    st.pyplot(fig)
    
elif vega_clicked:
    
    sigma = 0.2
    sigma_array = []
    
    sigma_array.append(sigma-0.10)
    for i in range(1, 21):
        sigma_array.append(sigma_array[0]+0.01*i)
       
    vega_array = []
    for element in sigma_array:
        vega_array.append(vega(S0, K, element, r, T))
    
    fig, ax = plt.subplots()
    ax.plot(sigma_array, vega_array, label = "Vega",color ="blue")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Vega")
    ax.set_title("Vega vs. Volatility")
    ax.legend()
    
    st.pyplot(fig)
        


def binomial_tree_euro_call(S0, K, sigma, r, T, N):
    #price at maturity time step N
    #variables 
    dt = T/N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))

    q = (np.exp(r * dt) - d) / (u - d)


    S = [0] * (N+1)    
    for j in range(0, N+1):
        S[j] = S0 * (d ** (N - j)) * (u ** j)
        
    # option price
    C = [0] * (N+1)
    for j in range(0,N+1):
        C[j] = max(0, S[j] - K)
        
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            C[j] = np.exp(-r * dt) * (q * C[j+1] + (1 - q) * C[j])

    
    print("Option Price at t=0:", C[0])
    
S0 = 100
K = 100
T = 1
r = 0.06
sigma = 0.20
N = 100
    
binomial_tree_euro_call(S0, K, sigma, r, T, N)

def binomial_tree_euro_put(S0, K, sigma, r, T, N):
    #price at maturity time step N
    #variables 
    dt = T/N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))

    q = (np.exp(r * dt) - d) / (u - d)


    S = [0] * (N+1)    
    for j in range(0, N+1):
        S[j] = S0 * (d ** (N - j)) * (u ** j)
        
    # option price
    C = [0] * (N+1)
    for j in range(0,N+1):
        C[j] = max(0, K - S[j])
        
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            C[j] = np.exp(-r * dt) * (q * C[j+1] + (1 - q) * C[j])

    
    print("Option Price at t=0:", C[0])
        
binomial_tree_euro_put(S0, K, sigma, r, T, N)




















