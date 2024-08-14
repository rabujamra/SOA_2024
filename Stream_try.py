import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
a = 2
b = 5
c = 3
C = 4  # Constant cost (applied to R1)

# Define the functions R0 and R1
def R0(x):
    return a * x + (b - c)

def R1(x):
    return a * x + b

# Define the cost function (could be more complex if needed)
def Cost(x):
    return C

# Define the ratios with cost for R1
def ratio_R0(x, k):
    return (R0(x)**k) #/ (C**(1 - k))

def ratio_R1(x, k):
    return (R1(x)**k) / (Cost(x)**(1 - k))

# Streamlit app
st.title('Interactive Plot for R0 and R1 Ratios')

# Slider for k value
k = st.slider('Select value of k', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Define a range of x values
x = np.linspace(0, 10, 100)

# Compute the ratios
R0_plot = ratio_R0(x, k)
R1_plot = ratio_R1(x, k)

# Create a plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, R0_plot, label=f'R0(x)^{k} / C^(1-k)', color='blue')
ax.plot(x, R1_plot, label=f'R1(x)^{k} / Cost(x)^{round(1-k, 2)}', color='red')

# Set axis properties
ax.set_xlabel('x')
ax.set_ylabel('Ratio')
ax.set_title(f'Ratios for k={k}')
ax.legend()
ax.grid(True)

# Show plot in Streamlit
st.pyplot(fig)
