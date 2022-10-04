# Cryptocurrency Prediction Application

Developed an application to predict the price of cryprocurrency coins for the next day using Python, a deep learning model – LSTM and streamlit for design of the web application. 
25 cryptocurrencies were selected at random and utilized in this app. 
Live data was gotten directly from yahoo finance from respective coins start date using the pandas data reader for web scraping. 
The moving average, trends and time series analysis were performed using Matplotlib and Tableau for the visualizations.
Hyperparameter tuning was done manually, and LSTM achieved a R2 score of 95% and a MSE of 0.1.

Library Requirements before running the app:
- numpy--1.22.3
- matplotlib--3.5.2
- pandas-datareader--0.10.0
- pandas--1.4.3
- scikit-learn--1.1.1
- streamlit--1.11.0
- tensorflow--2.9.1
- yfinance--0.1.74

To run:
- ---> Go to terminal
- ---> Type in streamlit run .\main.py
- ---> Copy the Local URL into your web browser and interact.








