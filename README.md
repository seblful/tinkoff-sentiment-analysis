# Sentiment Analysis of Financial Records on Tinkoff Website

This project aims to analyze the sentiment of financial records on a website using natural language processing techniques. By analyzing the tone and emotion of financial statements, earnings reports, and other financial documents, this tool can help investors and analysts make more informed decisions.

## Components

**Web Scraping**: The project includes a script that scrapes user messages from a website's API, allowing for the collection of data for sentiment analysis.

**Sentiment Classification:** The project utilizes the "rubert-tiny2-russian-sentiment" model and XGBoost Dart to classify the sentiment of the scraped user messages. This allows for the identification of positive, negative, or neutral sentiment in the messages.

**Forecasting:** The project includes a script that forecasts the sentiment of new user messages based on the trained model. This enables the prediction of future sentiment trends in the stock market based on user discussions.
