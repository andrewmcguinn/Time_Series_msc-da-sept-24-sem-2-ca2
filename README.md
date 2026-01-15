# Time_Series_msc-da-sept-24-sem-2-ca2

# 📈 Scalable Stock Price Prediction with Big Data & Deep Learning

End-to-end stock price prediction system combining **NoSQL databases**, **distributed computing**, **time series forecasting**, and **deep learning**.

## 🚀 Overview

This project implements a **scalable big data architecture** for stock price prediction using historical market data and social sentiment. It integrates **MongoDB**, **Apache Spark**, **time series models**, and **LSTM networks**, with support for **batch and near real-time streaming pipelines**.

## 🏗️ Architecture

* **Storage:** MongoDB (NoSQL, schema-flexible)
* **Processing:** Apache Spark (RDDs, DAG execution, in-memory computation)
* **Streaming:** Kafka-style simulated pipeline (Reddit API → Spark)
* **Modeling:** SARIMAX, Recursive Forecasting, LSTM
* **Benchmarking:** YCSB (MongoDB vs MySQL vs Cassandra)

## 🛠️ Key Features

* Distributed data ingestion and preprocessing with Spark
* NoSQL performance benchmarking using **YCSB workloads A, B, C**
* **SARIMAX** with exogenous sentiment variables
* **Recursive multi-step forecasting** using SKForecast
* **Multivariate LSTM** models with TensorFlow & Keras Tuner
* Near real-time sentiment streaming inspired by **Apache Kafka**
* Interactive visualizations following **Tufte’s data-ink principles**

## 📊 Models Implemented

* **SARIMAX** (seasonality + exogenous variables)
* **Recursive Forecasting** (multi-step horizon prediction)
* **LSTM** (multivariate deep learning model)

## ⚙️ Benchmarks & Evaluation

* Database performance evaluated using **YCSB**
* Metrics: throughput, latency, runtime, percentile latency
* MongoDB demonstrated superior performance across mixed read/write workloads

## 📈 Data Sources

* Historical stock prices (FAANG)
* Reddit sentiment data (`r/stocks`) via PRAW
* Simulated streaming batches for real-time analysis

## 🧰 Tech Stack

`Python` · `MongoDB` · `Apache Spark` · `YCSB` · `TensorFlow` · `Statsmodels` · `SKForecast` · `Pandas` · `NumPy`
