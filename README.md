Sentiment Analysis for Portfolio Division and Fama-French Enhancement
This project explores how news sentiment analysis can enhance portfolio construction and stock return forecasting using the Fama-French three-factor model. Conducted by Dickinson College students, it analyzes 88 companies across Technology, Financial Services, and Electric Vehicle (EV) sectors from 2018 to 2024.
Overview

Objective: Use sentiment analysis (via FinBERT) to stratify portfolios into terciles (Positive, Neutral, Negative) and augment the Fama-French model with sentiment scores for improved return prediction.
Sectors: Technology (e.g., AAPL, MSFT), Financial Services (e.g., GS, ALLY), and EV (e.g., TSLA, NIO).
Methods: Sentiment scoring with a 60-day rolling window, OLS regression, and fundamental analysis of financial ratios.
Test Cases: Apple (AAPL), Goldman Sachs (GS), and Ally Financial (ALLY).

Repository Structure

data/: Sentiment scores (./data/sentiment/) and quantitative data (./data/quantitative/).
results/: Plots (./results/plots/) and fundamental analysis visuals (./results/funda_plot/).
src/: Python scripts for data processing, modeling, and visualization.
paper.tex: LaTeX source for the project report.
README.md: This file.

Setup
Prerequisites

Python 3.8+
Libraries: pandas, numpy, statsmodels, sklearn, transformers, torch, gnews
LaTeX distribution (e.g., TeX Live) for compiling the paper
Git for version control

Installation

Clone the repository:
git clone https://github.com/annepham1512/Sentiment-Analysis-for-Portfolio-Division-and-Fama-French-Enhancemen.git
cd Sentiment-Analysis-for-Portfolio-Division-and-Fama-French-Enhancemen


Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

(Note: Create a requirements.txt file with listed libraries if not present.)

Ensure data files are in data/ (e.g., Fama-French factors from Kenneth French’s library).


Usage
Running the Analysis

Execute sentiment scoring:
python src/sentiment_scoring.py

(Assumes sentiment_scoring.py for FinBERT processing.)

Run Fama-French model:
python src/fama_french_model.py

(Assumes fama_french_model.py for OLS regression.)

Generate visualizations:
python src/visualize_results.py

(Assumes visualize_results.py for bar plots, heatmaps, radar charts.)


Compiling the Paper
Compile the LaTeX paper:
pdflatex paper.tex
pdflatex paper.tex  # Run twice for references

Results

Portfolio Stratification: Positive terciles (e.g., AMAT) show P/E 68.28, ROE 12.37%, Revenue Growth 12.35%. Negative terciles (e.g., TSLA) offer value opportunities.
Model Performance:
GS: FF + ESG+General (Train R² 0.7262, Test R² 0.5596).
AAPL: FF + ESG+General (Train R² 0.6965, Test R² 0.3796).
ALLY: FF baseline (Test R² 0.4530); limited by sparse data (6–27 days).


Limitations: Sparse sentiment data (19–27 days in 500) constrained predictive power.

Discussion

ESG sentiment enhances Fama-French models, especially for Technology firms.
Proposed strategy: 50% Positive, 30% Neutral, 20% Negative terciles.
Future work: Expand data, improve NLP models, test across market conditions.

Contributors

Anne Pham
Robin Nguyen
Emmanuel Arung Bate
Affiliation: Dickinson College

License
MIT License (Add a LICENSE file if not present.)
Acknowledgments

Data from Kenneth French’s library and gnews API.
Inspiration from Pisaneschi (2023) and Huang (2023).

Contact
For questions, open an issue or contact contributors via Dickinson College.
