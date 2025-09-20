# Streamlit-Project

This project is a **data science application** built with **Streamlit**.  
The goal is to identify whether a bank customer should be classified as **"Risk"** or **"No Risk"** for a lending system, based on customer characteristics such as:

- Gender  
- Age  
- Credit History  
- Employment Duration  
- Loan Amount, Loan Purpose, and other features  

Several machine learning models were tested, and **CatBoostClassifier** was chosen for its strong performance.  
The model’s results are evaluated with **confusion matrix** and **classification metrics** (precision, recall, f1-score).

## Process
- Data preprocessing (duplicate removal, feature encoding)
- Exploratory visualizations:
  - Risk vs. Sex
  - Risk vs. Credit History
  - Risk vs. Employment Duration
- CatBoost classification model with configurable threshold
- Confusion matrix and classification report (precision/recall/f1)
- Automated tests with `pytest`
- Dockerized for easy deployment

## Quick Start
```bash
git clone https://github.com/Hanhan2024hec/Streamlit-Project.git\
cd homework
pip install -r requirements.txt
streamlit run app.py
# Open the browser using http://localhost:8600/
```

## Tests
```bash
pytest -q
# See the CI opeartion : .github/workflows/ci.yml
```

## Docker
1) Build image
docker build -t wanghan25/german-credit-app .
2) Docker Run
docker run --rm -p 8600:8501 wanghan25/german-credit-app
3) Docker Hub
docker push wanghan25/german-credit-app
4) Link for image
http://localhost:8600/

