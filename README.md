# Streamlit-Project

A Streamlit app for exploring and modeling the **German Credit dataset**.  
This project demonstrates **data visualization**, **ML classification with CatBoost**, **CI testing with GitHub Actions**, and **Docker containerization**.

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

