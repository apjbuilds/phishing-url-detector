# Phishing URL Detector

A Random Forest model that flags phishing URLs. Also a website (see below)

**Live site:** (https://phishing-url-detector-xscr.onrender.com/)

**Dataset:** [PhiUSIIL Phishing URL Dataset](https://www.kaggle.com/datasets/sharmageetika/phishing-url-dataset) (235,795 URLs). Download from Kaggle and put it in the project folder as `PhiUSIIL_Phishing_URL_Dataset.csv` before running `train_model.py`.

## How I built it

Started with 7 features from the URL text (length, digits, hyphens, HTTPS, etc). Used Logistic Regression, Random Forest, and XGBoost. For the final model I used XGBoost. Later added a brand similarity feature (for example amaozn vs. amazon, amaozn gets flagged.) It also looks at page-content features (external links, code length, etc) and hit 99.8% accuracy (got this number from testing it (using rf_model_v3) on ~50k URLs from the dataset). 
_
**Issues: 
**_
Once I put the model into a real web app, ir was flagging Google, Amazon, PayPal, and other big sites as phishing. found out my training data never had real brand domains labeled as legit (brandsimiliarity). Also found my scraper was returning fake blank data when a site blocked it (eg. Amazon )

Fixed by pulling 1,000 real top domains from the Tranco list, ran them through the same code the live app uses, and retrained with extra weight on those examples. Fixed 7 of 8 test cases (good enough for now).

## Known issues

- Some sites block automated requests
- some login pages still get misclassified 

## Files

- `phishing_detector.ipynb` — all the analysis and model training
- `train_model.py` — trains the final model
- `legit_domains_patch.csv` — real domain data used to fix the brand bias
- `app.py` / `templates/` — the actual web app
- `test.py` — script to test the API on a list of URLs
