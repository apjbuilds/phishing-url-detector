from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import re
import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from urllib.parse import urlsplit

app = Flask(__name__)
CORS(app)

with open('phishing_model.pkl', 'rb') as f:
    saved_model = pickle.load(f)


if isinstance(saved_model, dict):
    model = saved_model['model']
    feature_order = saved_model['feature_columns']
else:
    model = saved_model
    feature_order = [
        'url_length', 'num_dots', 'num_hyphens', 'num_digits',
        'num_special_chars', 'has_https', 'has_ip', 'brand_similarity',
        'NoOfExternalRef', 'LineOfCode', 'HasSocialNet', 'IsResponsive',
        'NoOfiFrame'
    ]

common_brands = ['amazon', 'paypal', 'metamask', 'microsoft', 'apple', 'google',
                  'facebook', 'netflix', 'bank', 'chase', 'wellsfargo', 'coinbase',
                  'binance', 'instagram', 'linkedin', 'ebay', 'walmart']


@app.route('/')
def index():
    return render_template('index.html')


def brand_similarity(domain):
    domain_clean = domain.lower().replace('www.', '').split('.')[0]
    best_score = 0
    for brand in common_brands:
        score = SequenceMatcher(None, domain_clean, brand).ratio()
        best_score = max(best_score, score)
    return best_score


def normalize_for_prediction(url):
    """Use one consistent main-domain form for equivalent website addresses."""
    parsed = urlsplit(url)
    domain = parsed.hostname
    if not domain:
        return url

    domain = domain.lower()
    if domain.startswith('www.'):
        domain = domain[4:]

    return f'{parsed.scheme}://{domain}'


def extract_url_features(url):
    domain = url.replace('https://', '').replace('http://', '').split('/')[0]
    return {
        'url_length': len(url),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_digits': sum(c.isdigit() for c in url),
        'num_special_chars': len(re.findall(r'[@%&=?]', url)),
        'has_https': 1 if url.startswith('https://') else 0,
        'has_ip': 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0,
        'brand_similarity': brand_similarity(domain)
    }


def extract_page_features(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return None

        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        domain = url.replace('https://', '').replace('http://', '').split('/')[0]
        links = soup.find_all('a', href=True)
        external_links = [link for link in links if link['href'].startswith('http') and domain not in link['href']]

        return {
            'NoOfExternalRef': len(external_links),
            'LineOfCode': len(html.splitlines()),
            'HasSocialNet': int(any(site in html.lower() for site in
                                    ['facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com'])),
            'IsResponsive': 1 if 'viewport' in html.lower() else 0,
            'NoOfiFrame': len(soup.find_all('iframe'))
        }
    except requests.RequestException:
        return None


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url = data.get('url', '')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    if not url.startswith('http'):
        url = 'https://' + url
    url = normalize_for_prediction(url)

    url_features = extract_url_features(url)
    page_features = extract_page_features(url)
    if page_features is None:
        return jsonify({
            'error': 'Could not inspect this website. It may block automated requests or be unavailable.'
        }), 422
    all_features = {**url_features, **page_features}

    feature_values = [[all_features[feature] for feature in feature_order]]

    prediction = model.predict(feature_values)[0]
    probability = model.predict_proba(feature_values)[0]

    result = 'Legitimate' if prediction == 1 else 'Phishing'
    confidence = float(max(probability))

    return jsonify({'prediction': result, 'confidence': round(confidence * 100, 1)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
