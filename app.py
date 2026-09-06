from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import re
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright

app = Flask(__name__)
CORS(app)

with open('phishing_model.pkl', 'rb') as f:
    saved_model = pickle.load(f)

if isinstance(saved_model, dict):
    model = saved_model['model']
    feature_order = saved_model['feature_columns']
    url_only_model = saved_model.get('url_only_model')
    url_only_feature_order = saved_model.get('url_only_feature_columns')
else:
    model = saved_model
    feature_order = [
        'url_length', 'num_dots', 'num_hyphens', 'num_digits',
        'num_special_chars', 'has_https', 'has_ip', 'brand_similarity',
        'NoOfExternalRef', 'LineOfCode', 'HasSocialNet', 'IsResponsive',
        'NoOfiFrame'
    ]
    url_only_model = None
    url_only_feature_order = None

# Known-legitimate domains - checked before running the ML model at all.
# Real phishing detectors (Google Safe Browsing, etc.) use allowlists like this
# for well-known sites rather than relying purely on a live classifier.
try:
    with open('allowlist_domains.txt', 'r') as f:
        ALLOWLIST = set(line.strip().lower() for line in f if line.strip())
except FileNotFoundError:
    ALLOWLIST = set()

common_brands = ['amazon', 'paypal', 'metamask', 'microsoft', 'apple', 'google',
                  'facebook', 'netflix', 'bank', 'chase', 'wellsfargo', 'coinbase',
                  'binance', 'instagram', 'linkedin', 'ebay', 'walmart']

# One shared headless browser for the life of the app, instead of launching a
# new one per request - much faster for a live web app handling requests one at a time.
_playwright = sync_playwright().start()
_browser = _playwright.chromium.launch()


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


def get_base_domain(url):
    parsed = urlsplit(url)
    domain = (parsed.hostname or '').lower()
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain


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
    """Fetches the page with a real headless browser so JavaScript-heavy sites
    render fully before we read the HTML, instead of seeing a loading skeleton."""
    try:
        page = _browser.new_page()
        try:
            page.goto(url, timeout=10000, wait_until='networkidle')
            html = page.content()
        finally:
            page.close()

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
    except Exception:
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

    # Fast path: skip the model entirely for well-known domains
    base_domain = get_base_domain(url)
    if base_domain in ALLOWLIST:
        return jsonify({
            'prediction': 'Legitimate',
            'confidence': 99.0,
            'method': 'allowlist'
        })

    url_features = extract_url_features(url)
    page_features = extract_page_features(url)

    if page_features is None:
        # Fall back to a URL-only prediction instead of giving up entirely
        if url_only_model is None:
            return jsonify({
                'error': 'Could not inspect this website. It may block automated requests or be unavailable.'
            }), 422

        feature_values = [[url_features[feature] for feature in url_only_feature_order]]
        prediction = url_only_model.predict(feature_values)[0]
        probability = url_only_model.predict_proba(feature_values)[0]
        result = 'Legitimate' if prediction == 1 else 'Phishing'
        confidence = float(max(probability))

        return jsonify({
            'prediction': result,
            'confidence': round(confidence * 100, 1),
            'note': 'Could not load the page content - this result is based on the URL only, so it may be less reliable.'
        })

    all_features = {**url_features, **page_features}
    feature_values = [[all_features[feature] for feature in feature_order]]

    prediction = model.predict(feature_values)[0]
    probability = model.predict_proba(feature_values)[0]

    result = 'Legitimate' if prediction == 1 else 'Phishing'
    confidence = float(max(probability))

    return jsonify({'prediction': result, 'confidence': round(confidence * 100, 1)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
