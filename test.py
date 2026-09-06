import requests

test_urls = [
    # legit
    'https://www.google.com',
    'https://www.microsoft.com',
    'https://www.apple.com',
    'https://www.wikipedia.org',
    'https://www.amazon.com',
    'https://github.com',
    'https://www.paypal.com',
    'https://discord.com',
    'https://www.netflix.com',
    'https://www.linkedin.com',
    'https://accounts.google.com',
    'https://login.microsoftonline.com',
    'https://store.steampowered.com',
    'https://www.bankofamerica.com',
    'https://www.nasa.gov'
]

for url in test_urls:
    try:
        response = requests.post(
            'http://127.0.0.1:5000/predict',
            json={'url': url}
        )
        print(url, '->', response.json())

    except Exception as e:
        print(url, '-> ERROR:', e)
