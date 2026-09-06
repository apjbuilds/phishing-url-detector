"""
Generates a plain list of well-known legitimate domains for the allowlist fast-path.
Run this once locally: py generate_allowlist.py
Produces allowlist_domains.txt - one domain per line, no www prefix.
"""
from tranco import Tranco

t = Tranco(cache=True, cache_dir='.tranco')
latest_list = t.list()
top_domains = latest_list.top(2000)

# Skip obvious infra/CDN domains - same filter idea as the training data patch
infra_keywords = ['gtld-servers', 'gstatic', 'googleapis', 'amazonaws', 'akamai',
                   'fbcdn', 'googletagmanager', 'googlevideo', 'cloudflare-dns',
                   'edgekey', 'edgesuite', 'cloudfront', 'fastly', 'azureedge',
                   'akadns', 'domaincontrol', 'nsone', 'dnsmadeeasy', 'msedge',
                   'doubleclick', 'adnxs', 'rlcdn']

clean_domains = [d for d in top_domains if not any(kw in d for kw in infra_keywords)]

with open('allowlist_domains.txt', 'w') as f:
    for domain in clean_domains:
        f.write(domain + '\n')

print(f"Saved {len(clean_domains)} domains to allowlist_domains.txt")
