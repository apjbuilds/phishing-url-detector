#training
import pickle
import re

import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "url_length", "num_dots", "num_hyphens", "num_digits",
    "num_special_chars", "has_https", "has_ip", "brand_similarity",
    "NoOfExternalRef", "LineOfCode", "HasSocialNet", "IsResponsive",
    "NoOfiFrame",
]

# URL-only subset - used as a fallback when a page can't be fetched
URL_ONLY_FEATURE_COLUMNS = [
    "url_length", "num_dots", "num_hyphens", "num_digits",
    "num_special_chars", "has_https", "has_ip", "brand_similarity",
]

COMMON_BRANDS = [
    "amazon", "paypal", "metamask", "microsoft", "apple", "google",
    "facebook", "netflix", "bank", "chase", "wellsfargo", "coinbase",
    "binance", "instagram", "linkedin", "ebay", "walmart",
]


def brand_similarity(domain: str) -> float:
    domain_clean = domain.lower().replace("www.", "").split(".")[0]
    return max(SequenceMatcher(None, domain_clean, brand).ratio() for brand in COMMON_BRANDS)


def main() -> None:
    df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")
    patch_df = pd.read_csv("legit_domains_patch.csv")

    df["url_length"] = df["URL"].str.len()
    df["num_dots"] = df["URL"].str.count(r"\.")
    df["num_hyphens"] = df["URL"].str.count("-")
    df["num_digits"] = df["URL"].apply(lambda value: sum(char.isdigit() for char in value))
    df["num_special_chars"] = df["URL"].str.count(r"[@%&=?]")
    df["has_https"] = df["URL"].str.startswith("https://").astype(int)
    df["has_ip"] = df["URL"].str.contains(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}").astype(int)
    df["brand_similarity"] = df["Domain"].apply(brand_similarity)

    original_subset = df[FEATURE_COLUMNS + ["label"]].copy()
    patch_subset = patch_df[FEATURE_COLUMNS + ["label"]].copy()
    combined_df = pd.concat([original_subset, patch_subset], ignore_index=True)

    weights = np.ones(len(combined_df))
    weights[len(original_subset):] = 100

    # Main model - all 13 features
    x_train, _, y_train, _, weights_train, _ = train_test_split(
        combined_df[FEATURE_COLUMNS], combined_df["label"], weights,
        test_size=0.2, random_state=42,
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train, sample_weight=weights_train)

    # URL-only fallback model - used when a page fetch fails, so the app can
    # still give a real answer instead of just giving up
    x_train_url, _, y_train_url, _, weights_train_url, _ = train_test_split(
        combined_df[URL_ONLY_FEATURE_COLUMNS], combined_df["label"], weights,
        test_size=0.2, random_state=42,
    )
    url_only_model = RandomForestClassifier(n_estimators=100, random_state=42)
    url_only_model.fit(x_train_url, y_train_url, sample_weight=weights_train_url)

    with open("phishing_model.pkl", "wb") as file:
        pickle.dump({
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "url_only_model": url_only_model,
            "url_only_feature_columns": URL_ONLY_FEATURE_COLUMNS,
        }, file)

    print(f"Saved phishing_model.pkl using {len(original_subset):,} original rows and {len(patch_subset):,} verified domains.")
    print("Includes both the main 13-feature model and a URL-only fallback model.")


if __name__ == "__main__":
    main()
