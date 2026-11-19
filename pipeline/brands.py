"""Work out a product's brand from its name using data/brands.csv.

Each row is a brand. The optional alias column holds another way product names
start, for example "PIXI" for PIXI Beauty.
"""
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

BRANDS_PATH = Path(__file__).resolve().parents[1] / 'data' / 'brands.csv'


def _key(text):
    text = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode().lower()
    text = text.replace("'", '')
    return ' '.join(re.sub(r'[^a-z0-9+&]', ' ', text).split())


@lru_cache(maxsize=1)
def _brand_keys(path=BRANDS_PATH):
    table = pd.read_csv(path, keep_default_na=False)
    keyed = set()
    for brand, alias in table[['brand', 'alias']].itertuples(index=False):
        for name in (brand, alias):
            if name and _key(name):
                keyed.add((_key(name), brand))
    return sorted(keyed, key=lambda kb: -len(kb[0]))  # longest first, so "Estée Lauder" beats "Estée"


def infer_brand(product_name, path=BRANDS_PATH):
    """Return the longest known brand that the product name starts with, or None."""
    name = _key(product_name) + ' '
    for key, brand in _brand_keys(path):
        if name.startswith(key + ' '):
            return brand
    return None
