"""Clean the saved catalogue without scraping again.

Fixes product names and ingredients damaged by text encoding, tidies ingredient lists
and fills missing brands from data/brands.csv. Safe to run more than once.

    python -m pipeline.repair
"""
import ast
from datetime import date
from pathlib import Path

import pandas as pd

from pipeline import config
from pipeline.brands import infer_brand
from pipeline.merge import COLUMNS
from pipeline.checks import report_markdown, run_checks
from pipeline.parse import clean_ingredient_list, fix_mojibake, is_bundle, product_id_from_url


def restore_redirected(df, history_path='data/skincare_products_2026.csv'):
    """Rows whose link now points at a category page are discontinued products.

    The first scrape overwrote their details with the category page's. Put the 2026
    details back and mark them unavailable.
    """
    df = df.copy()
    redirected = df['product_url'].astype(str).str.contains('/c/') & ~df['product_url'].astype(str).str.contains('/p/')
    if not redirected.any():
        return df, 0
    old = pd.read_csv(history_path)
    old['product_id'] = old['product_url'].map(product_id_from_url)
    old['_n'] = old.groupby('product_id').cumcount()
    df['_n'] = df.groupby('product_id').cumcount()
    cols = ['product_name', 'product_url', 'brand', 'updated_price', 'product_rating', 'product_image_url']
    lookup = old.set_index(['product_id', '_n'])[cols]
    for idx in df.index[redirected]:
        key = (df.at[idx, 'product_id'], df.at[idx, '_n'])
        if key in lookup.index:
            for col in cols:
                df.at[idx, col] = lookup.at[key, col]
        df.at[idx, 'available'] = False
    return df.drop(columns='_n'), int(redirected.sum())


def repair(df):
    df, restored = restore_redirected(df)
    print(f"Discontinued products restored from their category-page redirect: {restored}")
    for col in ['product_name', 'brand']:
        df[col] = df[col].map(fix_mojibake)
    df['ingredients'] = df['ingredients'].map(lambda v: str(clean_ingredient_list(ast.literal_eval(v))))
    missing = df['brand'].isna() | (df['brand'].astype(str).str.strip() == '')
    df.loc[missing, 'brand'] = df.loc[missing, 'product_name'].map(infer_brand)
    df['is_bundle'] = df['product_name'].map(is_bundle)
    # Sizes that share a page were all given one price in the 2026 refresh. Drop that
    # price so the app falls back to each size's own price.
    shared = df['product_id'].duplicated(keep=False)
    same_price = df[shared].groupby('product_id')['updated_price'].transform('nunique').eq(1)
    sizes = df[shared].groupby('product_id')['product_name'].transform('nunique').gt(1)
    df.loc[same_price[same_price & sizes].index, 'updated_price'] = None
    return df[COLUMNS]


def main():
    before = pd.read_csv(config.CATALOGUE_PATH, dtype={'product_id': str})
    after = repair(before)
    print(f"Names repaired: {(before['product_name'] != after['product_name']).sum()}")
    print(f"Ingredient lists changed: {(before['ingredients'] != after['ingredients']).sum()}")
    print(f"Brands filled: {before['brand'].isna().sum() - after['brand'].isna().sum()}")
    print(f"Gift sets and bundles flagged: {after['is_bundle'].sum()}")
    checks = run_checks(after, before)
    summary = {'Run date': date.today().isoformat(), 'Step': 'repair (no scraping)',
               'Products in catalogue': len(after),
               'Available products': int((after['available'].astype(str).str.lower() == 'true').sum())}
    Path(config.REPORT_PATH).write_text(report_markdown(checks, summary))
    if any(not c.passed and c.severity == 'error' for c in checks):
        print('Checks failed. Catalogue not saved.')
        return 1
    after.to_csv(config.CATALOGUE_PATH, index=False)
    print('Saved.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
