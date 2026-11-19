"""Data quality checks. Errors stop the catalogue from being saved. Warnings are reported."""
import ast
from dataclasses import dataclass

import pandas as pd

from pipeline import config
from pipeline.merge import COLUMNS


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    severity: str = 'error'  # 'error' blocks the save, 'warning' does not


def _ingredient_count(value):
    try:
        return len(ast.literal_eval(value))
    except (ValueError, SyntaxError):
        return 0


def run_checks(new, old=None, attempted=0, failed=0):
    checks = []
    add = lambda *a, **k: checks.append(Check(*a, **k))  # noqa: E731

    missing_cols = [c for c in COLUMNS if c not in new.columns]
    add('All columns present', not missing_cols, f'missing: {missing_cols}' if missing_cols else 'ok')
    if missing_cols:
        return checks

    dupes = new.duplicated(['product_id', 'product_name']).sum()
    add('No duplicate products', dupes == 0, f'{dupes} rows share an ID and name')

    for col in ['product_id', 'product_name', 'product_url', 'product_type']:
        nulls = new[col].isna().sum()
        add(f'No missing {col}', nulls == 0, f'{nulls} missing')

    bad_types = sorted(set(new['product_type'].dropna()) - config.PRODUCT_TYPES)
    add('Product types are known', not bad_types, f'unknown: {bad_types}' if bad_types else 'ok')

    prices = pd.to_numeric(new['updated_price'], errors='coerce').dropna()
    bad_prices = ((prices <= 0) | (prices > config.MAX_PRICE_GBP)).sum()
    add('Prices are between £0 and £1,000', bad_prices == 0, f'{bad_prices} out of range')

    ratings = pd.to_numeric(new['product_rating'], errors='coerce').dropna()
    bad_ratings = ((ratings < 0) | (ratings > 5)).sum()
    add('Ratings are between 0 and 5', bad_ratings == 0, f'{bad_ratings} out of range')

    available = new[new['available'].astype(str).str.lower() == 'true']
    counts = available['ingredients'].map(_ingredient_count)
    coverage = (counts > 0).mean() if len(available) else 0
    add('Available products have ingredients', coverage >= config.MIN_INGREDIENT_COVERAGE,
        f'{coverage:.1%} have ingredients (minimum {config.MIN_INGREDIENT_COVERAGE:.0%})')

    if old is not None:
        add('No products were lost', len(new) >= len(old), f'{len(old)} before, {len(new)} after')

    if attempted:
        rate = failed / attempted
        add('Most pages loaded', rate <= config.MAX_FETCH_FAILURE_RATE,
            f'{failed} of {attempted} requests failed ({rate:.0%})')

    # Warnings
    if old is not None:
        # Compare by product ID. Skip IDs shared by several sizes, since names can change.
        def single(df):
            df = df[~df['product_id'].duplicated(keep=False)]
            return pd.to_numeric(df.set_index('product_id')['updated_price'], errors='coerce')
        both = pd.concat([single(old), single(new)], axis=1, keys=['before', 'after'], join='inner').dropna()
        ratio = both['after'] / both['before']
        jumps = both[(ratio > 3) | (ratio < 1 / 3)]
        examples = ', '.join(f'{pid} £{b:.2f} to £{a:.2f}' for pid, b, a in jumps.head(3).itertuples(name=None))
        add('Prices did not jump', jumps.empty,
            f'{len(jumps)} prices changed more than 3x' + (f' (e.g. {examples})' if examples else ''), 'warning')
    short = ((counts > 0) & (counts < 3)).sum()
    add('Ingredient lists look complete', short == 0, f'{short} lists with fewer than 3 items', 'warning')
    no_brand = available['brand'].isna().mean() if len(available) else 0
    add('Brands are filled in', no_brand < 0.05, f'{no_brand:.1%} of available products have no brand', 'warning')
    old_source = (available['ingredients_source'] != 'lookfantastic').mean() if len(available) else 0
    add('Ingredients come from one source', old_source == 0,
        f'{old_source:.1%} of available products still use 2021 Kaggle ingredients', 'warning')
    return checks


def report_markdown(checks, summary):
    lines = ['# Data quality report', '']
    lines += [f'- {k}: {v}' for k, v in summary.items()]
    lines += ['', '| Check | Result | Detail |', '|---|---|---|']
    for c in checks:
        status = 'Pass' if c.passed else ('Fail' if c.severity == 'error' else 'Warning')
        lines.append(f'| {c.name} | {status} | {c.detail} |')
    return '\n'.join(lines) + '\n'
