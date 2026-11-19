"""One-off: turn the 2026 refresh into the pipeline's catalogue format.

    python -m pipeline.bootstrap
"""
import pandas as pd

from pipeline import config
from pipeline.merge import COLUMNS
from pipeline.parse import product_id_from_url


def main():
    df = pd.read_csv('data/skincare_products_2026.csv')
    df['product_id'] = df['product_url'].map(product_id_from_url)
    df['ingredients_source'] = 'kaggle_2021'
    df['review_count'] = None
    df['in_stock'] = None
    df['first_seen'] = '2021'
    df['last_checked'] = None
    df[COLUMNS].to_csv(config.CATALOGUE_PATH, index=False)
    print(f'Wrote {len(df)} products to {config.CATALOGUE_PATH}')


if __name__ == '__main__':
    main()
