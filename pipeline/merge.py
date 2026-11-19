"""Combine freshly scraped results with the existing catalogue."""
from datetime import date

from pipeline.parse import product_id_from_url

import pandas as pd

COLUMNS = [
    'product_id', 'product_name', 'product_url', 'product_type', 'brand', 'ingredients',
    'ingredients_source', 'price', 'updated_price', 'price_change', 'product_rating',
    'review_count', 'product_image_url', 'available', 'in_stock', 'first_seen', 'last_checked',
]


def _as_list_string(ingredients):
    return str(list(ingredients))


def apply_results(catalogue, results, today=None):
    """Update the catalogue with scrape results.

    results is a list of dicts with keys: product_id, url, product_type, status, fields.
    Existing rows are never deleted. A 404 or 410 marks a product unavailable.
    Network errors leave the row as it was.
    """
    today = today or date.today().isoformat()
    cat = catalogue.copy().reset_index(drop=True)
    text_cols = ['product_name', 'product_url', 'brand', 'ingredients', 'ingredients_source',
                 'product_image_url', 'available', 'in_stock', 'first_seen', 'last_checked']
    cat[text_cols] = cat[text_cols].astype(object)
    known = set(cat['product_id'])
    new_rows = []

    # Handle products already in the catalogue first, so a product that moved to a new ID
    # is re-keyed before the same new ID could be added again as a "new" product.
    results = sorted(results, key=lambda res: res['product_id'] not in known)

    for res in results:
        pid, status, fields = res['product_id'], res['status'], res.get('fields') or {}
        rows = cat.index[cat['product_id'] == pid]

        if status in (404, 410):
            cat.loc[rows, ['available', 'last_checked']] = [False, today]
            continue
        if status != 200:
            continue

        # Lookfantastic sometimes moves a product to a new ID. Follow it.
        moved_to = product_id_from_url(fields.get('canonical_url') or '')
        if len(rows) and moved_to and moved_to != pid:
            if moved_to in known:
                continue  # the new listing is handled under its own ID
            cat.loc[rows, 'product_id'] = moved_to
            known.add(moved_to)
            pid = moved_to
        elif pid not in known and moved_to and moved_to in known:
            continue  # a discovered link that turns out to be a product we already have

        if pid not in known:
            if not fields.get('product_name') or not fields.get('ingredients'):
                continue  # not enough to recommend from
            known.add(pid)
            new_rows.append({
                'product_id': pid,
                'product_name': fields['product_name'],
                'product_url': fields.get('canonical_url') or res['url'],
                'product_type': res['product_type'],
                'brand': fields.get('brand'),
                'ingredients': _as_list_string(fields['ingredients']),
                'ingredients_source': 'lookfantastic',
                'price': fields.get('updated_price'),
                'updated_price': fields.get('updated_price'),
                'price_change': None,
                'product_rating': fields.get('product_rating'),
                'review_count': fields.get('review_count'),
                'product_image_url': fields.get('product_image_url'),
                'available': True,
                'in_stock': fields.get('in_stock'),
                'first_seen': today,
                'last_checked': today,
            })
            continue

        # Some sizes share one product page. Then only update fields that are the same for every size.
        shared_page = len(rows) > 1
        cat.loc[rows, ['available', 'last_checked']] = [True, today]
        per_size = ['product_name', 'updated_price', 'in_stock']
        for col in ['brand', 'product_rating', 'review_count', 'product_image_url'] + ([] if shared_page else per_size):
            if fields.get(col) is not None:
                cat.loc[rows, col] = fields[col]
        if fields.get('canonical_url'):
            cat.loc[rows, 'product_url'] = fields['canonical_url']
        if fields.get('ingredients'):
            cat.loc[rows, 'ingredients'] = _as_list_string(fields['ingredients'])
            cat.loc[rows, 'ingredients_source'] = 'lookfantastic'

    out = pd.concat([cat, pd.DataFrame(new_rows)], ignore_index=True)
    out['price_change'] = pd.to_numeric(out['updated_price'], errors='coerce') - pd.to_numeric(out['price'], errors='coerce')
    return out[COLUMNS]
