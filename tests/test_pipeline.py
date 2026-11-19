import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import run as pipeline_run  # noqa: E402
from pipeline.checks import run_checks  # noqa: E402
from pipeline.merge import COLUMNS, apply_results  # noqa: E402
from pipeline.parse import (extract_product_links, page_count, parse_product_page,  # noqa: E402
                            product_id_from_url, split_ingredients)

FIXTURES = Path(__file__).parent / 'fixtures'
BASE = 'https://www.lookfantastic.com'


def fixture(name):
    return (FIXTURES / name).read_text()


# Parsing

def test_product_id_from_old_and_new_urls():
    assert product_id_from_url('https://www.lookfantastic.com/cerave-lotion-52ml/11798689.html') == '11798689'
    assert product_id_from_url('https://www.lookfantastic.com/p/cerave-lotion-52ml/11798689/') == '11798689'


def test_category_links_and_page_count():
    html = fixture('category_page.html')
    links = extract_product_links(html, BASE)
    assert set(links) == {'17879631', '17355606'}
    assert links['17879631'] == f'{BASE}/p/bobbi-brown-vitamin-enriched-face-base-50ml/17879631/'
    assert page_count(html) == 39


def test_parse_product_page_with_structured_data():
    p = parse_product_page(fixture('product_page.html'))
    assert p['product_name'] == 'CeraVe AM Facial Moisturising Lotion SPF25 52ml'
    assert p['brand'] == 'CeraVe'
    assert p['updated_price'] == 16.5
    assert p['product_rating'] == 4.62
    assert p['review_count'] == 318
    assert p['in_stock'] is False
    assert p['product_image_url'].endswith('large.jpg')
    assert p['canonical_url'].endswith('/11798689/')
    assert p['ingredients'][:3] == ['aqua/water', 'homosalate', 'glycerin']
    assert 'phenoxyethanol (preservative, antimicrobial)' in p['ingredients']
    assert p['ingredients'][-1] == 'ethylhexylglycerin'


def test_parse_product_page_without_structured_data():
    p = parse_product_page(fixture('product_page_no_jsonld.html'))
    assert p['product_name'] == 'Pixi Glow Tonic 250ml'
    assert p['updated_price'] == 18.0
    assert p['ingredients'] == ['aqua (water)', 'glycolic acid', 'aloe barbadensis leaf juice', 'glycerin', 'panthenol']


def test_split_ingredients_keeps_brackets_together():
    assert split_ingredients('A, B (C, D), E') == ['a', 'b (c, d)', 'e']


def test_split_ingredients_drops_footnotes():
    assert split_ingredients('Glycerin, Moringa Oleifera Seed Oil. *From organic farming') == ['glycerin', 'moringa oleifera seed oil']


def test_checks_flag_price_jumps():
    new = catalogue()
    new.loc[0, 'updated_price'] = 99.0
    warnings = {c.name for c in run_checks(new, catalogue()) if not c.passed}
    assert 'Prices did not jump' in warnings


# Merging

def catalogue():
    rows = [
        dict(product_id='1', product_name='Cream 50ml', product_type='Moisturiser', updated_price=10.0, available=True),
        dict(product_id='2', product_name='Serum 30ml', product_type='Serum', updated_price=20.0, available=True),
        dict(product_id='3', product_name='Toner 200ml', product_type='Toner', updated_price=8.0, available=True),
        dict(product_id='3', product_name='Toner 400ml', product_type='Toner', updated_price=14.0, available=True),
    ]
    df = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in df:
            df[col] = None
    df['product_url'] = 'https://example.com/' + df['product_id']
    df['ingredients'] = "['glycerin', 'niacinamide', 'panthenol']"
    df['ingredients_source'] = 'kaggle_2021'
    df['price'] = df['updated_price']
    return df[COLUMNS]


def page(**kw):
    base = dict(product_name='Name', brand='Brand', updated_price=12.0, product_rating=4.5,
                ingredients=['squalane', 'glycerin', 'ceramide np'])
    base.update(kw)
    return base


def test_merge_updates_marks_missing_and_adds_new():
    results = [
        {'product_id': '1', 'url': 'u1', 'product_type': 'Moisturiser', 'status': 200,
         'fields': page(product_name='Cream 50ml', updated_price=11.0)},
        {'product_id': '2', 'url': 'u2', 'product_type': 'Serum', 'status': 404, 'fields': {}},
        {'product_id': '9', 'url': 'u9', 'product_type': 'Serum', 'status': 200,
         'fields': page(product_name='New Serum 30ml')},
    ]
    out = apply_results(catalogue(), results, today='2026-11-19').set_index('product_name')
    assert out.loc['Cream 50ml', 'updated_price'] == 11.0
    assert out.loc['Cream 50ml', 'ingredients_source'] == 'lookfantastic'
    assert out.loc['Serum 30ml', 'available'] == False  # noqa: E712
    assert out.loc['New Serum 30ml', 'first_seen'] == '2026-11-19'
    assert len(out) == 5


def test_merge_does_not_overwrite_sizes_that_share_a_page():
    results = [{'product_id': '3', 'url': 'u3', 'product_type': 'Toner', 'status': 200,
                'fields': page(product_name='Toner', updated_price=99.0)}]
    out = apply_results(catalogue(), results).set_index('product_name')
    assert out.loc['Toner 200ml', 'updated_price'] == 8.0
    assert out.loc['Toner 400ml', 'updated_price'] == 14.0
    assert out.loc['Toner 400ml', 'ingredients'] == str(['squalane', 'glycerin', 'ceramide np'])


def test_merge_leaves_rows_alone_on_network_error():
    results = [{'product_id': '1', 'url': 'u1', 'product_type': 'Moisturiser', 'status': None, 'fields': {}}]
    before = catalogue()
    after = apply_results(before, results)
    cols = [c for c in COLUMNS if c != 'price_change']
    pd.testing.assert_frame_equal(after[cols], before[cols], check_dtype=False)


def test_merge_follows_products_that_moved_to_a_new_id():
    moved = page(product_name='Cream 50ml', canonical_url='https://www.lookfantastic.com/p/cream-50ml/77777777/')
    results = [
        {'product_id': '77777777', 'url': 'u7', 'product_type': 'Moisturiser', 'status': 200, 'fields': moved},
        {'product_id': '1', 'url': 'u1', 'product_type': 'Moisturiser', 'status': 200, 'fields': moved},
    ]
    out = apply_results(catalogue(), results)
    assert len(out) == 4
    assert out.loc[out['product_name'] == 'Cream 50ml', 'product_id'].item() == '77777777'


# Checks

def failed(checks):
    return {c.name for c in checks if not c.passed and c.severity == 'error'}


def test_checks_pass_on_clean_data():
    assert failed(run_checks(catalogue(), catalogue(), attempted=10, failed=1)) == set()


def test_checks_catch_bad_data():
    bad = catalogue()
    bad.loc[0, 'updated_price'] = -5
    bad.loc[1, 'product_type'] = 'Lipstick'
    bad = pd.concat([bad, bad.iloc[[2]]])
    names = failed(run_checks(bad, catalogue(), attempted=10, failed=5))
    assert {'Prices are between £0 and £1,000', 'Product types are known',
            'No duplicate products', 'Most pages loaded'} <= names


def test_checks_catch_lost_products():
    assert 'No products were lost' in failed(run_checks(catalogue().iloc[:2], catalogue()))


# End to end with a fake website

class FakeFetcher:
    def __init__(self, delay=0):
        self.pages = {
            f'{BASE}/c/health-beauty/face/skincare-products/moisturisers/': fixture('category_page.html').replace('Page 1 of 39', 'Page 1 of 1'),
        }

    def get(self, url):
        if url in self.pages:
            return 200, self.pages[url]
        if '17879631' in url:
            return 200, fixture('product_page.html')
        if url.endswith('/2'):
            return 404, ''
        if '/c/' in url:
            return 404, ''
        return 200, fixture('product_page_no_jsonld.html')


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    cat_path, report = tmp_path / 'catalogue.csv', tmp_path / 'report.md'
    catalogue().to_csv(cat_path, index=False)
    monkeypatch.setattr(pipeline_run, 'Fetcher', FakeFetcher)
    code = pipeline_run.main(['--catalogue', str(cat_path), '--report', str(report), '--max-pages', '1'])
    assert code == 0
    out = pd.read_csv(cat_path, dtype={'product_id': str})
    assert '17879631' in set(out['product_id'])
    assert out.loc[out['product_id'] == '2', 'available'].item() == False  # noqa: E712
    assert '# Data quality report' in report.read_text()
