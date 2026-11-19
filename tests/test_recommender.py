import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recommender import parse_ingredients, position_weighted, recommend  # noqa: E402


def make_df():
    return pd.DataFrame({
        'product_name': ['Base', 'Twin', 'Cheap twin', 'Unrelated', 'Same brand'],
        'brand': ['A', 'B', 'C', 'D', 'A'],
        'ingredients': [
            ['niacinamide', 'sodium hyaluronate', 'glycerin'],
            ['niacinamide', 'sodium hyaluronate', 'glycerin'],
            ['niacinamide', 'sodium hyaluronate', 'glycerin'],
            ['shea butter', 'cocoa butter'],
            ['niacinamide', 'glycerin', 'panthenol'],
        ],
        'display_price': [50.0, 50.0, 10.0, 50.0, 45.0],
        'product_rating': [4.5, 4.5, 4.0, 5.0, None],
    })


def test_parse_ingredients_keeps_full_names():
    assert parse_ingredients("['Sodium Hyaluronate', 'glycerin']") == ['sodium hyaluronate', 'glycerin']


def test_position_weighting_favours_top_ingredients():
    tokens = position_weighted(['a', 'b', 'c', 'd', 'e', 'f'])
    assert tokens.count('a') == 5
    assert tokens.count('b') == 4
    assert tokens.count('f') == 1


def test_multi_word_ingredients_are_not_split():
    df = pd.DataFrame({
        'product_name': ['x', 'y'],
        'brand': ['A', 'B'],
        'ingredients': [['sodium hyaluronate'], ['sodium chloride']],
        'display_price': [10.0, 10.0],
        'product_rating': [4.0, 4.0],
    })
    # Sharing the word "sodium" alone must not count as a match
    assert recommend(df, 0).empty


def test_unrelated_products_are_excluded():
    result = recommend(make_df(), 0)
    assert 'Unrelated' not in result['product_name'].tolist()
    assert 'Base' not in result['product_name'].tolist()


def test_cheaper_only_returns_cheaper_products():
    result = recommend(make_df(), 0, cheaper_only=True)
    assert (result['display_price'] < 50.0).all()
    assert result.iloc[0]['product_name'] == 'Cheap twin'


def test_exclude_same_brand():
    result = recommend(make_df(), 0, exclude_same_brand=True)
    assert 'A' not in result['brand'].tolist()
