import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingredients import ingredient_family, ingredient_tokens, normalise_ingredient  # noqa: E402
from recommender import base_name, parse_ingredients, recommend  # noqa: E402


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
    tokens = ingredient_tokens(['a', 'b', 'c', 'd', 'e', 'f'], use_families=False)
    assert tokens.count('a') == 5
    assert tokens.count('b') == 4
    assert tokens.count('f') == 1


def test_water_is_dropped_and_ranks_start_after_it():
    tokens = ingredient_tokens(['Aqua/Water', 'Glycerine', 'niacinamide'], use_families=False)
    assert 'aqua/water' not in tokens
    assert tokens.count('glycerin') == 5


def test_families_group_related_ingredients():
    assert ingredient_family('sodium hyaluronate') == ingredient_family('hyaluronic acid') == 'hyaluronic acid'
    assert ingredient_family('tocopheryl acetate') == 'vitamin e'
    assert ingredient_family('citric acid') is None
    assert ingredient_family('ethylhexyl salicylate') is None
    assert normalise_ingredient('Aqua (Water)') == ''
    assert normalise_ingredient('Glycerine*') == 'glycerin'


def test_families_give_partial_credit():
    df = pd.DataFrame({
        'product_name': ['x', 'related', 'unrelated'],
        'brand': ['A', 'B', 'C'],
        'ingredients': [['sodium hyaluronate', 'glycerin'], ['hyaluronic acid', 'butylene glycol'], ['shea butter', 'cocoa butter']],
        'display_price': [10.0, 10.0, 10.0],
        'product_rating': [4.0, 4.0, 4.0],
    })
    from recommender import ingredient_similarity
    assert ingredient_similarity(df, 0, use_families=False)[1] == 0
    assert ingredient_similarity(df, 0, use_families=True)[1] > 0


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


def test_base_name_strips_sizes():
    assert base_name('CeraVe Moisturising Cream 454g') == base_name('CeraVe Moisturising Cream 50ml')
    assert base_name('Caudalie Supersize Grape Water (200ml)') == base_name('Caudalie Grape Water 75ml')


def test_size_variants_are_not_recommended():
    df = make_df()
    df.loc[1, 'product_name'] = 'Base 200ml'
    df.loc[0, 'product_name'] = 'Base 50ml'
    df.loc[2, 'product_name'] = 'Cheap twin 30ml'
    df = pd.concat([df, df.iloc[[2]].assign(product_name='Cheap twin 100ml', display_price=15.0)],
                   ignore_index=True)
    names = recommend(df, 0)['product_name'].tolist()
    assert 'Base 200ml' not in names
    assert sum(n.startswith('Cheap twin') for n in names) == 1


def test_same_brand_identical_formula_counts_as_another_size():
    df = pd.DataFrame({
        'product_name': ['Cream Tube 50ml', 'Cream Pot with Ceramides 454g', 'Other Brand Cream', 'Lotion'],
        'brand': ['A', 'A', 'B', 'A'],
        'ingredients': [['glycerin', 'ceramide np', 'cetearyl alcohol']] * 3 + [['glycerin', 'ceramide np', 'squalane']],
        'display_price': [7.0, 15.5, 9.0, 12.0],
        'product_rating': [4.6, 4.7, 4.1, 4.2],
    })
    names = recommend(df, 0)['product_name'].tolist()
    assert 'Cream Pot with Ceramides 454g' not in names  # same brand, same formula
    assert 'Other Brand Cream' in names                  # same formula, other brand: a dupe
    assert 'Lotion' in names
