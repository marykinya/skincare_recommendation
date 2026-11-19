"""Ingredient name cleaning and ingredient families.

Two products can do the same job with differently named ingredients.
"Sodium hyaluronate" and "hyaluronic acid" are both hyaluronic acid.
Families give those products partial credit for matching.
"""
import re

# Water carries no information about a formula. The 2021 Kaggle lists already had it removed.
WATER = {'water', 'aqua', 'eau', 'aqua/water', 'water/aqua', 'aqua/water/eau', 'water/aqua/eau', 'purified water'}

FAMILIES = {
    'hyaluronic acid': r'hyaluron',
    'niacinamide': r'niacinamide|nicotinamide',
    'vitamin c': r'ascorb',
    'vitamin e': r'tocopher',
    'retinoid': r'retin(?:ol|al|yl|oate)\b|retinoid',
    'panthenol': r'panthenol|pantothen',
    'ceramide': r'ceramide',
    'peptide': r'peptide',
    'aha': r'\b(?:glycolic|lactic|mandelic|malic|tartaric) acid\b',
    'bha': r'salicylic acid|betaine salicylate|salix alba',  # not sunscreen filters like ethylhexyl salicylate
    'squalane': r'squal[ae]ne',
    'centella': r'centella|madecass|asiatic',
    'shea butter': r'butyrospermum|\bshea\b',
    'azelaic acid': r'azela',
}
_FAMILY_PATTERNS = [(name, re.compile(pattern)) for name, pattern in FAMILIES.items()]


def normalise_ingredient(raw):
    """Lowercase, tidy and unify spelling. Returns '' for water or junk."""
    name = str(raw).lower().strip()
    name = re.sub(r'[*†‡]+', '', name)
    name = re.sub(r'\s+', ' ', name).strip(' .;:-')
    name = name.replace('glycerine', 'glycerin')
    # "aqua (water)", "aqua/water/eau" and similar
    water_parts = set(re.split(r'[/()\s]+', name)) - {''}
    if not name or name in WATER or water_parts <= {'aqua', 'water', 'eau', 'purified'} or len(name) > 80:
        return ''
    return name


def ingredient_family(name):
    """Return the family an ingredient belongs to, or None."""
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.search(name):
            return family
    return None


def ingredient_tokens(ingredient_list, use_families=True, top_k=5):
    """Turn an ingredient list into weighted tokens for TF-IDF.

    Each ingredient is one token. Ingredients listed first are repeated more often:
    rank 1 five times, rank 2 four times and so on, then once from rank 5.
    When use_families is on, ingredients in a family also add a family token
    with the same weight.
    """
    tokens, rank = [], 0
    for raw in ingredient_list:
        name = normalise_ingredient(raw)
        if not name:
            continue
        rank += 1
        weight = max(1, top_k + 1 - rank)
        tokens.extend([name] * weight)
        if use_families:
            family = ingredient_family(name)
            if family:
                tokens.extend([f'family:{family}'] * weight)
    return tokens
