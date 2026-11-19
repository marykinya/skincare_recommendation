import ast
import html
import re
from pathlib import Path

import pandas as pd
import streamlit as st
import validators

from ingredients import ingredient_tokens

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PLACEHOLDER_IMAGE_URL = 'https://placehold.co/300x300/f9fafb/6b7280?text=No+image'

# Styling
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@300;400;500;600;700&display=swap');

    .stApp { background-color: #ffffff; }

    html, body, [class*="css"] {
        font-family: 'Figtree', sans-serif;
        font-weight: 300;
        color: #000014;
    }

    p, li, span, div {
        font-weight: 300;
    }

    h1 {
        font-family: 'Figtree', sans-serif !important;
        font-weight: 300 !important;
        font-size: 2rem !important;
        color: #000014 !important;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    h2, h3 {
        font-family: 'Figtree', sans-serif !important;
        font-weight: 300 !important;
        color: #000014 !important;
    }

    /* Labels above dropdowns */
    label {
        font-size: 0.75rem !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: #6b7280 !important;
        font-weight: 400 !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #000014;
        color: #ffffff;
        border: none;
        border-radius: 0;
        padding: 0.75rem 2rem;
        font-family: 'Figtree', sans-serif;
        font-size: 0.75rem;
        font-weight: 400;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        width: 100%;
        transition: background-color 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #374151;
        color: #ffffff;
    }

    hr { border: none; border-top: 1px solid #e5e7eb; margin: 1.5rem 0; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #e5e7eb;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Figtree', sans-serif !important;
        font-size: 0.875rem !important;
        font-weight: 300 !important;
        letter-spacing: 0.01em !important;
        color: #6b7280 !important;
        padding: 0.75rem 1.25rem !important;
        border-bottom: 2px solid transparent !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #000014 !important;
        border-bottom: 2px solid #000014 !important;
        font-weight: 400 !important;
    }

    /* Product card */
    .product-card {
        background: #ffffff;
        border-radius: 0;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
        transition: box-shadow 0.2s ease-in-out;
        height: 100%;
    }
    .product-card:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 20, 0.1);
    }
    .product-card img {
        width: 100%;
        max-height: 180px;
        object-fit: contain;
        margin-bottom: 0.75rem;
        background: #f9fafb;
    }
    .product-brand {
        font-size: 0.7rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #6b7280;
        margin: 0 0 0.3rem;
        font-weight: 300;
    }
    .product-title a {
        font-family: 'Figtree', sans-serif;
        font-size: 0.875rem;
        font-weight: 300;
        color: #000014;
        text-decoration: none;
        line-height: 1.4;
    }
    .product-title a:hover { color: #4b5563; text-decoration: underline; }
    .product-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.75rem;
        padding-top: 0.75rem;
        border-top: 1px solid #e5e7eb;
        font-size: 0.8rem;
        font-weight: 300;
        color: #6b7280;
    }
    .product-price {
        font-family: 'Figtree', sans-serif;
        font-size: 0.95rem;
        font-weight: 300;
        color: #000014;
    }

    .product-match {
        font-size: 0.75rem;
        color: #6b7280;
        margin: 0.4rem 0 0;
        font-weight: 300;
    }

    /* Eyebrow label */
    .eyebrow {
        font-size: 0.7rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 0.25rem;
        font-weight: 300;
    }
    </style>
    """, unsafe_allow_html=True)

# Data
DATA_PATH = Path(__file__).parent / 'data' / 'skincare_products.csv'


def parse_ingredients(value):
    """Turn the stored list string into a clean list of lowercase ingredient names."""
    try:
        items = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        items = str(value).strip('[]').replace("'", '').split(',')
    return [str(i).strip().lower() for i in items if str(i).strip()]


@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    if 'available' in df.columns:
        df = df[df['available'].astype(str).str.lower() == 'true'].copy()
    df['ingredients'] = df['ingredients'].apply(parse_ingredients)
    if 'updated_price' in df.columns:
        df['display_price'] = df['updated_price'].fillna(df['price'])
    else:
        df['display_price'] = df['price']
    df['product_image_url'] = df['product_image_url'].apply(
        lambda x: x if pd.notnull(x) and str(x).strip() != '' else PLACEHOLDER_IMAGE_URL
    )
    return df.reset_index(drop=True)


# Helper functions
def validate_image_url(url):
    return url if validators.url(str(url)) else PLACEHOLDER_IMAGE_URL


SIZE_PATTERN = re.compile(
    r'\b\d+(?:[.,]\d+)?\s?(?:ml|g|kg|l|oz|fl\.? ?oz|pieces?|pads?|capsules?|sachets?|masks?|wipes?)\b'
)
PACK_WORDS = re.compile(
    r'\b(supersize|travel size|exclusive|pump|tube|bottle|jar|duo pack|duo|worth £\d+|sleeved version)\b'
)


def base_name(product_name):
    """Strip sizes and pack words so size variants of one product share a name."""
    s = str(product_name).lower()
    s = re.sub(r'\(.*?\)', '', s)
    s = SIZE_PATTERN.sub('', s)
    s = re.sub(r'\b(x|-)\s*\d+\b', '', s)
    s = PACK_WORDS.sub('', s)
    s = re.sub(r'[^a-z0-9+ ]', ' ', s)
    return ' '.join(s.split())


def _identity(tokens):
    # Each full ingredient name is one token, so "sodium hyaluronate" is never split.
    return tokens


# Recommendation logic
WEIGHTS = {'ingredients': 0.75, 'price': 0.15, 'rating': 0.10}


def ingredient_similarity(df, product_index, use_families=True):
    vectors = TfidfVectorizer(analyzer=_identity).fit_transform(
        df['ingredients'].apply(ingredient_tokens, use_families=use_families)
    )
    return cosine_similarity(vectors[product_index], vectors).flatten()


def price_scores(prices, sel_price, cheaper_only=False):
    if pd.isnull(sel_price):
        return pd.Series(0.5, index=prices.index)
    if cheaper_only and sel_price > 0:
        # A bigger saving scores higher
        score = (sel_price - prices) / sel_price
    else:
        # A closer price scores higher
        price_range = prices.max() - prices.min()
        if not price_range > 0:
            return pd.Series(0.5, index=prices.index)
        score = 1 - (prices - sel_price).abs() / price_range
    return score.clip(0, 1).fillna(0.5)


def rating_scores(ratings):
    # Higher rated products score higher. Missing ratings get the median of the set.
    return (ratings.fillna(ratings.median()) / 5).clip(0, 1).fillna(0.5)


def recommend(df, product_index, top_n=5, similarity_threshold=0.1,
              cheaper_only=False, exclude_same_brand=False, weights=WEIGHTS):
    """Return the top products most similar to df.iloc[product_index].

    df should already be filtered to one product type.
    """
    df = df.reset_index(drop=True)

    ingr_sim = ingredient_similarity(df, product_index)
    prices = pd.to_numeric(df['display_price'], errors='coerce')
    sel_price = prices.iloc[product_index]
    price_score = price_scores(prices, sel_price, cheaper_only)
    rating_score = rating_scores(pd.to_numeric(df['product_rating'], errors='coerce'))

    scores = (weights['ingredients'] * ingr_sim
              + weights['price'] * price_score.values
              + weights['rating'] * rating_score.values)

    # Never recommend the selected product or another size of it
    bases = df['product_name'].map(base_name)
    mask = ingr_sim >= similarity_threshold
    mask &= (bases != bases.iloc[product_index]).values
    if cheaper_only and pd.notnull(sel_price):
        mask &= (prices < sel_price).values
    if exclude_same_brand:
        brand = df['brand'].iloc[product_index]
        if pd.notnull(brand) and str(brand).strip():
            mask &= (df['brand'].fillna('').str.lower() != str(brand).lower()).values

    return (df[mask]
            .assign(ingredient_match=ingr_sim[mask], score=scores[mask], _base=bases[mask])
            .sort_values('score', ascending=False)
            .drop_duplicates('_base')  # show each product once, at its best scoring size
            .drop(columns='_base')
            .head(top_n))

# UI components
HOW_IT_WORKS = """
## How it works

Pick a product. The app compares it with every other product of the same type.
Each candidate gets a score from three signals.

### Ingredient match (75%)

Ingredient lists are ordered by concentration. The first ingredient makes up the
biggest share of the formula.

- Each full ingredient name is treated as one unit. "Sodium hyaluronate" is matched
  as a whole. It is never split into "sodium" and "hyaluronate".
- Ingredients near the top of the list count more. The first ingredient counts five
  times and the second four times. From the fifth ingredient on each one counts once.
- TF-IDF gives rare ingredients more weight than common ones like glycerin. Sharing a
  rare active says more about two products than sharing a common base.
- Cosine similarity then turns this into a match score between 0 and 1.
- Water is ignored because nearly every product has it.

**Ingredient families.** Related ingredients get partial credit. Sodium hyaluronate and
hyaluronic acid both count as hyaluronic acid, for example. There are fourteen families
including vitamin C, retinoids, ceramides, peptides, AHAs and BHAs. Exact matches still
score highest.

A product needs a match score of at least 0.1 to appear at all. Price and rating
cannot push an unrelated product into the results.

### Price (15%)

By default the app favours products with a similar price.

Turn on **Only show cheaper alternatives** to find a dupe. The app then shows only
products that cost less. A bigger saving scores higher.

### Rating (10%)

Higher rated products score a little higher. Products with no rating get the median
rating for their product type.

### Same brand filter

Similar formulas often come from the same brand. Turn on **Exclude the same brand**
to see other brands only.

### Final score

Score = 0.75 × ingredient match + 0.15 × price + 0.10 × rating

The top five products are shown. Each card shows its ingredient match.
Other sizes of the product you picked are hidden and each product appears once.

### About the data

The catalogue started from a Kaggle dataset of Lookfantastic skincare products. A data
pipeline now refreshes it from Lookfantastic every week. It updates prices, ratings,
images, availability and ingredient lists and adds new products. Discontinued products
are hidden.
"""


def render_product_card(row):
    img_url = html.escape(validate_image_url(row['product_image_url']), quote=True)
    brand = html.escape(str(row.get('brand') or '')) if pd.notnull(row.get('brand')) else ''
    name = html.escape(str(row['product_name']))
    url = html.escape(str(row['product_url']), quote=True)
    rating = row['product_rating']
    price = row['display_price']
    match = row.get('ingredient_match')

    rating_str = f"★ {rating:.2f}" if pd.notnull(rating) else "No rating"
    price_str = f"£{price:.2f}" if pd.notnull(price) else "—"
    match_str = f"{match:.0%} ingredient match" if pd.notnull(match) else ""

    return f"""
    <div class="product-card">
        <img src="{img_url}" alt="{name}" />
        <p class="product-brand">{brand}</p>
        <div class="product-title"><a href="{url}" target="_blank">{name}</a></div>
        <p class="product-match">{match_str}</p>
        <div class="product-meta">
            <span>{rating_str}</span>
            <span class="product-price">{price_str}</span>
        </div>
    </div>
    """


def display_header():
    st.markdown('<p class="eyebrow">Lookfantastic · Skincare</p>', unsafe_allow_html=True)
    st.title("Skincare Recommendations")
    st.markdown("""
    <p style="font-size:0.95rem; color:#4b5563; line-height:1.6; margin-top:-0.5rem;">
    Discover similar skincare products based on what they are made of. Find an alternative, a budget swap or simply something new to try.
    </p>
    """, unsafe_allow_html=True)


# Main
def main():
    st.set_page_config(
        page_title="Skincare Recommender",
        page_icon="✨",
        layout="wide",
    )
    inject_custom_css()

    df = load_data()
    display_header()

    tab_rec, tab_how = st.tabs(["Recommendations", "How does this work?"])

    with tab_rec:
        col_select, col_preview = st.columns([3, 1])

        with col_select:
            product_types = sorted(df['product_type'].unique().tolist())
            selected_type = st.selectbox("Product type", product_types)

            filtered_df = df[df['product_type'] == selected_type].reset_index(drop=True)
            selected_product = st.selectbox("Product", sorted(filtered_df['product_name'].tolist()))
            product_index = filtered_df[filtered_df['product_name'] == selected_product].index[0]

            col_cheap, col_brand = st.columns(2)
            with col_cheap:
                cheaper_only = st.toggle("Only show cheaper alternatives")
            with col_brand:
                exclude_brand = st.toggle("Exclude the same brand")

        with col_preview:
            img_url = validate_image_url(filtered_df.loc[product_index, 'product_image_url'])
            st.image(img_url, width=160)
            price = filtered_df.loc[product_index, 'display_price']
            rating = filtered_df.loc[product_index, 'product_rating']
            parts = []
            if pd.notnull(rating):
                parts.append(f"★ {rating:.2f}")
            if pd.notnull(price):
                parts.append(f"£{price:.2f}")
            if parts:
                st.caption("  ·  ".join(parts))

        st.markdown("<hr>", unsafe_allow_html=True)
        recommended = recommend(
            filtered_df, product_index,
            cheaper_only=cheaper_only, exclude_same_brand=exclude_brand,
        )

        if not recommended.empty:
            label = "Cheaper alternatives" if cheaper_only else "You might also like"
            st.markdown(f'<p class="eyebrow">{label}</p>', unsafe_allow_html=True)
            st.markdown(f"### Similar to *{selected_product}*")

            cols = st.columns(3)
            for i, (_, row) in enumerate(recommended.iterrows()):
                with cols[i % 3]:
                    st.markdown(render_product_card(row), unsafe_allow_html=True)
        else:
            st.info("No similar products found. Try another product or turn off a filter.")

    with tab_how:
        st.markdown(HOW_IT_WORKS)


if __name__ == "__main__":
    main()
