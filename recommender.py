import pandas as pd
import streamlit as st
import validators

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PLACEHOLDER_IMAGE_URL = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR5gw3cdavqQK__0yvmmnfE5kdQAHB8PxMYhQ&s'

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
@st.cache_data
def load_data():
    df = pd.read_csv('skincare_products_2026.csv')
    if 'available' in df.columns:
        df = df[df['available'].astype(str).str.lower() == 'true'].copy()
    df['ingredients'] = df['ingredients'].apply(
        lambda x: x.strip("[]").replace("'", "").split(", ")
    )
    if 'updated_price' in df.columns:
        df['display_price'] = df['updated_price'].fillna(df['price'])
    else:
        df['display_price'] = df['price']
    df['product_image_url'] = df['product_image_url'].apply(
        lambda x: x if pd.notnull(x) and str(x).strip() != '' else PLACEHOLDER_IMAGE_URL
    )
    return df

# Helper Functions 
def validate_image_url(url):
    return url if validators.url(url) else PLACEHOLDER_IMAGE_URL

def _weighted_ingredient_str(ingredient_list):
    parts = []
    for rank, ingr in enumerate(ingredient_list, start=1):
        parts.extend([str(ingr).strip()] * max(1, 6 - rank))
    return ' '.join(parts)

# Recommendation logic
def calculate_cosine_similarity(df, product_index, top_n=5, similarity_threshold=0.1):
    df = df.reset_index(drop=True)

    vectors  = TfidfVectorizer().fit_transform(df['ingredients'].apply(_weighted_ingredient_str))
    ingr_sims = cosine_similarity(vectors[product_index], vectors).flatten()

    prices    = pd.to_numeric(df['display_price'], errors='coerce')
    sel_price = prices.iloc[product_index]
    price_range = prices.max() - prices.min()
    if pd.notnull(sel_price) and price_range > 0:
        price_close = (1 - (prices - sel_price).abs() / price_range).clip(0, 1).fillna(0.5)
    else:
        price_close = pd.Series(0.5, index=df.index)

    ratings    = pd.to_numeric(df['product_rating'], errors='coerce')
    sel_rating = ratings.iloc[product_index]
    if pd.notnull(sel_rating):
        rating_close = (1 - (ratings - sel_rating).abs() / 5.0).clip(0, 1).fillna(0.5)
    else:
        rating_close = pd.Series(0.5, index=df.index)

    scores = 0.75 * ingr_sims + 0.15 * price_close.values + 0.10 * rating_close.values

    recommended = []
    for idx in scores.argsort()[::-1]:
        if len(recommended) >= top_n:
            break
        if idx != product_index and ingr_sims[idx] >= similarity_threshold:
            recommended.append(df.iloc[idx])

    return pd.DataFrame(recommended)

# UI components
def render_product_card(row):
    img_url  = validate_image_url(row['product_image_url'])
    brand    = row.get('brand', '') or ''
    name     = row['product_name']
    url      = row['product_url']
    rating   = row['product_rating']
    price    = row['display_price']

    rating_str = f"★ {rating:.2f}" if pd.notnull(rating) else "★ —"
    price_str  = f"£{price:.2f}"   if pd.notnull(price)  else "—"

    return f"""
    <div class="product-card">
        <img src="{img_url}" alt="{name}" />
        <p class="product-brand">{brand}</p>
        <div class="product-title"><a href="{url}" target="_blank">{name}</a></div>
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
        # Product selection
        col_select, col_preview = st.columns([3, 1])

        with col_select:
            product_types = sorted(df['product_type'].unique().tolist())
            selected_type = st.selectbox("Product type", product_types)

            filtered_df = df[df['product_type'] == selected_type].reset_index(drop=True)
            selected_product = st.selectbox("Product", filtered_df['product_name'].tolist())
            product_index = filtered_df[filtered_df['product_name'] == selected_product].index[0]

        with col_preview:
            img_url = validate_image_url(filtered_df.loc[product_index, 'product_image_url'])
            st.image(img_url, width=160)
            price  = filtered_df.loc[product_index, 'display_price']
            rating = filtered_df.loc[product_index, 'product_rating']
            if pd.notnull(rating) and pd.notnull(price):
                st.caption(f"★ {rating:.2f}  ·  £{price:.2f}")

        st.markdown("<hr>", unsafe_allow_html=True)
        recommended = calculate_cosine_similarity(filtered_df, product_index)

        if not recommended.empty:
            st.markdown('<p class="eyebrow">You might also like</p>', unsafe_allow_html=True)
            st.markdown(f"### Similar to *{selected_product}*")

            cols = st.columns(3)
            for i, (_, row) in enumerate(recommended.iterrows()):
                with cols[i % 3]:
                    st.markdown(render_product_card(row), unsafe_allow_html=True)
        else:
            st.info("No similar products found. Try a different product or type.")

    with tab_how:
        st.markdown("""
        ## Finding Your Perfect Match

        When you select a product, every other product of the same category is scored
        against your selection using a weighted combination of three signals. Here's a
        deep dive into each one.

        ### 🧪 Ingredient Match (75%)

        This is the heart of the engine. Skincare products list their ingredients in
        **descending order of concentration**, the first ingredient makes up the largest
        share of the formula, the last the smallest. The recommender takes full advantage
        of this structure.

        **How it works:**
        - Each ingredient is assigned a **positional weight**: ingredients near the top of
          the list carry much more influence than those near the bottom.
        - The app then computes a **weighted cosine similarity** between two ingredient
          lists, meaning products that share the same key actives *and* the same base
          ingredients will score highly together.
        - Rare or specialty ingredients (e.g. niacinamide, retinol, peptides) that appear
          near the top are treated as strong signals of formulation intent and boost
          similarity more than common fillers like water or glycerin.

        **Why it matters:**
        Two moisturisers might both be called "hydrating" on their labels, but one might
        be built around hyaluronic acid while the other leans on shea butter. Ingredient
        matching surfaces the one that is chemically closest to what you already like.

        ### 💰 Price (15%)

        Once ingredient similarity is established, price acts as a **tiebreaker and
        alternative-finder**.

        **How it works:**
        - The absolute price difference between your chosen product and every candidate is
          calculated and normalised across the full price range in the dataset.
        - A candidate that costs nearly the same scores close to 1.0 on this signal; one
          that is drastically cheaper or more expensive scores lower.
        - The 15% weight means price can meaningfully separate two otherwise identical
          formulas, but it cannot rescue a product with poor ingredient overlap.

        **Why it matters:**
        This signal powers the "dupe" use case. If you love a luxury serum but want a
        budget alternative, the recommender will surface the closest-matching formula at
        the lowest price difference — not just the cheapest product in the category.

        ### ⭐ Rating (10%)

        A light quality signal drawn from aggregated customer reviews.

        **How it works:**
        - Ratings are normalised to a 0–1 scale across the dataset.
        - Products with a rating close to your chosen product's rating receive a small
          boost; a highly-rated recommendation is preferred over a poorly-rated one when
          everything else is equal.
        - The 10% weight keeps this signal advisory rather than dominant, a 5-star
          product with unrelated ingredients will never outrank a 3-star product with an
          almost identical formula.

        **Why it matters:**
        Real-world performance data from thousands of reviewers can catch things ingredient
        lists can't ; texture, scent, packaging, and skin-feel. The rating nudge helps
        surface products that work well in practice, not just on paper.

        ### 🔬 The Similarity Threshold

        Not every product makes it into your results. A candidate must clear a **minimum
        ingredient similarity score** to appear at all. This threshold filters out products
        that are categorically different in formulation, so price and rating alone can
        never surface an unrelated result.

        Products that pass the threshold are then ranked by their combined weighted score
        (ingredient × 0.75 + price × 0.15 + rating × 0.10) and the top matches are
        displayed.

        ### 📊 About the Data

        - The product catalogue was initially sourced from a publicly available skincare dataset on
        [Kaggle](https://www.kaggle.com) and includes ingredients, prices, and ratings across
        moisturisers, serums, cleansers, sunscreens, toners, eye creams, and more. 
        - The dataset has been periodically refreshed/scraped from [lookfantastic.com](https://www.lookfantastic.com) to reflect current product availability.
        """)

if __name__ == "__main__":
    main()