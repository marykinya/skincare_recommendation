# Skincare Product Recommendation System

## Overview

This project is an interactive **skincare product recommendation app** built with **Streamlit**. Given a product selected by the user, it surfaces up to five similar products from the same category using a weighted composite score based on ingredient overlap, price proximity, and customer rating.

Data is scraped from **Lookfantastic** and stored in `skincare_products_2026.csv`.

## Features

- **Ingredient-based similarity** — TF-IDF vectorisation with position weighting gives more importance to ingredients listed earlier (higher concentration), then cosine similarity is computed across all products in the same type.
- **Composite scoring** — The final score blends three signals:
  - **75%** ingredient similarity
  - **15%** price proximity (normalised to the price range of the filtered set)
  - **10%** rating proximity (normalised to a 5-point scale)
- **Availability filtering** — Only products where `available == True` are loaded.
- **Same-type filtering** — Recommendations are drawn exclusively from the same `product_type` as the selected product (e.g., Moisturiser, Cleanser).
- **Similarity threshold** — A product must reach a minimum ingredient similarity score of `0.1` to appear in results regardless of price or rating.
- **Image validation** — Product image URLs are validated with the `validators` library; invalid or missing URLs fall back to a placeholder image.
- **Responsive card layout** — Recommended products are rendered in a 3-column grid with product image, brand, name (linked to Lookfantastic), star rating, and price in GBP (£).

## Dataset

**File:** `skincare_products_2026.csv`

| Column | Description |
|---|---|
| `product_name` | Name of the product |
| `product_url` | Link to the Lookfantastic product page |
| `product_type` | Category (e.g., Moisturiser, Cleanser, Serum) |
| `ingredients` | String-encoded list of ingredients in descending concentration order |
| `price` | Original price (£) |
| `updated_price` | Updated price (£); used as `display_price` when available |
| `price_change` | Difference between original and updated price |
| `product_rating` | Average customer rating (out of 5) |
| `product_image_url` | URL to the product image |
| `brand` | Brand name |
| `available` | Whether the product is currently available (`True`/`False`) |

## How It Works

### 1. Data Loading (`load_data`)

- Reads `skincare_products_2026.csv` into a pandas DataFrame.
- Filters rows to keep only products where `available == True`.
- Parses the `ingredients` column from its string-encoded list format into a Python list.
- Derives `display_price` from `updated_price`, falling back to `price` if `updated_price` is missing.
- Replaces blank or null `product_image_url` values with a placeholder image URL.
- Results are cached with `@st.cache_data` to avoid reloading on each interaction.

### 2. Position-Weighted Ingredient Strings (`_weighted_ingredient_str`)

Ingredients on a cosmetic label are listed in descending order of concentration. To reflect this, each ingredient is repeated in the string passed to TF-IDF according to its position:

- 1st ingredient → repeated 5 times
- 2nd ingredient → repeated 4 times
- …
- 6th and beyond → appear once

This ensures that key actives and base ingredients at the top of the list contribute more to the similarity score than trace ingredients near the bottom.

### 3. Composite Similarity Score (`calculate_cosine_similarity`)

For a selected product at `product_index` within the type-filtered DataFrame:

1. **Ingredient similarity** — TF-IDF vectors are built from weighted ingredient strings; cosine similarity is computed between the selected product and every other product.
2. **Price proximity** — `1 - |price_selected - price_candidate| / price_range`, clipped to `[0, 1]`. Defaults to `0.5` if price is unavailable.
3. **Rating proximity** — `1 - |rating_selected - rating_candidate| / 5.0`, clipped to `[0, 1]`. Defaults to `0.5` if rating is unavailable.
4. **Final score** — `0.75 × ingredient_sim + 0.15 × price_proximity + 0.10 × rating_proximity`

Candidates are ranked by final score (descending). A candidate is only included if:
- It is not the selected product itself.
- Its raw ingredient similarity ≥ `0.1` (the `similarity_threshold`).

Up to `top_n=5` products are returned.

### 4. Streamlit Interface (`main`)

- **Product type selector** — dropdown populated with all unique `product_type` values.
- **Product selector** — dropdown filtered to products of the chosen type.
- **Preview pane** — shows the selected product's image, rating, and price alongside the selectors.
- **Recommendations grid** — automatically computed and rendered in a 3-column card layout whenever the selection changes. No manual submission step is required.
- If no products meet the similarity threshold, an info message is shown instead.

## Project Structure

```
skincare-recommendation/
├── recommender.py                    # Main Streamlit app and recommendation logic
├── skincare_products_2026.csv        # Updated product dataset
├── corrected_brands_dictionary.csv   # Brand name corrections used in the data pipeline
├── skincare_products_2021.csv        # Historical dataset
└── requirements.txt                  # Python dependencies
```

## Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
scikit-learn>=1.3.0
validators>=0.20.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run recommender.py
```