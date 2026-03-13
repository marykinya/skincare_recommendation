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

When you select a product, every other product of the same category is scored against your selection using a weighted combination of three signals. Here's a deep dive into each one.

### 🧪 Ingredient Match (75%)

This is the heart of the engine. Skincare products list their ingredients in **descending order of concentration** — the first ingredient makes up the largest share of the formula, the last the smallest. The recommender takes full advantage of this structure.

**How it works:**
- Each ingredient is assigned a **positional weight**: ingredients near the top of the list carry much more influence than those near the bottom.
- The app then computes a **weighted cosine similarity** between two ingredient lists, meaning products that share the same key actives *and* the same base ingredients will score highly together.
- Rare or specialty ingredients (e.g. niacinamide, retinol, peptides) that appear near the top are treated as strong signals of formulation intent and boost similarity more than common fillers like water or glycerin.

**Why it matters:**
Two moisturisers might both be called "hydrating" on their labels, but one might be built around hyaluronic acid while the other leans on shea butter. Ingredient matching surfaces the one that is chemically closest to what you already like.

### 💰 Price (15%)

Once ingredient similarity is established, price acts as a **tiebreaker and alternative-finder**.

**How it works:**
- The absolute price difference between your chosen product and every candidate is calculated and normalised across the full price range in the dataset.
- A candidate that costs nearly the same scores close to 1.0 on this signal; one that is drastically cheaper or more expensive scores lower.
- The 15% weight means price can meaningfully separate two otherwise identical formulas, but it cannot rescue a product with poor ingredient overlap.

**Why it matters:**
This signal powers the "dupe" use case. If you love a luxury serum but want a budget alternative, the recommender will surface the closest-matching formula at the lowest price difference — not just the cheapest product in the category.

### ⭐ Rating (10%)

A light quality signal drawn from aggregated customer reviews.

**How it works:**
- Ratings are normalised to a 0–1 scale across the dataset.
- Products with a rating close to your chosen product's rating receive a small boost; a highly-rated recommendation is preferred over a poorly-rated one when everything else is equal.
- The 10% weight keeps this signal advisory rather than dominant — a 5-star product with unrelated ingredients will never outrank a 3-star product with an almost identical formula.

**Why it matters:**
Real-world performance data from thousands of reviewers can catch things ingredient lists can't: texture, scent, packaging, and skin-feel. The rating nudge helps surface products that work well in practice, not just on paper.

### 🔬 The Similarity Threshold

Not every product makes it into your results. A candidate must clear a **minimum ingredient similarity score** to appear at all. This threshold filters out products that are categorically different in formulation, so price and rating alone can never surface an unrelated result.

Products that pass the threshold are then ranked by their combined weighted score (ingredient × 0.75 + price × 0.15 + rating × 0.10) and the top matches are displayed.

### 📊 About the Data

- The product catalogue was initially sourced from a publicly available skincare dataset on [Kaggle](https://www.kaggle.com) and includes ingredients, prices, and ratings across moisturisers, serums, cleansers, sunscreens, toners, eye creams, and more.
- The dataset has been periodically refreshed/scraped from [lookfantastic.com](https://www.lookfantastic.com) to reflect current product availability.

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