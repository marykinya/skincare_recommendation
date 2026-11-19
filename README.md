# Skincare Product Recommender

Find skincare products with a similar formula. Pick a product and the app shows up to five similar products of the same type. You can also look for cheaper alternatives or other brands.

**Live app:** [mk-skincare-recommendation.streamlit.app](https://mk-skincare-recommendation.streamlit.app/)

<!-- Add a screenshot or GIF of the app here, for example: ![App screenshot](docs/screenshot.png) -->

## Features

- **Ingredient matching.** Products are compared on their full ingredient lists. Ingredients listed first carry more weight.
- **Dupe finder.** Show only products that cost less than the one you picked.
- **Other brands only.** Hide products from the same brand.
- **Transparent results.** Each card shows its ingredient match, rating and price with a link to the product page.

## How it works

Every product is compared with all other products of the same type. Each candidate gets a score from three signals.

| Signal | Weight | What it measures |
|---|---|---|
| Ingredient match | 75% | How similar the two formulas are |
| Price | 15% | Similar price by default. In dupe mode a bigger saving scores higher. |
| Rating | 10% | Higher rated products score a little higher |

**Ingredient match.** Each full ingredient name is one token. "Sodium hyaluronate" is matched as a whole and is never split into separate words. Ingredient lists are ordered by concentration so the top five ingredients are repeated to give them more weight. TF-IDF then gives rare ingredients more weight than common ones like glycerin. Cosine similarity turns this into a score between 0 and 1.

**Similarity threshold.** A product needs an ingredient match of at least 0.1 to appear. Price and rating cannot push an unrelated product into the results.

**Missing ratings.** Products with no rating get the median rating for their product type.

```
score = 0.75 × ingredient match + 0.15 × price + 0.10 × rating
```

## Data

The catalogue started from a [Kaggle](https://www.kaggle.com) dataset of [Lookfantastic](https://www.lookfantastic.com) skincare products from 2021. I refreshed it in 2026 by scraping each product page for current prices, ratings, images and availability. Brands were extracted from product names and fixed with a manual corrections file.

| | |
|---|---|
| Products | 1,138 |
| Available (shown in the app) | 1,032 |
| Product types | 14 (Mask, Body Wash, Moisturiser, Cleanser, Serum and more) |
| Average ingredients per product | 24 |

**File:** `data/skincare_products_2026.csv`

| Column | Description |
|---|---|
| `product_name` | Name of the product |
| `product_url` | Link to the Lookfantastic product page |
| `product_type` | Category such as Moisturiser, Cleanser or Serum |
| `ingredients` | List of ingredients in order of concentration |
| `price` | Price in the 2021 dataset (£) |
| `updated_price` | Price at the 2026 refresh (£). Used in the app when available. |
| `price_change` | Difference between the two prices |
| `product_rating` | Average customer rating out of 5 |
| `product_image_url` | Product image |
| `brand` | Brand name |
| `available` | Whether the product is still sold |

## Project structure

```
skincare_recommendation/
├── recommender.py                         # Streamlit app and recommendation logic
├── requirements.txt                       # App dependencies
├── data/
│   ├── skincare_products_2026.csv         # Refreshed dataset used by the app
│   ├── skincare_products_2021.csv         # Original Kaggle dataset
│   └── corrected_brands_dictionary.csv    # Manual brand corrections
├── notebooks/
│   └── Skincare.ipynb                     # Data refresh: scraping, brand matching and cleaning
└── tests/
    └── test_recommender.py                # Tests for the scoring logic
```

## Run it locally

```bash
pip install -r requirements.txt
streamlit run recommender.py
```

## Run the tests

```bash
pip install pytest
pytest
```

## Roadmap

- Evaluate recommendation quality against known dupe pairs and a random baseline
- Move the scraping steps from the notebook into a scheduled pipeline with data quality checks
- Fill in the remaining missing brands
- Let users tune the signal weights in the app
