# Data quality report

- Run date: 2026-11-19
- Step: repair (no scraping)
- Products in catalogue: 4799
- Available products: 4611

| Check | Result | Detail |
|---|---|---|
| All columns present | Pass | ok |
| No duplicate products | Pass | 0 rows share an ID and name |
| No missing product_id | Pass | 0 missing |
| No missing product_name | Pass | 0 missing |
| No missing product_url | Pass | 0 missing |
| No missing product_type | Pass | 0 missing |
| Product types are known | Pass | ok |
| Prices are between £0 and £1,000 | Pass | 0 out of range |
| Ratings are between 0 and 5 | Pass | 0 out of range |
| Available products have ingredients | Pass | 100.0% have ingredients (minimum 90%) |
| No products were lost | Pass | 4799 before, 4799 after |
| Prices did not jump | Pass | 0 prices changed more than 3x |
| Ingredient lists look complete | Warning | 41 lists with fewer than 3 items |
| Brands are filled in | Pass | 1.0% of available products (not counting gift sets) have no brand |
| Ingredients come from one source | Warning | 2.6% of available products still use 2021 Kaggle ingredients |
