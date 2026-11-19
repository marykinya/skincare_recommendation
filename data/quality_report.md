# Data quality report

- Run date: 2026-11-19
- Products in catalogue: 1138
- Available products: 1032
- New products added: 0
- Pages visited: 20
- Failed requests: 0

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
| No products were lost | Pass | 1138 before, 1138 after |
| Most pages loaded | Pass | 0 of 20 requests failed (0%) |
| Ingredient lists look complete | Warning | 23 lists with fewer than 3 items |
| Brands are filled in | Warning | 5.3% of available products have no brand |
| Ingredients come from one source | Warning | 98.4% of available products still use 2021 Kaggle ingredients |
