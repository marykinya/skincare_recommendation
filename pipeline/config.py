"""Pipeline settings. Edit categories here to add or remove product types."""

BASE_URL = 'https://www.lookfantastic.com'
CATALOGUE_PATH = 'data/skincare_products.csv'
REPORT_PATH = 'data/quality_report.md'

# Category listing pages used to discover products, keyed by the app's product type.
# Products already in the catalogue keep their type. New products take the type of
# the first category they are found in.
CATEGORIES = {
    'Moisturiser': '/c/health-beauty/face/skincare-products/moisturisers/',
    'Serum': '/c/health-beauty/face/skincare-products/specific-care/serums/',
    'Cleanser': '/c/health-beauty/face/cleansers/',
    'Eye Care': '/c/health-beauty/face/skincare-products/eye-care/',
    'Toner': '/c/health-beauty/face/skincare-products/toners/',
    'Exfoliator': '/c/health-beauty/face/skincare-products/exfoliators/',
    'Peel': '/c/health-beauty/face/peels/',
    'Mask': '/c/health-beauty/face/masks/',
    'Oil': '/c/health-beauty/face/oils/',
    'Mist': '/c/health-beauty/face/mists/',
    'Body Wash': '/c/health-beauty/body/bath-shower/body-washes/',
    'Bath Oil': '/c/health-beauty/body/bath-shower/bath-oils-soaks/',
    'Bath Salts': '/c/health-beauty/body/bath-shower/salts/',
}

# Every product type the app knows. Balm has no single category page so balms are
# refreshed but not discovered.
PRODUCT_TYPES = set(CATEGORIES) | {'Balm'}

# Be polite: one request at a time with a pause between them.
REQUEST_DELAY_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRIES = 3
USER_AGENT = (
    'Mozilla/5.0 (compatible; skincare-recommender-portfolio/1.0; '
    '+https://github.com/marykinya/skincare_recommendation)'
)

# Data quality thresholds
MAX_FETCH_FAILURE_RATE = 0.20
MIN_INGREDIENT_COVERAGE = 0.90
MAX_PRICE_GBP = 1000
