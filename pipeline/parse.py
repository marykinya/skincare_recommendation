"""Pure parsing functions. They take HTML and return data, with no network calls."""
import json
import re

from bs4 import BeautifulSoup

PRODUCT_LINK = re.compile(r'(?:https?://(?:www\.)?lookfantastic\.com)?/p/(?:[a-z0-9-]+/)*(\d{6,9})/?')
LEGACY_ID = re.compile(r'/(\d{6,9})(?:\.html|/)?(?:[?#].*)?$')
PAGE_COUNT = re.compile(r'Page\s+\d+\s+of\s+(\d+)', re.I)
INGREDIENT_LABEL = re.compile(r'^\s*(?:full\s+)?ingredients?(?:\s+list)?\s*:?\s*$', re.I)
SECTION_LABELS = re.compile(
    r'^\s*(how to use|description|ingredient benefits|key ingredients|delivery|reviews?|'
    r'product details|brand|range|volume|size)\s*:?\s*$', re.I)
FIL_CODE = re.compile(r'\(?\s*code\s*f\.?\s*i\.?\s*l\.?[^)]*\)?', re.I)
DISCLAIMER = re.compile(r'(please note|ingredients? (?:are|may be) subject to change|'
                        r'refer to the packaging|for the most up to date).*$', re.I)


def product_id_from_url(url):
    """Lookfantastic product ID from a new (/p/slug/123/) or old (/slug/123.html) URL."""
    url = str(url)
    match = PRODUCT_LINK.search(url) or LEGACY_ID.search(url)
    return match.group(1) if match else None


def extract_product_links(html, base_url):
    """All product links on a category page, as {product_id: canonical_url}."""
    links = {}
    for match in PRODUCT_LINK.finditer(html):
        path = match.group(0)
        url = path if path.startswith('http') else base_url + path
        url = url if url.endswith('/') else url + '/'
        links.setdefault(match.group(1), url)
    return links


def page_count(html):
    match = PAGE_COUNT.search(BeautifulSoup(html, 'html.parser').get_text(' '))
    return int(match.group(1)) if match else 1


def split_ingredients(text):
    """Split an INCI list on commas that are not inside brackets."""
    text = FIL_CODE.sub('', text)
    text = DISCLAIMER.sub('', text)
    items, depth, current = [], 0, []
    for ch in text:
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth = max(0, depth - 1)
        if ch in ',•;' and depth == 0:
            items.append(''.join(current))
            current = []
        else:
            current.append(ch)
    items.append(''.join(current))
    cleaned = []
    for item in items:
        item = re.sub(r'\.\s*\*.*$', '', item)  # footnotes such as ". *from organic farming"
        item = re.sub(r'\s+', ' ', item).strip(' .*\n\t')
        if item and len(item) <= 120:
            cleaned.append(item.lower())
    return cleaned


def _json_ld_product(soup):
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(script.string or '')
        except (json.JSONDecodeError, TypeError):
            continue
        candidates = data if isinstance(data, list) else data.get('@graph', [data])
        for item in candidates:
            if isinstance(item, dict) and item.get('@type') in ('Product', ['Product']):
                return item
    return {}


def _ingredients_text(soup):
    """Find the text that follows an ingredients label."""
    for label in soup.find_all(string=INGREDIENT_LABEL):
        for text in label.find_all_next(string=True, limit=60):
            t = text.strip()
            if not t or INGREDIENT_LABEL.match(t):
                continue
            if SECTION_LABELS.match(t):
                break
            if t.count(',') >= 2:
                return t
    # Fallback: label and list in the same text block
    match = re.search(r'Full Ingredients List\s*:?\s*([^\n]{20,})', soup.get_text('\n'), re.I)
    return match.group(1) if match else ''


def _number(value):
    try:
        return float(str(value).replace('£', '').replace(',', '').strip())
    except (TypeError, ValueError):
        return None


def parse_product_page(html):
    """Pull product details from a product page. Missing fields come back as None."""
    soup = BeautifulSoup(html, 'html.parser')
    ld = _json_ld_product(soup)

    name = ld.get('name')
    if not name and soup.h1:
        name = soup.h1.get_text(' ', strip=True)

    brand = ld.get('brand')
    if isinstance(brand, dict):
        brand = brand.get('name')

    offers = ld.get('offers') or {}
    if isinstance(offers, list):
        offers = offers[0] if offers else {}
    price = _number(offers.get('price')) if isinstance(offers, dict) else None
    if price is None:
        match = re.search(r'Current price:\s*£\s*([\d,.]+)', soup.get_text(' '))
        price = _number(match.group(1)) if match else None
    in_stock = None
    availability = str(offers.get('availability', '')) if isinstance(offers, dict) else ''
    if availability:
        in_stock = 'instock' in availability.lower().replace('_', '')

    rating = ld.get('aggregateRating') or {}
    image = ld.get('image')
    if isinstance(image, list):
        image = image[0] if image else None
    if isinstance(image, dict):
        image = image.get('url')
    if not image:
        og = soup.find('meta', property='og:image')
        image = og.get('content') if og else None

    canonical = soup.find('link', rel='canonical')
    return {
        'product_name': name,
        'brand': brand,
        'updated_price': price,
        'in_stock': in_stock,
        'product_rating': _number(rating.get('ratingValue')),
        'review_count': _number(rating.get('reviewCount') or rating.get('ratingCount')),
        'product_image_url': image,
        'canonical_url': canonical.get('href') if canonical else None,
        'ingredients': split_ingredients(_ingredients_text(soup)),
    }
