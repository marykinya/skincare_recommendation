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
FIL_CODE = re.compile(r'\(?\s*code\s*:?\s*f\.?\s*i\.?\s*l\.?[^)]*\)?', re.I)
SYMBOLS = re.compile(r'[®™©¹²³⁴⁵⁶⁷⁸⁹⁰ⁿ*†‡◊•]')
FOOTNOTE_ITEM = re.compile(r'^\(?\s*[\d*†‡]?\s*\)?\s*(naturally occurring|certified organic|from organic|organic|natural origin|of natural origin)', re.I)
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


_CP1252_EXTRA = '€‚ƒ„…†‡ˆ‰Š‹ŒŽ‘’“”•–—˜™š›œžŸ'
_MOJIBAKE = re.compile(r'[\u00c2-\u00f4][\u0080-\u00bf' + _CP1252_EXTRA + r']{1,3}')


def _decode_run(run):
    for enc in ('cp1252', 'latin-1'):
        try:
            return run.encode(enc).decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return run


def fix_mojibake(text):
    """Repair UTF-8 text that was read as Latin-1, e.g. 'EstÃ©e' to 'Estée'.

    Works piece by piece, so one odd character does not stop the rest being fixed.
    Also handles text that was lowercased after the damage ('estã©e').
    """
    if not isinstance(text, str) or not re.search(r'[\u00c2-\u00f4][\u0080-\u00bf' + _CP1252_EXTRA + ']', text):
        return text
    lowered = text == text.lower()
    candidate = text
    if lowered:
        # Lowercasing turned the lead byte Ã into ã and Â into â. Undo that first.
        candidate = re.sub(r'ã(?=[\u0080-\u00bf' + _CP1252_EXTRA + '])', 'Ã', candidate)
        candidate = re.sub(r'â(?=[®©°±²³¹º·¼½¾\u00ad¬])', 'Â', candidate)
    fixed = _MOJIBAKE.sub(lambda m: _decode_run(m.group(0)), candidate)
    return fixed.lower() if lowered else fixed


def clean_ingredient_list(items):
    """Tidy a list of ingredient names. Safe to run more than once."""
    out = []
    for item in items:
        item = fix_mojibake(str(item)).replace('\u00ad', '')  # soft hyphens
        item = FIL_CODE.sub('', item)
        item = DISCLAIMER.sub('', item)
        item = re.sub(r'\.\s*\*.*$', '', item)  # footnotes such as ". *from organic farming"
        item = SYMBOLS.sub('', item)
        # Some brands separate ingredients with full stops instead of commas
        parts = re.split(r'\.\s+(?=[a-z(])', item, flags=re.I) if re.search(r'\.\s+[a-z(]', item, re.I) else [item]
        for part in parts:
            part = re.sub(r'\s+', ' ', part).strip(' .;:\n\t').lower()
            if not part or len(part) > 120 or FOOTNOTE_ITEM.match(part):
                continue
            # "1,2-hexanediol" split at its comma: glue the number back on
            if out and re.fullmatch(r'\d+', out[-1]) and re.match(r'\d', part):
                out[-1] = f'{out[-1]},{part}'
                continue
            out.append(part)
    return out


def split_ingredients(text):
    """Split an INCI list on commas that are not inside brackets or between digits."""
    text = FIL_CODE.sub('', text)
    text = DISCLAIMER.sub('', text)
    items, depth, current = [], 0, []
    for k, ch in enumerate(text):
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth = max(0, depth - 1)
        between_digits = ch == ',' and 0 < k < len(text) - 1 and text[k - 1].isdigit() and text[k + 1].isdigit()
        if ch in ',•;' and depth == 0 and not between_digits:
            items.append(''.join(current))
            current = []
        else:
            current.append(ch)
    items.append(''.join(current))
    return clean_ingredient_list(items)


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


BUNDLE = re.compile(
    r'advent|beauty box|subscription|\bbundle\b|voucher|discovery|\btrio\b|\bgift\b|'
    r'\bset\b|\bkit\b|\bedit\b|\bvault\b|\broutine\b|\(worth', re.I)
MIXED_DUO = re.compile(r'\bduo\b', re.I)


def is_bundle(name):
    """True for gift boxes, sets and mixed bundles, which have no single formula.

    A duo of one product ("Skin Food 75ml Duo") is kept. A duo of two products
    ("Cleanser and Rescue Mask Duo") is a bundle.
    """
    name = str(name)
    if BUNDLE.search(name):
        # "(Worth £108)" alone on a single product with a size is a value pack, not a set
        if re.search(r'\(worth', name, re.I) and not re.search(
                r'advent|box|bundle|set|kit|edit|vault|collection|duo|trio|\+|&|\band\b', name, re.I):
            return False
        return True
    if MIXED_DUO.search(name):
        # Two different products, or a duo with no size (a pairing of two products)
        has_size = re.search(r'\d+(?:\.\d+)?\s?(?:ml|g)\b', name, re.I)
        return bool(re.search(r'\+|&|\band\b', name, re.I) or not has_size)
    return False
