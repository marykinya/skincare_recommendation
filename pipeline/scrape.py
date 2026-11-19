"""Network layer: fetch category and product pages politely."""
import logging
import time

import requests

from pipeline import config
from pipeline.parse import extract_product_links, page_count, parse_product_page

log = logging.getLogger(__name__)


class Fetcher:
    """One request at a time, a pause between requests, retries with backoff."""

    def __init__(self, delay=config.REQUEST_DELAY_SECONDS):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.USER_AGENT,
            'Accept-Language': 'en-GB,en;q=0.9',
        })
        self._last = 0.0

    def get(self, url):
        """Return (status_code, html). status_code is None after repeated network errors."""
        for attempt in range(1, config.MAX_RETRIES + 1):
            wait = self.delay - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self.session.get(url, timeout=config.REQUEST_TIMEOUT_SECONDS)
            except requests.RequestException as exc:
                log.warning('Attempt %s failed for %s: %s', attempt, url, exc)
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                log.warning('Attempt %s got %s for %s', attempt, resp.status_code, url)
                time.sleep(2 ** attempt * 2)
                continue
            # Lookfantastic serves UTF-8 but does not always say so. Without this,
            # "Estée" turns into "EstÃ©e".
            if not resp.encoding or resp.encoding.lower() in ('iso-8859-1', 'latin-1'):
                resp.encoding = 'utf-8'
            return resp.status_code, resp.text
        return None, ''


def discover(fetcher, categories, max_pages):
    """Walk category listings. Returns {product_id: (url, product_type)}."""
    found = {}
    for product_type, path in categories.items():
        url = config.BASE_URL + path
        status, html = fetcher.get(url)
        if status != 200:
            log.warning('Could not load category %s (%s)', product_type, status)
            continue
        pages = min(page_count(html), max_pages)
        for page in range(1, pages + 1):
            if page > 1:
                status, html = fetcher.get(f'{url}?pageNumber={page}')
                if status != 200:
                    log.warning('Stopped %s at page %s (%s)', product_type, page, status)
                    break
            links = extract_product_links(html, config.BASE_URL)
            for pid, link in links.items():
                found.setdefault(pid, (link, product_type))
        log.info('%s: %s pages read, %s products known so far', product_type, pages, len(found))
    return found


def fetch_product(fetcher, url):
    """Return (status, parsed_fields). Parsed fields are empty unless status is 200."""
    status, html = fetcher.get(url)
    if status != 200:
        return status, {}
    return status, parse_product_page(html)
