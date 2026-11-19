"""Refresh the catalogue.

Usage:
    python -m pipeline.run                  # full refresh
    python -m pipeline.run --max-pages 2    # quick run with 2 pages per category
    python -m pipeline.run --limit 50       # only re-check 50 existing products (for testing)
    python -m pipeline.run --no-discover    # refresh existing products only
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from pipeline import config
from pipeline.checks import report_markdown, run_checks
from pipeline.merge import apply_results
from pipeline.scrape import Fetcher, discover, fetch_product

log = logging.getLogger('pipeline')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--max-pages', type=int, default=50, help='pages to read per category')
    parser.add_argument('--limit', type=int, default=None, help='only re-check this many existing products')
    parser.add_argument('--no-discover', action='store_true', help='skip finding new products')
    parser.add_argument('--catalogue', default=config.CATALOGUE_PATH)
    parser.add_argument('--report', default=config.REPORT_PATH)
    parser.add_argument('--delay', type=float, default=config.REQUEST_DELAY_SECONDS)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    old = pd.read_csv(args.catalogue, dtype={'product_id': str})
    fetcher = Fetcher(delay=args.delay)

    # 1. Work out which pages to visit
    targets = {pid: (url, ptype) for pid, url, ptype in
               old[['product_id', 'product_url', 'product_type']].itertuples(index=False)}
    if args.limit:
        targets = dict(list(targets.items())[:args.limit])
    new_found = 0
    if not args.no_discover:
        for pid, (url, ptype) in discover(fetcher, config.CATEGORIES, args.max_pages).items():
            if pid not in targets and pid not in set(old['product_id']):
                targets[pid] = (url, ptype)
                new_found += 1
    log.info('Visiting %s product pages (%s newly discovered)', len(targets), new_found)

    # 2. Fetch every product page
    results, failed = [], 0
    for n, (pid, (url, ptype)) in enumerate(targets.items(), 1):
        status, fields = fetch_product(fetcher, url)
        if status is None or (status != 200 and status not in (404, 410)):
            failed += 1
        results.append({'product_id': pid, 'url': url, 'product_type': ptype, 'status': status, 'fields': fields})
        if n % 50 == 0:
            log.info('%s / %s pages done', n, len(targets))

    # 3. Merge, check, save
    new = apply_results(old, results)
    checks = run_checks(new, old, attempted=len(targets), failed=failed)
    summary = {
        'Run date': date.today().isoformat(),
        'Products in catalogue': len(new),
        'Available products': int((new['available'].astype(str).str.lower() == 'true').sum()),
        'New products added': len(new) - len(old),
        'Pages visited': len(targets),
        'Failed requests': failed,
    }
    Path(args.report).write_text(report_markdown(checks, summary))
    errors = [c for c in checks if not c.passed and c.severity == 'error']
    for c in checks:
        log.info('%s %s: %s', 'PASS' if c.passed else c.severity.upper(), c.name, c.detail)
    if errors:
        log.error('%s checks failed. Catalogue not saved. See %s', len(errors), args.report)
        return 1
    new.to_csv(args.catalogue, index=False)
    log.info('Saved %s products to %s', len(new), args.catalogue)
    return 0


if __name__ == '__main__':
    sys.exit(main())
