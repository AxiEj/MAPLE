"""Reviewed qualification receipts installed after candidate validation.

Entries are (packaged receipt filename, SHA256 of its bytes). This data-only
module is excluded from the earlier qualification source snapshot and included
in the subsequent no-injection activation replay, avoiding a self-hash cycle.
"""

INSTALLED_QUALIFICATIONS: tuple[tuple[str, str], ...] = ()
