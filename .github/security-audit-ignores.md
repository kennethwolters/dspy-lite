# Dependency audit exceptions

No advisories are currently excluded from the automated audit. The former
`PYSEC-2026-2447` exception was removed with the DiskCache dependency; persistent
caching now uses non-executable JSON storage. See issue #71.
