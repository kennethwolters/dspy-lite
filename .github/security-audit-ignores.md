# Dependency audit exceptions

| Advisory | Package | Rationale | Owner | Review date |
|---|---|---|---|---|
| `PYSEC-2026-2447` | `diskcache==5.6.3` | No fixed diskcache release exists. dspy-lite defaults to restricted deserialization, rejects unsafe entries as cache misses, warns on explicit unrestricted mode, and pins an exploit regression test. See issue #18 and PR #25. | Kenneth Wolters | 2026-10-01 |

Exceptions apply only to the automated audit exit status. The JSON artifacts retain the complete audit input and the mitigation must be reviewed on or before the listed date.
