# Diagnostic Trace Retention

Twenty-one large historical diagnostic CSV traces were moved out of the active
output tree after verified archival. Compact JSON reports, manifests, and
keyframes remain in `diagnostics/` so failures and design changes are still
readable without carrying 718 MiB of raw debug traces.

The external archive, hash, member count, and verification procedure are stored
in `DIAGNOSTIC_RETENTION.json`. Restore from the repository root only when a
specific historical trace needs to be re-analyzed.
