# Documentation provenance before the per-idea catalogue

The active documentation is the [research catalogue](../../research/README.md),
[architecture/status](../../STATUS.md), [command reference](../../COMMANDS.md)
and [development guide](../../DEVELOPMENT.md). This directory is evidence custody,
not another user-guide archive to browse in parallel.

`pre-catalogue-20261106.json.gz` preserves exact original text for 41 documents
and documentation manifests before reorganization. Each record has its original
repository-relative path, byte count, SHA-256 and UTF-8 text. It includes the
retired chronological ledgers, detailed encoding/lifting notes, duplicate
experiment reports, old README/status/guide text and original diagrams.

Archive SHA-256:
`182ae897e0ad4c18b4cfdae0a8b81f3d9abbbcd320746d9050a865c1a1a286de`.

[migration.json](migration.json) maps old active pages to their replacements and
identifies compatibility pointers. [reorganization.json](reorganization.json)
records the unchanged compiler/spec/test/environment source check. Raw hardware
measurements and native instruction/source identities remain in their original
locations; no measurement was rerun or rewritten by this documentation change.

To inspect one original page without installing packages:

```sh
python3 - <<'PYCODE'
import gzip, json
from pathlib import Path
p = Path('docs/evidence/documentation-history/pre-catalogue-20261106.json.gz')
archive = json.loads(gzip.decompress(p.read_bytes()))
old_path = 'docs/research/A100_MMA_RESULTS.md'
print(next(item['text'] for item in archive['documents'] if item['path'] == old_path))
PYCODE
```

Historical manifests that fingerprinted human documentation are themselves
preserved in this snapshot. Current metadata records documentation relocation;
production code, raw JSON measurements and original hardware source fingerprints
are not silently refreshed or reinterpreted as new benchmark results.

## November 9 dispatch-integration cleanup

Seven later chronological reports and plans were also consolidated into their
canonical idea dossiers and reverse-engineering topics. Their active text
contained superseded promotion statements and an accepted-region scheduler
claim overturned by the strict dependency audit. Their exact original text,
byte counts and hashes are retained in
[`pre-dispatch-integration-20261109.json.gz`](pre-dispatch-integration-20261109.json.gz).
The [cleanup map](dispatch-cleanup-20261109.json) names each retired path and
its current replacement. Archive SHA-256:
`856d303b3edc6c23db4f9cf5b4af7c81252d2f156ba113da6cdf712fec58795d`.

All original hardware JSON records, success/failure gates, cubins and source
identities remain intact. Removing a duplicate narrative does not turn a
historical experiment into a current compiler validation.

## November 11 current-documentation refresh

The [refresh record](current-docs-refresh-20261111.json) identifies the base
revision, changed documentation hashes, current compiler/table identities and
offline validation scope for the comprehensive current-docs update. It records
no new GPU measurement. Original binary, benchmark and snapshot manifests stay
unchanged; their dated source identity remains the authority for those runs.

The older `exp/weaknesses/check_evidence.py` compares its frozen snapshot with
files on disk. It now reports both documentation changes and compiler/test/log
drift already present before this refresh. The refresh record separates those
differences. A historical snapshot mismatch is not a failed fresh kernel test,
and refreshing the old manifest would not revalidate that kernel.
