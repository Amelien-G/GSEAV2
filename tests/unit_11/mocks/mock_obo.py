"""Tiny synthetic OBO fixture for Unit 11 tests.

The fixture covers two namespaces (BP and MF) with a small enough term set
that the resulting tree fits in a single test snapshot. GO IDs and is_a
relationships mirror real ontology shapes but are entirely synthetic.
"""

from pathlib import Path


SYNTHETIC_OBO_TEXT = """format-version: 1.2
ontology: synthetic-test

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0009987
name: cellular process
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0008152
name: metabolic process
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0006412
name: translation
namespace: biological_process
is_a: GO:0009987 ! cellular process
is_a: GO:0008152 ! metabolic process

[Term]
id: GO:0006119
name: oxidative phosphorylation
namespace: biological_process
is_a: GO:0008152 ! metabolic process

[Term]
id: GO:0042254
name: ribosome biogenesis
namespace: biological_process
is_a: GO:0009987 ! cellular process

[Term]
id: GO:0003674
name: molecular_function
namespace: molecular_function

[Term]
id: GO:0003824
name: catalytic activity
namespace: molecular_function
is_a: GO:0003674 ! molecular_function

[Term]
id: GO:0016740
name: transferase activity
namespace: molecular_function
is_a: GO:0003824 ! catalytic activity
"""


def write_synthetic_obo(target_dir: Path) -> Path:
    """Write the synthetic OBO to target_dir/synthetic.obo and return the path."""
    p = target_dir / "synthetic.obo"
    p.write_text(SYNTHETIC_OBO_TEXT, encoding="utf-8")
    return p
