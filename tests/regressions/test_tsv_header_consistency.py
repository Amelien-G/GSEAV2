"""Regression test for BUG-007 -- TSV column-header drift.

All three TSV writers emit a column holding the human-readable GO term name:

  * Unit 6 write_pvalue_matrix_tsv      -> pvalue_matrix.tsv
  * Unit 6 write_fisher_results_tsv     -> fisher_combined_pvalues.tsv
  * Unit 7 clustered results writer     -> fisher_combined_pvalues.tsv

Two of them headed that column "Term_Name"; write_pvalue_matrix_tsv headed it
"GO_Term". The value is a term *name*, not a GO term ID, so "GO_Term" was both
the minority and the misleading spelling.

The drift survived because tests/unit_6 asserted
    assert "Term_Name" in header or "GO_Term" in header
which accepts either. This test pins the exact name in column 2 of every writer.
"""

import numpy as np

from gsea_tool.meta_analysis import (
    FisherResult,
    write_fisher_results_tsv,
    write_pvalue_matrix_tsv,
)

EXPECTED_TERM_COLUMN = "Term_Name"


def _header_of(path):
    return path.read_text().splitlines()[0].split("\t")


def _fisher_result(n_contributing: int = 1) -> FisherResult:
    """Minimal single-term FisherResult sufficient for the TSV writers."""
    return FisherResult(
        go_ids=["GO:0008150"],
        go_id_to_name={"GO:0008150": "biological_process"},
        combined_pvalues={"GO:0008150": 0.001},
        n_contributing={"GO:0008150": n_contributing},
        pvalue_matrix=np.array([[0.01]]),
        mutant_ids=["mutA"],
        go_id_order=["GO:0008150"],
        n_mutants=1,
        corrected_pvalues=None,
    )


class TestHeaderConsistency:

    def test_pvalue_matrix_term_column(self, tmp_path):
        matrix = np.array([[0.01, 0.02]])
        nes = np.array([[1.5, -1.2]])
        path = write_pvalue_matrix_tsv(
            matrix,
            nes,
            ["GO:0008150"],
            {"GO:0008150": "biological_process"},
            ["mutA", "mutB"],
            tmp_path,
        )
        header = _header_of(path)
        assert header[0] == "GO_ID"
        assert header[1] == EXPECTED_TERM_COLUMN

    def test_fisher_results_term_column(self, tmp_path):
        result = _fisher_result(n_contributing=2)
        path = write_fisher_results_tsv(result, tmp_path)
        header = _header_of(path)
        assert header[0] == "GO_ID"
        assert header[1] == EXPECTED_TERM_COLUMN

    def test_all_writers_agree(self, tmp_path):
        """The two files must not disagree about what column 2 is called."""
        matrix = np.array([[0.01]])
        nes = np.array([[1.5]])
        m = write_pvalue_matrix_tsv(
            matrix, nes, ["GO:0008150"],
            {"GO:0008150": "biological_process"}, ["mutA"], tmp_path,
        )
        f = write_fisher_results_tsv(_fisher_result(), tmp_path)
        assert _header_of(m)[1] == _header_of(f)[1]

    def test_go_term_spelling_is_gone(self, tmp_path):
        matrix = np.array([[0.01]])
        nes = np.array([[1.5]])
        path = write_pvalue_matrix_tsv(
            matrix, nes, ["GO:0008150"],
            {"GO:0008150": "biological_process"}, ["mutA"], tmp_path,
        )
        assert "GO_Term" not in path.read_text().splitlines()[0]
