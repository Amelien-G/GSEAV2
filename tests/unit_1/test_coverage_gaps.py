"""
Additional edge-case coverage tests for Unit 1 -- Data Ingestion.

These tests cover edge cases and boundary conditions not fully addressed
in the main test file, focusing on nuanced blueprint requirements.

Synthetic Data Assumptions
--------------------------
DATA ASSUMPTION: Same conventions as the main test file. Mutant folder names
    follow `<mutant_id>.GseaPreranked.<timestamp>` pattern. TSV files use the
    standard GSEA preranked output format with HTML artifact in column 2 and
    trailing tab on each data row.
"""

import re
from pathlib import Path

import pytest

from gsea_tool.data_ingestion import (
    CohortData,
    DataIngestionError,
    MutantProfile,
    TermRecord,
    discover_mutant_folders,
    ingest_data,
    locate_report_files,
    merge_pos_neg,
    parse_gsea_report,
)


# ---------------------------------------------------------------------------
# Helpers (duplicated from main test file to keep this file self-contained)
# ---------------------------------------------------------------------------

GSEA_TSV_HEADER = (
    "NAME\tGS<br> follow link to MSigDB\tGS DETAILS\tSIZE\tES\t"
    "NES\tNOM p-val\tFDR q-val\tFWER p-val\tRANK AT MAX\tLEADING EDGE"
)


def _make_tsv_row(go_id: str, term_name: str, size: int, es: float,
                  nes: float, nom_p: float, fdr: float, fwer: float,
                  rank: int, leading_edge: str = "tags=50%") -> str:
    """Build one GSEA TSV data row with trailing tab artifact."""
    name = f"{go_id} {term_name}"
    return (
        f"{name}\tdetails_link\tdetails\t{size}\t{es}\t{nes}\t{nom_p}\t"
        f"{fdr}\t{fwer}\t{rank}\t{leading_edge}\t"
    )


def _write_tsv(path: Path, rows: list[str]) -> None:
    """Write a complete TSV report file with header and data rows."""
    content = GSEA_TSV_HEADER + "\n"
    for row in rows:
        content += row + "\n"
    path.write_text(content)


def _create_mutant_folder(
    base_dir: Path,
    mutant_id: str,
    pos_rows: list[str],
    neg_rows: list[str],
    suffix: str = ".GseaPreranked.1234567890",
    pos_filename: str = "gsea_report_for_na_pos_1234567890.tsv",
    neg_filename: str = "gsea_report_for_na_neg_1234567890.tsv",
) -> Path:
    """Create a complete mutant subfolder with pos and neg TSV files."""
    folder_name = f"{mutant_id}{suffix}"
    folder = base_dir / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    _write_tsv(folder / pos_filename, pos_rows)
    _write_tsv(folder / neg_filename, neg_rows)
    return folder


SAMPLE_POS_ROW = _make_tsv_row(
    "GO:0000001", "MITOCHONDRION_INHERITANCE", 50, 0.65, 1.85, 0.001, 0.05, 0.01, 5000
)
SAMPLE_NEG_ROW = _make_tsv_row(
    "GO:0000003", "REPRODUCTION", 120, -0.45, -1.55, 0.005, 0.08, 0.03, 7000
)


# ===========================================================================
# Edge case: merge_pos_neg conflict resolution with equal nom_pval
# ===========================================================================


class TestMergePosNegEdgeCases:
    """Edge cases for merge_pos_neg conflict resolution."""

    def test_conflict_with_equal_nom_pval_returns_one_entry(self):
        """When a term appears in both pos and neg with equal nom_pval,
        one entry should be retained (exactly one key in result)."""
        # DATA ASSUMPTION: Equal p-values are rare but possible. The blueprint
        # says "smaller p-value is retained"; with equal values either is
        # acceptable as long as exactly one entry exists.
        pos = [TermRecord(
            term_name="SHARED", go_id="GO:0000001",
            nes=1.5, fdr=0.05, nom_pval=0.01, size=50
        )]
        neg = [TermRecord(
            term_name="SHARED", go_id="GO:0000001",
            nes=-1.5, fdr=0.05, nom_pval=0.01, size=50
        )]

        result = merge_pos_neg(pos, neg)
        assert len(result) == 1
        assert "SHARED" in result

    def test_multiple_conflicts_all_resolved(self):
        """Multiple overlapping terms should each be resolved individually."""
        # DATA ASSUMPTION: Multiple terms may overlap between pos and neg.
        pos = [
            TermRecord(term_name="TERM_A", go_id="GO:0000001", nes=1.0, fdr=0.1, nom_pval=0.05, size=10),
            TermRecord(term_name="TERM_B", go_id="GO:0000002", nes=1.2, fdr=0.05, nom_pval=0.001, size=20),
        ]
        neg = [
            TermRecord(term_name="TERM_A", go_id="GO:0000001", nes=-1.1, fdr=0.08, nom_pval=0.01, size=15),
            TermRecord(term_name="TERM_B", go_id="GO:0000002", nes=-1.3, fdr=0.03, nom_pval=0.02, size=25),
        ]

        result = merge_pos_neg(pos, neg)
        assert len(result) == 2
        # TERM_A: neg has smaller nom_pval (0.01 < 0.05)
        assert result["TERM_A"].nom_pval == pytest.approx(0.01)
        assert result["TERM_A"].nes == pytest.approx(-1.1)
        # TERM_B: pos has smaller nom_pval (0.001 < 0.02)
        assert result["TERM_B"].nom_pval == pytest.approx(0.001)
        assert result["TERM_B"].nes == pytest.approx(1.2)


# ===========================================================================
# Edge case: parse_gsea_report with only invalid rows
# ===========================================================================


class TestParseGseaReportEdgeCases:
    """Edge cases for parse_gsea_report."""

    def test_all_rows_invalid_go_id_returns_empty(self, tmp_path):
        """If all rows lack a valid GO ID, the result should be empty."""
        # DATA ASSUMPTION: A report where no row has a valid GO ID pattern
        # should not crash but return an empty list.
        content = GSEA_TSV_HEADER + "\n"
        content += (
            "INVALID_ROW\tdetails_link\tdetails\t25\t0.3\t1.1\t0.05\t"
            "0.15\t0.10\t2000\ttags=30%\t\n"
        )
        content += (
            "ALSO_INVALID\tdetails_link\tdetails\t30\t0.4\t1.2\t0.03\t"
            "0.10\t0.05\t3000\ttags=40%\t\n"
        )
        tsv_path = tmp_path / "report.tsv"
        tsv_path.write_text(content)

        result = parse_gsea_report(tsv_path)
        assert result == []

    def test_go_id_extracted_as_first_matching_token(self, tmp_path):
        """Contract #4: GO ID is the first token matching GO:\\d{7}.
        If the NAME has multiple GO-like patterns, only the first is used."""
        # DATA ASSUMPTION: Unusual NAME field with multiple GO-like patterns.
        content = GSEA_TSV_HEADER + "\n"
        name = "GO:0000001 TERM_WITH GO:9999999 EXTRA"
        content += (
            f"{name}\tdetails_link\tdetails\t25\t0.3\t1.1\t0.05\t"
            f"0.15\t0.10\t2000\ttags=30%\t\n"
        )
        tsv_path = tmp_path / "report.tsv"
        tsv_path.write_text(content)

        result = parse_gsea_report(tsv_path)
        assert len(result) == 1
        assert result[0].go_id == "GO:0000001"

    def test_nom_pval_zero_is_valid(self, tmp_path):
        """A NOM p-val of 0.0 is a valid numeric value and should be parsed."""
        # DATA ASSUMPTION: NOM p-val of 0.0 occurs when enrichment is highly
        # significant.
        row = _make_tsv_row(
            "GO:0000001", "HIGHLY_SIGNIFICANT", 100, 0.8, 2.5, 0.0, 0.001, 0.0, 9000
        )
        tsv_path = tmp_path / "report.tsv"
        _write_tsv(tsv_path, [row])

        result = parse_gsea_report(tsv_path)
        assert len(result) == 1
        assert result[0].nom_pval == pytest.approx(0.0)


# ===========================================================================
# Edge case: ingest_data error messages are descriptive
# ===========================================================================


class TestIngestDataErrorMessages:
    """Verify that DataIngestionError messages are descriptive."""

    def test_insufficient_mutants_error_message(self, tmp_path):
        """Contract #15: Error message should be descriptive when fewer
        than 2 mutant subfolders are discovered."""
        _create_mutant_folder(
            tmp_path, "solo",
            [SAMPLE_POS_ROW], [SAMPLE_NEG_ROW],
        )
        with pytest.raises(DataIngestionError) as exc_info:
            ingest_data(tmp_path)
        assert str(exc_info.value) != ""

    def test_no_partial_output_on_insufficient_mutants(self, tmp_path):
        """Contract #15: No partial output is produced when fewer than 2
        mutant subfolders are discovered -- the exception halts processing."""
        # Only 1 mutant folder
        _create_mutant_folder(
            tmp_path, "only_one",
            [SAMPLE_POS_ROW], [SAMPLE_NEG_ROW],
        )
        with pytest.raises(DataIngestionError):
            ingest_data(tmp_path)
        # If we got here, the exception was raised and no return value
        # was produced (no partial output).


# ===========================================================================
# Edge case: discover_mutant_folders with special characters in mutant_id
# ===========================================================================


class TestDiscoverMutantFoldersEdgeCases:
    """Edge cases for discover_mutant_folders."""

    def test_mutant_id_with_hyphens_and_underscores(self, tmp_path):
        """Mutant IDs with hyphens and underscores should be preserved."""
        # DATA ASSUMPTION: Mutant identifiers may contain hyphens and
        # underscores as part of standard naming conventions.
        (tmp_path / "my-mutant_01.GseaPreranked.12345").mkdir()

        result = discover_mutant_folders(tmp_path)
        assert len(result) == 1
        assert result[0][0] == "my-mutant_01"

    def test_empty_mutant_id(self, tmp_path):
        """A folder named '.GseaPreranked.12345' has empty mutant_id before
        the first '.GseaPreranked' -- this should still be discovered."""
        # DATA ASSUMPTION: This is a degenerate case. The blueprint says
        # "the portion before the first .GseaPreranked", which is empty string.
        (tmp_path / ".GseaPreranked.12345").mkdir()

        result = discover_mutant_folders(tmp_path)
        assert len(result) == 1
        assert result[0][0] == ""
