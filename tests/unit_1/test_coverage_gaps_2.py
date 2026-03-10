"""
Additional coverage gap tests for Unit 1 -- Data Ingestion (round 2).

These tests cover behavioral contracts and edge cases not fully addressed
by the existing test files, identified through systematic blueprint review.

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
# Helpers (duplicated to keep this file self-contained)
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
# Gap 1: Contract #4 -- term_name content when NAME has multiple GO patterns
# ===========================================================================


class TestParseTermNameWithMultipleGOPatterns:
    """Contract #4: When NAME contains multiple GO ID patterns, the term_name
    should be everything after the FIRST GO ID, stripped and uppercased."""

    def test_term_name_includes_subsequent_go_patterns(self, tmp_path):
        """When the NAME field contains multiple GO-like patterns,
        the term_name is everything after the first GO ID (including
        the second GO pattern as literal text)."""
        # DATA ASSUMPTION: Unusual NAME field with multiple GO-like patterns.
        # The first GO ID is extracted; everything after it becomes term_name.
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
        # The term_name should be everything after "GO:0000001 ", uppercased
        assert result[0].term_name == "TERM_WITH GO:9999999 EXTRA"


# ===========================================================================
# Gap 2: Contract #5 -- stderr warning for rows with invalid GO ID
# ===========================================================================


class TestStderrWarningForInvalidGOID:
    """Contract #5: Rows without a valid GO ID are skipped with a warning
    to stderr. This test verifies the warning is actually emitted."""

    def test_warning_to_stderr_for_invalid_go_id(self, tmp_path, capsys):
        """Contract #5: Skipping a row without a valid GO ID should produce
        a warning message on stderr."""
        # DATA ASSUMPTION: A row without any GO:NNNNNNN pattern should trigger
        # a warning to stderr.
        content = GSEA_TSV_HEADER + "\n"
        content += (
            "NO_GO_ID_HERE\tdetails_link\tdetails\t25\t0.3\t1.1\t0.05\t"
            "0.15\t0.10\t2000\ttags=30%\t\n"
        )
        tsv_path = tmp_path / "report.tsv"
        tsv_path.write_text(content)

        parse_gsea_report(tsv_path)
        captured = capsys.readouterr()
        # The blueprint says "skipped with a warning to stderr"
        assert len(captured.err) > 0, "Expected a warning on stderr for invalid GO ID row"


# ===========================================================================
# Gap 3: Contract #6 -- stderr warning for rows with non-numeric values
# ===========================================================================


class TestStderrWarningForNonNumericValues:
    """Contract #6: Rows with missing or non-numeric values in NES, FDR q-val,
    NOM p-val, or SIZE are skipped with a warning to stderr."""

    def test_warning_to_stderr_for_nonnumeric_nes(self, tmp_path, capsys):
        """Contract #6: A row with non-numeric NES should produce a warning
        on stderr."""
        # DATA ASSUMPTION: Some rows may have "---" or similar placeholder
        # values in NES column when GSEA cannot compute enrichment.
        content = GSEA_TSV_HEADER + "\n"
        content += (
            "GO:0000099 BAD_NES\tdetails_link\tdetails\t25\t0.3\t---\t0.05\t"
            "0.15\t0.10\t2000\ttags=30%\t\n"
        )
        tsv_path = tmp_path / "report.tsv"
        tsv_path.write_text(content)

        parse_gsea_report(tsv_path)
        captured = capsys.readouterr()
        assert len(captured.err) > 0, "Expected a warning on stderr for non-numeric NES"

    def test_warning_to_stderr_for_nonnumeric_size(self, tmp_path, capsys):
        """Contract #6: A row with non-numeric SIZE should produce a warning
        on stderr."""
        # DATA ASSUMPTION: SIZE column with non-numeric value like "NA"
        content = GSEA_TSV_HEADER + "\n"
        content += (
            "GO:0000099 BAD_SIZE\tdetails_link\tdetails\tNA\t0.3\t1.1\t0.05\t"
            "0.15\t0.10\t2000\ttags=30%\t\n"
        )
        tsv_path = tmp_path / "report.tsv"
        tsv_path.write_text(content)

        parse_gsea_report(tsv_path)
        captured = capsys.readouterr()
        assert len(captured.err) > 0, "Expected a warning on stderr for non-numeric SIZE"

    def test_warning_to_stderr_for_nonnumeric_fdr(self, tmp_path, capsys):
        """Contract #6: A row with non-numeric FDR q-val should produce a
        warning on stderr."""
        # DATA ASSUMPTION: FDR column with non-numeric value
        content = GSEA_TSV_HEADER + "\n"
        content += (
            "GO:0000099 BAD_FDR\tdetails_link\tdetails\t25\t0.3\t1.1\t0.05\t"
            "N/A\t0.10\t2000\ttags=30%\t\n"
        )
        tsv_path = tmp_path / "report.tsv"
        tsv_path.write_text(content)

        parse_gsea_report(tsv_path)
        captured = capsys.readouterr()
        assert len(captured.err) > 0, "Expected a warning on stderr for non-numeric FDR"

    def test_warning_to_stderr_for_nonnumeric_nom_pval(self, tmp_path, capsys):
        """Contract #6: A row with non-numeric NOM p-val should produce a
        warning on stderr."""
        # DATA ASSUMPTION: NOM p-val column with non-numeric value
        content = GSEA_TSV_HEADER + "\n"
        content += (
            "GO:0000099 BAD_PVAL\tdetails_link\tdetails\t25\t0.3\t1.1\tXXX\t"
            "0.15\t0.10\t2000\ttags=30%\t\n"
        )
        tsv_path = tmp_path / "report.tsv"
        tsv_path.write_text(content)

        parse_gsea_report(tsv_path)
        captured = capsys.readouterr()
        assert len(captured.err) > 0, "Expected a warning on stderr for non-numeric NOM p-val"


# ===========================================================================
# Gap 4: locate_report_files error messages include mutant_id
# ===========================================================================


class TestLocateReportFilesErrorDescriptiveness:
    """Verify that locate_report_files error messages are descriptive and
    include the mutant_id so the user knows which mutant folder is problematic."""

    def test_missing_pos_file_error_includes_mutant_id(self, tmp_path):
        """Error message for missing positive report should include mutant_id."""
        # DATA ASSUMPTION: Only a neg file exists; pos is missing.
        (tmp_path / "gsea_report_for_na_neg_1234.tsv").touch()

        with pytest.raises(DataIngestionError, match="my_mutant"):
            locate_report_files(tmp_path, "my_mutant")

    def test_missing_neg_file_error_includes_mutant_id(self, tmp_path):
        """Error message for missing negative report should include mutant_id."""
        # DATA ASSUMPTION: Only a pos file exists; neg is missing.
        (tmp_path / "gsea_report_for_na_pos_1234.tsv").touch()

        with pytest.raises(DataIngestionError, match="my_mutant"):
            locate_report_files(tmp_path, "my_mutant")

    def test_ambiguous_pos_file_error_includes_mutant_id(self, tmp_path):
        """Error message for ambiguous positive report should include mutant_id."""
        # DATA ASSUMPTION: Two pos files exist, triggering ambiguity error.
        (tmp_path / "gsea_report_for_na_pos_111.tsv").touch()
        (tmp_path / "gsea_report_for_na_pos_222.tsv").touch()
        (tmp_path / "gsea_report_for_na_neg_111.tsv").touch()

        with pytest.raises(DataIngestionError, match="test_mut"):
            locate_report_files(tmp_path, "test_mut")

    def test_ambiguous_neg_file_error_includes_mutant_id(self, tmp_path):
        """Error message for ambiguous negative report should include mutant_id."""
        # DATA ASSUMPTION: Two neg files exist, triggering ambiguity error.
        (tmp_path / "gsea_report_for_na_pos_111.tsv").touch()
        (tmp_path / "gsea_report_for_na_neg_111.tsv").touch()
        (tmp_path / "gsea_report_for_na_neg_222.tsv").touch()

        with pytest.raises(DataIngestionError, match="test_mut"):
            locate_report_files(tmp_path, "test_mut")


# ===========================================================================
# Gap 5: Contract #2 -- non-GseaPreranked entries are SILENTLY ignored
#   (no error, no warning for regular files or folders without the pattern)
# ===========================================================================


class TestSilentIgnoreNonGseaEntries:
    """Contract #2: Other entries (files, folders without .GseaPreranked)
    are silently ignored -- they produce no error and no warning."""

    def test_files_in_data_dir_silently_ignored(self, tmp_path, capsys):
        """Regular files in data_dir should be silently ignored (no stderr)."""
        # DATA ASSUMPTION: data_dir may contain stray files like .DS_Store
        # or README files that should not trigger any warnings.
        (tmp_path / "alpha.GseaPreranked.12345").mkdir()
        (tmp_path / ".DS_Store").touch()
        (tmp_path / "README.txt").touch()

        result = discover_mutant_folders(tmp_path)
        captured = capsys.readouterr()
        assert len(result) == 1
        assert captured.err == "", "Non-GseaPreranked entries should be silently ignored"

    def test_non_gsea_folders_silently_ignored(self, tmp_path, capsys):
        """Folders without .GseaPreranked should be silently ignored (no stderr)."""
        # DATA ASSUMPTION: data_dir may contain other folders like "archive/"
        # that are unrelated to GSEA output.
        (tmp_path / "alpha.GseaPreranked.12345").mkdir()
        (tmp_path / "archive").mkdir()
        (tmp_path / "old_results").mkdir()

        result = discover_mutant_folders(tmp_path)
        captured = capsys.readouterr()
        assert len(result) == 1
        assert captured.err == "", "Non-GseaPreranked folders should be silently ignored"


# ===========================================================================
# Gap 6: ingest_data end-to-end with conflict resolution via merge
# ===========================================================================


class TestIngestDataConflictResolution:
    """Verify that conflict resolution (Contract #10) works correctly
    through the full ingest_data pipeline, not just through merge_pos_neg."""

    def test_ingest_data_resolves_pos_neg_overlap(self, tmp_path):
        """Contract #10: When the same GO term appears in both pos and neg
        files for a single mutant, the entry with the smaller nom_pval is
        retained in the final CohortData."""
        # DATA ASSUMPTION: A term appears in both pos and neg reports for the
        # same mutant. Pos has nom_pval=0.01, neg has nom_pval=0.005.
        # The neg entry (smaller p-value) should be retained.
        pos_row = _make_tsv_row(
            "GO:0000001", "OVERLAPPING_TERM", 50, 0.5, 1.5, 0.01, 0.05, 0.01, 5000
        )
        neg_row = _make_tsv_row(
            "GO:0000001", "OVERLAPPING_TERM", 80, -0.6, -1.8, 0.005, 0.03, 0.005, 7000
        )
        _create_mutant_folder(tmp_path, "alpha", [pos_row], [neg_row])
        _create_mutant_folder(tmp_path, "beta", [SAMPLE_POS_ROW], [SAMPLE_NEG_ROW])

        result = ingest_data(tmp_path)
        rec = result.profiles["alpha"].records["OVERLAPPING_TERM"]
        # Neg has smaller nom_pval (0.005 < 0.01), so neg entry is retained
        assert rec.nom_pval == pytest.approx(0.005)
        assert rec.nes == pytest.approx(-1.8)


# ===========================================================================
# Gap 7: CohortData.profiles values are MutantProfile instances
# ===========================================================================


class TestCohortDataProfileTypes:
    """Verify structural types within CohortData returned by ingest_data."""

    def test_profiles_values_are_mutant_profile_instances(self, tmp_path):
        """Each value in CohortData.profiles should be a MutantProfile."""
        _create_mutant_folder(tmp_path, "alpha", [SAMPLE_POS_ROW], [SAMPLE_NEG_ROW])
        _create_mutant_folder(tmp_path, "beta", [SAMPLE_POS_ROW], [SAMPLE_NEG_ROW])

        result = ingest_data(tmp_path)
        for mid, profile in result.profiles.items():
            assert isinstance(profile, MutantProfile), (
                f"Profile for '{mid}' is {type(profile)}, expected MutantProfile"
            )

    def test_profile_records_values_are_term_record_instances(self, tmp_path):
        """Each value in MutantProfile.records should be a TermRecord."""
        _create_mutant_folder(tmp_path, "alpha", [SAMPLE_POS_ROW], [SAMPLE_NEG_ROW])
        _create_mutant_folder(tmp_path, "beta", [SAMPLE_POS_ROW], [SAMPLE_NEG_ROW])

        result = ingest_data(tmp_path)
        for mid, profile in result.profiles.items():
            for term_name, rec in profile.records.items():
                assert isinstance(rec, TermRecord), (
                    f"Record for term '{term_name}' in mutant '{mid}' is "
                    f"{type(rec)}, expected TermRecord"
                )

    def test_profile_records_keyed_by_term_name(self, tmp_path):
        """MutantProfile.records should be keyed by term_name, and each
        key should match the corresponding TermRecord.term_name."""
        _create_mutant_folder(tmp_path, "alpha", [SAMPLE_POS_ROW], [SAMPLE_NEG_ROW])
        _create_mutant_folder(tmp_path, "beta", [SAMPLE_POS_ROW], [SAMPLE_NEG_ROW])

        result = ingest_data(tmp_path)
        for profile in result.profiles.values():
            for key, rec in profile.records.items():
                assert key == rec.term_name, (
                    f"Record key '{key}' does not match term_name '{rec.term_name}'"
                )


# ===========================================================================
# Gap 8: Contract #14 -- non-matching files in mutant subfolders ignored
#   via locate_report_files (tested directly, not through ingest_data)
# ===========================================================================


class TestLocateReportFilesIgnoresNonMatching:
    """Contract #14: Files in mutant subfolders that do not match either
    the pos or neg glob pattern are silently ignored."""

    def test_extra_files_do_not_affect_result(self, tmp_path):
        """Extra files (HTML, Excel, etc.) in the mutant subfolder should
        not interfere with locating the pos and neg report files."""
        # DATA ASSUMPTION: GSEA output folders often contain additional
        # files like index.html, heatmaps, etc.
        (tmp_path / "gsea_report_for_na_pos_1234.tsv").touch()
        (tmp_path / "gsea_report_for_na_neg_1234.tsv").touch()
        (tmp_path / "index.html").touch()
        (tmp_path / "heatmap.png").touch()
        (tmp_path / "enrichment_results.xls").touch()
        (tmp_path / "gsea_snapshot.zip").touch()

        pos_result, neg_result = locate_report_files(tmp_path, "test")
        assert pos_result.name == "gsea_report_for_na_pos_1234.tsv"
        assert neg_result.name == "gsea_report_for_na_neg_1234.tsv"


# ===========================================================================
# Gap 9: all_term_names and all_go_ids reflect parsed data accurately
# ===========================================================================


class TestCohortDataSetContents:
    """Verify that all_term_names and all_go_ids contain exactly the
    expected values after ingestion."""

    def test_all_term_names_exact_contents(self, tmp_path):
        """Contract #11: all_term_names should contain exactly the union
        of all term names from all mutant profiles."""
        # DATA ASSUMPTION: Two mutants with distinct terms to verify
        # exact set contents.
        _create_mutant_folder(
            tmp_path, "alpha",
            [_make_tsv_row("GO:0000010", "TERM_A", 10, 0.5, 1.5, 0.01, 0.05, 0.01, 1000)],
            [_make_tsv_row("GO:0000011", "TERM_B", 20, -0.4, -1.3, 0.02, 0.08, 0.03, 2000)],
        )
        _create_mutant_folder(
            tmp_path, "beta",
            [_make_tsv_row("GO:0000012", "TERM_C", 15, 0.6, 1.7, 0.005, 0.03, 0.005, 3000)],
            [_make_tsv_row("GO:0000010", "TERM_A", 25, -0.5, -1.6, 0.01, 0.04, 0.01, 4000)],
        )

        result = ingest_data(tmp_path)
        assert result.all_term_names == {"TERM_A", "TERM_B", "TERM_C"}

    def test_all_go_ids_exact_contents(self, tmp_path):
        """Contract #12: all_go_ids should contain exactly the union
        of all GO IDs from all mutant profiles."""
        # DATA ASSUMPTION: Same setup as above, verifying GO IDs.
        _create_mutant_folder(
            tmp_path, "alpha",
            [_make_tsv_row("GO:0000010", "TERM_A", 10, 0.5, 1.5, 0.01, 0.05, 0.01, 1000)],
            [_make_tsv_row("GO:0000011", "TERM_B", 20, -0.4, -1.3, 0.02, 0.08, 0.03, 2000)],
        )
        _create_mutant_folder(
            tmp_path, "beta",
            [_make_tsv_row("GO:0000012", "TERM_C", 15, 0.6, 1.7, 0.005, 0.03, 0.005, 3000)],
            [_make_tsv_row("GO:0000010", "TERM_A", 25, -0.5, -1.6, 0.01, 0.04, 0.01, 4000)],
        )

        result = ingest_data(tmp_path)
        assert result.all_go_ids == {"GO:0000010", "GO:0000011", "GO:0000012"}
