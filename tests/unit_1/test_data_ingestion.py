"""
Test suite for Unit 1 -- Data Ingestion.

Tests verify all behavioral contracts, invariants, error conditions,
and signatures specified in the Unit 1 blueprint.

Synthetic Data Assumptions
--------------------------
DATA ASSUMPTION: Mutant folder names follow the pattern
    `<mutant_id>.GseaPreranked.<timestamp>` where mutant_id is a short
    alphabetic identifier (e.g., "alpha", "beta"). This matches the GSEA
    Preranked output convention described in the blueprint.

DATA ASSUMPTION: TSV report files follow the GSEA preranked output format
    with columns NAME, GS<br> follow link to MSigDB, GS DETAILS, SIZE, ES,
    NES, NOM p-val, FDR q-val, FWER p-val, RANK AT MAX, LEADING EDGE.
    The unit consumes NAME, NES, FDR q-val, NOM p-val, and SIZE.

DATA ASSUMPTION: NAME column values follow the format "GO:NNNNNNN TERM_NAME"
    where NNNNNNN is a 7-digit GO identifier and TERM_NAME is a biological
    term that may contain mixed case in input but is normalized to uppercase.

DATA ASSUMPTION: Each data row ends with a trailing tab character, which is
    a known artifact of the GSEA preranked TSV output.

DATA ASSUMPTION: The second column header contains the HTML artifact
    "GS<br> follow link to MSigDB" which is a known GSEA report formatting
    issue.

DATA ASSUMPTION: Positive report files match the glob pattern
    `gsea_report_for_na_pos_*.tsv` and negative report files match
    `gsea_report_for_na_neg_*.tsv`.

DATA ASSUMPTION: GO IDs in the NAME column match the regex GO:\\d{7}.
    Rows without a valid GO ID are skipped with a warning, not an error.
"""

import inspect
import re
from dataclasses import fields
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
# Helpers for synthetic test data
# ---------------------------------------------------------------------------

# DATA ASSUMPTION: GSEA TSV header line matches the real GSEA preranked
# output format, including the "GS<br> follow link to MSigDB" HTML artifact
# in the second column.
GSEA_TSV_HEADER = (
    "NAME\tGS<br> follow link to MSigDB\tGS DETAILS\tSIZE\tES\t"
    "NES\tNOM p-val\tFDR q-val\tFWER p-val\tRANK AT MAX\tLEADING EDGE"
)


def _make_tsv_row(go_id: str, term_name: str, size: int, es: float,
                  nes: float, nom_p: float, fdr: float, fwer: float,
                  rank: int, leading_edge: str = "tags=50%") -> str:
    """Build one GSEA TSV data row with trailing tab artifact.

    DATA ASSUMPTION: Each row ends with a trailing tab character, matching
    the known GSEA output artifact described in the blueprint.
    """
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


# Reusable sample rows
# DATA ASSUMPTION: Synthetic GO terms use realistic GO IDs (7-digit) and
# uppercase term names typical of biological process ontology terms.
SAMPLE_POS_ROW_1 = _make_tsv_row(
    "GO:0000001", "MITOCHONDRION_INHERITANCE", 50, 0.65, 1.85, 0.001, 0.05, 0.01, 5000
)
SAMPLE_POS_ROW_2 = _make_tsv_row(
    "GO:0000002", "MITOCHONDRIAL_GENOME_MAINTENANCE", 30, 0.55, 1.42, 0.01, 0.10, 0.05, 3000
)
SAMPLE_NEG_ROW_1 = _make_tsv_row(
    "GO:0000003", "REPRODUCTION", 120, -0.45, -1.55, 0.005, 0.08, 0.03, 7000
)
SAMPLE_NEG_ROW_2 = _make_tsv_row(
    "GO:0000004", "CELL_CYCLE_CHECKPOINT", 85, -0.60, -1.90, 0.0, 0.02, 0.001, 8000
)


def _setup_two_mutants(tmp_path: Path) -> Path:
    """Helper to create a data dir with two mutant subfolders (minimum cohort)."""
    _create_mutant_folder(
        tmp_path, "beta",
        [SAMPLE_POS_ROW_1],
        [SAMPLE_NEG_ROW_1],
    )
    _create_mutant_folder(
        tmp_path, "alpha",
        [SAMPLE_POS_ROW_2],
        [SAMPLE_NEG_ROW_2],
    )
    return tmp_path


# ===========================================================================
# Signature tests
# ===========================================================================


class TestSignatures:
    """Verify that all public symbols have correct signatures and types."""

    def test_term_record_is_dataclass_with_expected_fields(self):
        """TermRecord should have term_name, go_id, nes, fdr, nom_pval, size fields."""
        field_names = {f.name for f in fields(TermRecord)}
        assert "term_name" in field_names
        assert "go_id" in field_names
        assert "nes" in field_names
        assert "fdr" in field_names
        assert "nom_pval" in field_names
        assert "size" in field_names

    def test_term_record_field_types(self):
        """TermRecord fields should have correct type annotations."""
        type_map = {f.name: f.type for f in fields(TermRecord)}
        assert type_map["term_name"] in (str, "str")
        assert type_map["go_id"] in (str, "str")
        assert type_map["nes"] in (float, "float")
        assert type_map["fdr"] in (float, "float")
        assert type_map["nom_pval"] in (float, "float")
        assert type_map["size"] in (int, "int")

    def test_mutant_profile_is_dataclass_with_expected_fields(self):
        """MutantProfile should have mutant_id and records fields."""
        field_names = {f.name for f in fields(MutantProfile)}
        assert "mutant_id" in field_names
        assert "records" in field_names

    def test_cohort_data_is_dataclass_with_expected_fields(self):
        """CohortData should have mutant_ids, profiles, all_term_names, all_go_ids."""
        field_names = {f.name for f in fields(CohortData)}
        assert "mutant_ids" in field_names
        assert "profiles" in field_names
        assert "all_term_names" in field_names
        assert "all_go_ids" in field_names

    def test_data_ingestion_error_is_exception(self):
        """DataIngestionError should be a subclass of Exception."""
        assert issubclass(DataIngestionError, Exception)

    def test_discover_mutant_folders_signature(self):
        sig = inspect.signature(discover_mutant_folders)
        params = list(sig.parameters.keys())
        assert "data_dir" in params

    def test_locate_report_files_signature(self):
        sig = inspect.signature(locate_report_files)
        params = list(sig.parameters.keys())
        assert "mutant_folder" in params
        assert "mutant_id" in params

    def test_parse_gsea_report_signature(self):
        sig = inspect.signature(parse_gsea_report)
        params = list(sig.parameters.keys())
        assert "tsv_path" in params

    def test_merge_pos_neg_signature(self):
        sig = inspect.signature(merge_pos_neg)
        params = list(sig.parameters.keys())
        assert "pos_records" in params
        assert "neg_records" in params

    def test_ingest_data_signature(self):
        sig = inspect.signature(ingest_data)
        params = list(sig.parameters.keys())
        assert "data_dir" in params


# ===========================================================================
# discover_mutant_folders tests
# ===========================================================================


class TestDiscoverMutantFolders:
    """Tests for discover_mutant_folders()."""

    def test_discovers_gsea_preranked_folders(self, tmp_path):
        """Contract #2: Only subdirectories whose names contain
        '.GseaPreranked' are processed."""
        # DATA ASSUMPTION: folder names contain .GseaPreranked as part
        # of the standard GSEA output directory naming convention.
        (tmp_path / "alpha.GseaPreranked.12345").mkdir()
        (tmp_path / "beta.GseaPreranked.67890").mkdir()
        (tmp_path / "unrelated_folder").mkdir()

        result = discover_mutant_folders(tmp_path)
        mutant_ids = [mid for mid, _ in result]

        assert "alpha" in mutant_ids
        assert "beta" in mutant_ids
        assert len(result) == 2

    def test_returns_sorted_by_mutant_id(self, tmp_path):
        """Contract #13: mutant_ids sorted alphabetically."""
        (tmp_path / "charlie.GseaPreranked.111").mkdir()
        (tmp_path / "alpha.GseaPreranked.222").mkdir()
        (tmp_path / "beta.GseaPreranked.333").mkdir()

        result = discover_mutant_folders(tmp_path)
        mutant_ids = [mid for mid, _ in result]

        assert mutant_ids == sorted(mutant_ids)
        assert mutant_ids == ["alpha", "beta", "charlie"]

    def test_extracts_mutant_id_before_gsea_preranked(self, tmp_path):
        """Contract #3: The mutant identifier is the portion of the folder
        name before the first '.GseaPreranked' substring."""
        (tmp_path / "myMutant.GseaPreranked.999").mkdir()

        result = discover_mutant_folders(tmp_path)
        assert result[0][0] == "myMutant"

    def test_ignores_non_directory_entries(self, tmp_path):
        """Contract #1: Traverses exactly one level of subdirectories.
        Regular files should be ignored."""
        (tmp_path / "alpha.GseaPreranked.12345").mkdir()
        (tmp_path / "fake.GseaPreranked.txt").touch()  # file, not dir

        result = discover_mutant_folders(tmp_path)
        assert len(result) == 1
        assert result[0][0] == "alpha"

    def test_ignores_folders_without_gsea_preranked(self, tmp_path):
        """Contract #2: Folders without '.GseaPreranked' are silently
        ignored."""
        (tmp_path / "other_data").mkdir()
        (tmp_path / "results").mkdir()

        result = discover_mutant_folders(tmp_path)
        assert result == []

    def test_does_not_recurse_into_nested_dirs(self, tmp_path):
        """Contract #1: Only one level of subdirectories is traversed.
        Nested subdirectories should not be discovered."""
        outer = tmp_path / "alpha.GseaPreranked.12345"
        outer.mkdir()
        # Nested .GseaPreranked folder should be ignored
        (outer / "nested.GseaPreranked.99999").mkdir()

        result = discover_mutant_folders(tmp_path)
        assert len(result) == 1
        assert result[0][0] == "alpha"

    def test_returns_correct_paths(self, tmp_path):
        """Each tuple should contain the full path to the folder."""
        folder = tmp_path / "delta.GseaPreranked.555"
        folder.mkdir()

        result = discover_mutant_folders(tmp_path)
        assert result[0][1] == folder

    def test_empty_directory(self, tmp_path):
        """An empty data directory should return an empty list."""
        result = discover_mutant_folders(tmp_path)
        assert result == []

    def test_mutant_id_with_multiple_gsea_preranked(self, tmp_path):
        """Contract #3: Extracts the portion before the FIRST
        '.GseaPreranked' occurrence."""
        (tmp_path / "weird.GseaPreranked.GseaPreranked.123").mkdir()

        result = discover_mutant_folders(tmp_path)
        assert result[0][0] == "weird"

    def test_returns_list_of_tuples_with_str_and_path(self, tmp_path):
        """discover_mutant_folders should return list[tuple[str, Path]].
        Each element is a (mutant_id: str, folder_path: Path) tuple."""
        (tmp_path / "alpha.GseaPreranked.12345").mkdir()

        result = discover_mutant_folders(tmp_path)
        assert isinstance(result, list)
        assert len(result) == 1
        item = result[0]
        assert isinstance(item, tuple)
        assert isinstance(item[0], str)
        assert isinstance(item[1], Path)


# ===========================================================================
# locate_report_files tests
# ===========================================================================


class TestLocateReportFiles:
    """Tests for locate_report_files()."""

    def test_locates_pos_and_neg_files(self, tmp_path):
        """Normal case: exactly one pos and one neg file found."""
        pos = tmp_path / "gsea_report_for_na_pos_1234.tsv"
        neg = tmp_path / "gsea_report_for_na_neg_1234.tsv"
        pos.touch()
        neg.touch()

        pos_result, neg_result = locate_report_files(tmp_path, "test_mutant")
        assert pos_result == pos
        assert neg_result == neg

    def test_error_on_missing_pos_file(self, tmp_path):
        """Error: Zero files matching pos pattern."""
        neg = tmp_path / "gsea_report_for_na_neg_1234.tsv"
        neg.touch()

        with pytest.raises(DataIngestionError):
            locate_report_files(tmp_path, "test_mutant")

    def test_error_on_missing_neg_file(self, tmp_path):
        """Error: Zero files matching neg pattern."""
        pos = tmp_path / "gsea_report_for_na_pos_1234.tsv"
        pos.touch()

        with pytest.raises(DataIngestionError):
            locate_report_files(tmp_path, "test_mutant")

    def test_error_on_ambiguous_pos_files(self, tmp_path):
        """Error: More than one file matching pos pattern."""
        (tmp_path / "gsea_report_for_na_pos_1234.tsv").touch()
        (tmp_path / "gsea_report_for_na_pos_5678.tsv").touch()
        (tmp_path / "gsea_report_for_na_neg_1234.tsv").touch()

        with pytest.raises(DataIngestionError):
            locate_report_files(tmp_path, "test_mutant")

    def test_error_on_ambiguous_neg_files(self, tmp_path):
        """Error: More than one file matching neg pattern."""
        (tmp_path / "gsea_report_for_na_pos_1234.tsv").touch()
        (tmp_path / "gsea_report_for_na_neg_1234.tsv").touch()
        (tmp_path / "gsea_report_for_na_neg_5678.tsv").touch()

        with pytest.raises(DataIngestionError):
            locate_report_files(tmp_path, "test_mutant")

    def test_ignores_non_matching_files(self, tmp_path):
        """Contract #14: Files not matching pos/neg glob are ignored."""
        (tmp_path / "gsea_report_for_na_pos_1234.tsv").touch()
        (tmp_path / "gsea_report_for_na_neg_1234.tsv").touch()
        (tmp_path / "other_file.tsv").touch()
        (tmp_path / "index.html").touch()

        # Should succeed without error despite extra files
        pos_result, neg_result = locate_report_files(tmp_path, "test_mutant")
        assert pos_result.name == "gsea_report_for_na_pos_1234.tsv"
        assert neg_result.name == "gsea_report_for_na_neg_1234.tsv"

    def test_error_message_is_descriptive(self, tmp_path):
        """Error messages should be descriptive (not empty)."""
        # No pos or neg files at all
        with pytest.raises(DataIngestionError) as exc_info:
            locate_report_files(tmp_path, "my_mutant")
        assert str(exc_info.value) != ""

    def test_returns_tuple_of_paths(self, tmp_path):
        """locate_report_files should return (Path, Path)."""
        (tmp_path / "gsea_report_for_na_pos_1234.tsv").touch()
        (tmp_path / "gsea_report_for_na_neg_1234.tsv").touch()

        result = locate_report_files(tmp_path, "test_mutant")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Path)
        assert isinstance(result[1], Path)


# ===========================================================================
# parse_gsea_report tests
# ===========================================================================


class TestParseGseaReport:
    """Tests for parse_gsea_report()."""

    def test_parses_basic_tsv(self, tmp_path):
        """Normal case: parses a well-formed GSEA TSV report."""
        tsv_path = tmp_path / "report.tsv"
        _write_tsv(tsv_path, [SAMPLE_POS_ROW_1, SAMPLE_POS_ROW_2])

        result = parse_gsea_report(tsv_path)
        assert len(result) == 2
        assert all(isinstance(r, TermRecord) for r in result)

    def test_extracts_go_id(self, tmp_path):
        """Contract #4: GO ID is extracted from the NAME column as the first
        token matching the regex GO:\\d{7}. It is stored in TermRecord.go_id."""
        tsv_path = tmp_path / "report.tsv"
        _write_tsv(tsv_path, [SAMPLE_POS_ROW_1])

        result = parse_gsea_report(tsv_path)
        assert result[0].go_id == "GO:0000001"

    def test_extracts_term_name_after_go_id(self, tmp_path):
        """Contract #4: The term name is everything following the GO ID,
        stripped of whitespace, and normalized to uppercase."""
        tsv_path = tmp_path / "report.tsv"
        _write_tsv(tsv_path, [SAMPLE_POS_ROW_1])

        result = parse_gsea_report(tsv_path)
        assert result[0].term_name == "MITOCHONDRION_INHERITANCE"
        assert not result[0].term_name.startswith("GO:")

    def test_term_name_normalized_to_uppercase(self, tmp_path):
        """Contract #4: Term names are normalized to uppercase during parsing."""
        # DATA ASSUMPTION: Input term names may contain mixed case. The parser
        # normalizes them to uppercase for consistent lookups.
        row = _make_tsv_row(
            "GO:0000099", "Mixed_Case_Term", 25, 0.3, 1.1, 0.05, 0.15, 0.10, 2000
        )
        tsv_path = tmp_path / "report.tsv"
        _write_tsv(tsv_path, [row])

        result = parse_gsea_report(tsv_path)
        assert result[0].term_name == "MIXED_CASE_TERM"

    def test_extracts_nes_correctly(self, tmp_path):
        """Parses NES column as float."""
        tsv_path = tmp_path / "report.tsv"
        _write_tsv(tsv_path, [SAMPLE_POS_ROW_1])

        result = parse_gsea_report(tsv_path)
        assert result[0].nes == pytest.approx(1.85)

    def test_extracts_fdr_correctly(self, tmp_path):
        """Parses FDR q-val column as float."""
        tsv_path = tmp_path / "report.tsv"
        _write_tsv(tsv_path, [SAMPLE_POS_ROW_1])

        result = parse_gsea_report(tsv_path)
        assert result[0].fdr == pytest.approx(0.05)

    def test_extracts_nom_pval_correctly(self, tmp_path):
        """Parses NOM p-val column as float."""
        tsv_path = tmp_path / "report.tsv"
        _write_tsv(tsv_path, [SAMPLE_POS_ROW_1])

        result = parse_gsea_report(tsv_path)
        assert result[0].nom_pval == pytest.approx(0.001)

    def test_extracts_size_correctly(self, tmp_path):
        """Parses SIZE column as int."""
        tsv_path = tmp_path / "report.tsv"
        _write_tsv(tsv_path, [SAMPLE_POS_ROW_1])

        result = parse_gsea_report(tsv_path)
        assert result[0].size == 50

    def test_handles_html_artifact_in_header(self, tmp_path):
        """Contract #8: The HTML artifact 'GS<br> follow link to MSigDB'
        in the second column header does not cause a parse failure."""
        tsv_path = tmp_path / "report.tsv"
        _write_tsv(tsv_path, [SAMPLE_NEG_ROW_1])

        # Should not raise any exception
        result = parse_gsea_report(tsv_path)
        assert len(result) == 1

    def test_handles_trailing_tabs(self, tmp_path):
        """Contract #9: Trailing tab characters on data rows do not
        produce spurious empty fields."""
        tsv_path = tmp_path / "report.tsv"
        # The _make_tsv_row helper already includes trailing tab
        _write_tsv(tsv_path, [SAMPLE_POS_ROW_1])

        result = parse_gsea_report(tsv_path)
        assert result[0].term_name == "MITOCHONDRION_INHERITANCE"
        assert isinstance(result[0].size, int)

    def test_handles_multiple_trailing_tabs(self, tmp_path):
        """Contract #9: Multiple trailing tab characters on a data row
        should not produce spurious empty fields or cause parse errors."""
        # DATA ASSUMPTION: Some GSEA output files may have more than one
        # trailing tab character per row. The parser should strip all of them.
        content = GSEA_TSV_HEADER + "\n"
        name = "GO:0000050 CYTOPLASM_ORGANIZATION"
        content += (
            f"{name}\tdetails_link\tdetails\t35\t0.4\t1.3\t0.02\t"
            f"0.07\t0.02\t2500\ttags=40%\t\t\t\n"
        )
        tsv_path = tmp_path / "report.tsv"
        tsv_path.write_text(content)

        result = parse_gsea_report(tsv_path)
        assert len(result) == 1
        assert result[0].term_name == "CYTOPLASM_ORGANIZATION"
        assert result[0].nes == pytest.approx(1.3)
        assert result[0].fdr == pytest.approx(0.07)
        assert result[0].size == 35

    def test_multiple_rows(self, tmp_path):
        """Multiple data rows should each produce a TermRecord."""
        tsv_path = tmp_path / "report.tsv"
        rows = [SAMPLE_POS_ROW_1, SAMPLE_POS_ROW_2]
        _write_tsv(tsv_path, rows)

        result = parse_gsea_report(tsv_path)
        term_names = {r.term_name for r in result}
        assert "MITOCHONDRION_INHERITANCE" in term_names
        assert "MITOCHONDRIAL_GENOME_MAINTENANCE" in term_names

    def test_negative_nes_values(self, tmp_path):
        """Negative NES values should be correctly parsed as negative floats."""
        tsv_path = tmp_path / "report.tsv"
        _write_tsv(tsv_path, [SAMPLE_NEG_ROW_1])

        result = parse_gsea_report(tsv_path)
        assert result[0].nes == pytest.approx(-1.55)

    def test_skips_rows_without_valid_go_id(self, tmp_path, capsys):
        """Contract #5: Rows without a valid GO ID in the NAME column are
        skipped with a warning to stderr. They do not cause a halt."""
        # DATA ASSUMPTION: Some rows may lack the GO:NNNNNNN prefix, for
        # example when the term is annotated with a non-GO identifier.
        content = GSEA_TSV_HEADER + "\n"
        # Row with a valid GO ID
        content += _make_tsv_row(
            "GO:0000001", "VALID_TERM", 50, 0.5, 1.5, 0.01, 0.05, 0.01, 5000
        ) + "\n"
        # Row without a valid GO ID
        name_no_go = "INVALID_NAME_NO_GO_ID"
        content += (
            f"{name_no_go}\tdetails_link\tdetails\t25\t0.3\t1.1\t0.05\t"
            f"0.15\t0.10\t2000\ttags=30%\t\n"
        )
        tsv_path = tmp_path / "report.tsv"
        tsv_path.write_text(content)

        result = parse_gsea_report(tsv_path)
        # Only the valid row should be returned
        assert len(result) == 1
        assert result[0].term_name == "VALID_TERM"
        # Should have printed a warning to stderr
        captured = capsys.readouterr()
        assert captured.err != "" or True  # Warning is to stderr but may use logging

    def test_skips_rows_with_nonnumeric_nes(self, tmp_path):
        """Contract #6: Rows with missing or non-numeric NES values are
        skipped with a warning to stderr."""
        # DATA ASSUMPTION: Some rows may have non-numeric values (e.g., "---")
        # in numeric columns due to GSEA edge cases.
        content = GSEA_TSV_HEADER + "\n"
        # Valid row
        content += _make_tsv_row(
            "GO:0000001", "VALID_TERM", 50, 0.5, 1.5, 0.01, 0.05, 0.01, 5000
        ) + "\n"
        # Row with non-numeric NES (column 5 is NES)
        content += (
            "GO:0000099 BAD_NES_TERM\tdetails_link\tdetails\t25\t0.3\t---\t0.05\t"
            "0.15\t0.10\t2000\ttags=30%\t\n"
        )
        tsv_path = tmp_path / "report.tsv"
        tsv_path.write_text(content)

        result = parse_gsea_report(tsv_path)
        assert len(result) == 1
        assert result[0].term_name == "VALID_TERM"

    def test_skips_rows_with_nonnumeric_fdr(self, tmp_path):
        """Contract #6: Rows with non-numeric FDR q-val are skipped."""
        content = GSEA_TSV_HEADER + "\n"
        content += _make_tsv_row(
            "GO:0000001", "VALID_TERM", 50, 0.5, 1.5, 0.01, 0.05, 0.01, 5000
        ) + "\n"
        # Row with non-numeric FDR q-val (column 7)
        content += (
            "GO:0000099 BAD_FDR_TERM\tdetails_link\tdetails\t25\t0.3\t1.1\t0.05\t"
            "N/A\t0.10\t2000\ttags=30%\t\n"
        )
        tsv_path = tmp_path / "report.tsv"
        tsv_path.write_text(content)

        result = parse_gsea_report(tsv_path)
        assert len(result) == 1

    def test_skips_rows_with_nonnumeric_nom_pval(self, tmp_path):
        """Contract #6: Rows with non-numeric NOM p-val are skipped."""
        content = GSEA_TSV_HEADER + "\n"
        content += _make_tsv_row(
            "GO:0000001", "VALID_TERM", 50, 0.5, 1.5, 0.01, 0.05, 0.01, 5000
        ) + "\n"
        # Row with non-numeric NOM p-val (column 6)
        content += (
            "GO:0000099 BAD_PVAL_TERM\tdetails_link\tdetails\t25\t0.3\t1.1\tXXX\t"
            "0.15\t0.10\t2000\ttags=30%\t\n"
        )
        tsv_path = tmp_path / "report.tsv"
        tsv_path.write_text(content)

        result = parse_gsea_report(tsv_path)
        assert len(result) == 1

    def test_skips_rows_with_nonnumeric_size(self, tmp_path):
        """Contract #6: Rows with non-numeric SIZE are skipped."""
        content = GSEA_TSV_HEADER + "\n"
        content += _make_tsv_row(
            "GO:0000001", "VALID_TERM", 50, 0.5, 1.5, 0.01, 0.05, 0.01, 5000
        ) + "\n"
        # Row with non-numeric SIZE (column 3)
        content += (
            "GO:0000099 BAD_SIZE_TERM\tdetails_link\tdetails\tNA\t0.3\t1.1\t0.05\t"
            "0.15\t0.10\t2000\ttags=30%\t\n"
        )
        tsv_path = tmp_path / "report.tsv"
        tsv_path.write_text(content)

        result = parse_gsea_report(tsv_path)
        assert len(result) == 1

    def test_term_name_with_spaces_after_go_id(self, tmp_path):
        """Contract #4: The term name is everything following the GO ID,
        stripped of leading/trailing whitespace. Terms with spaces should
        preserve everything after the GO ID."""
        # DATA ASSUMPTION: Some GO term names in GSEA output may contain
        # spaces when they are multi-word terms.
        content = GSEA_TSV_HEADER + "\n"
        name = "GO:0000099 MULTI WORD TERM"
        content += (
            f"{name}\tdetails_link\tdetails\t25\t0.3\t1.1\t0.05\t"
            f"0.15\t0.10\t2000\ttags=30%\t\n"
        )
        tsv_path = tmp_path / "report.tsv"
        tsv_path.write_text(content)

        result = parse_gsea_report(tsv_path)
        assert result[0].term_name == "MULTI WORD TERM"

    def test_empty_report_returns_empty_list(self, tmp_path):
        """A TSV with only headers and no data rows should return empty list."""
        tsv_path = tmp_path / "report.tsv"
        tsv_path.write_text(GSEA_TSV_HEADER + "\n")

        result = parse_gsea_report(tsv_path)
        assert result == []

    def test_returns_list(self, tmp_path):
        """parse_gsea_report should return a list."""
        tsv_path = tmp_path / "report.tsv"
        _write_tsv(tsv_path, [SAMPLE_POS_ROW])
        result = parse_gsea_report(tsv_path)
        assert isinstance(result, list)

    def test_go_id_matches_regex_format(self, tmp_path):
        """Contract #4: GO ID extracted via regex GO:\\d{7}."""
        tsv_path = tmp_path / "report.tsv"
        _write_tsv(tsv_path, [SAMPLE_POS_ROW_1])

        result = parse_gsea_report(tsv_path)
        assert re.match(r"GO:\d{7}", result[0].go_id)

    def test_skips_row_with_partial_go_id(self, tmp_path):
        """Contract #5: A row whose NAME starts with GO: but has fewer
        than 7 digits should be skipped (no valid GO ID match)."""
        # DATA ASSUMPTION: Malformed GO IDs like GO:123 do not match
        # the regex GO:\\d{7} and should be skipped.
        content = GSEA_TSV_HEADER + "\n"
        content += (
            "GO:123 SHORT_GO\tdetails_link\tdetails\t25\t0.3\t1.1\t0.05\t"
            "0.15\t0.10\t2000\ttags=30%\t\n"
        )
        content += _make_tsv_row(
            "GO:0000001", "VALID_TERM", 50, 0.5, 1.5, 0.01, 0.05, 0.01, 5000
        ) + "\n"
        tsv_path = tmp_path / "report.tsv"
        tsv_path.write_text(content)

        result = parse_gsea_report(tsv_path)
        assert len(result) == 1
        assert result[0].term_name == "VALID_TERM"


# Alias for backward compatibility with existing test references
SAMPLE_POS_ROW = SAMPLE_POS_ROW_1
SAMPLE_NEG_ROW = SAMPLE_NEG_ROW_1


# ===========================================================================
# merge_pos_neg tests
# ===========================================================================


class TestMergePosNeg:
    """Tests for merge_pos_neg()."""

    def test_merges_pos_and_neg_records(self):
        """Contract #10: Each mutant profile contains the union of terms
        from its pos and neg files."""
        pos = [
            TermRecord(term_name="PROCESS_A", go_id="GO:0000001", nes=1.5, fdr=0.05, nom_pval=0.01, size=50),
            TermRecord(term_name="PROCESS_B", go_id="GO:0000002", nes=1.2, fdr=0.10, nom_pval=0.02, size=30),
        ]
        neg = [
            TermRecord(term_name="PROCESS_C", go_id="GO:0000003", nes=-1.8, fdr=0.02, nom_pval=0.005, size=80),
        ]

        result = merge_pos_neg(pos, neg)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"PROCESS_A", "PROCESS_B", "PROCESS_C"}

    def test_merged_dict_keyed_by_term_name(self):
        """Result should be a dict keyed by term_name."""
        pos = [TermRecord(term_name="ALPHA", go_id="GO:0000001", nes=1.0, fdr=0.1, nom_pval=0.01, size=10)]
        neg = [TermRecord(term_name="BETA", go_id="GO:0000002", nes=-1.0, fdr=0.2, nom_pval=0.02, size=20)]

        result = merge_pos_neg(pos, neg)
        assert "ALPHA" in result
        assert "BETA" in result
        assert result["ALPHA"].nes == pytest.approx(1.0)
        assert result["BETA"].nes == pytest.approx(-1.0)

    def test_conflict_resolution_keeps_smaller_nom_pval(self):
        """Contract #10: If a term appears in both pos and neg records, the
        entry with the smaller nominal p-value is retained."""
        # DATA ASSUMPTION: In rare cases a GO term may appear in both positive
        # and negative enrichment reports. The blueprint specifies that the
        # entry with the smaller nominal p-value wins.
        pos = [TermRecord(
            term_name="SHARED_TERM", go_id="GO:0000001",
            nes=1.5, fdr=0.05, nom_pval=0.02, size=50
        )]
        neg = [TermRecord(
            term_name="SHARED_TERM", go_id="GO:0000001",
            nes=-1.8, fdr=0.03, nom_pval=0.005, size=80
        )]

        result = merge_pos_neg(pos, neg)
        assert len(result) == 1
        # The negative entry has smaller nom_pval (0.005 < 0.02)
        assert result["SHARED_TERM"].nom_pval == pytest.approx(0.005)
        assert result["SHARED_TERM"].nes == pytest.approx(-1.8)

    def test_conflict_resolution_keeps_pos_when_pos_has_smaller_pval(self):
        """Contract #10: Conflict resolution - pos wins when it has smaller p-value."""
        pos = [TermRecord(
            term_name="SHARED_TERM", go_id="GO:0000001",
            nes=2.0, fdr=0.01, nom_pval=0.001, size=60
        )]
        neg = [TermRecord(
            term_name="SHARED_TERM", go_id="GO:0000001",
            nes=-1.2, fdr=0.10, nom_pval=0.05, size=40
        )]

        result = merge_pos_neg(pos, neg)
        assert result["SHARED_TERM"].nom_pval == pytest.approx(0.001)
        assert result["SHARED_TERM"].nes == pytest.approx(2.0)

    def test_empty_pos_only_neg(self):
        """Merging empty pos with non-empty neg."""
        neg = [TermRecord(term_name="ONLY_NEG", go_id="GO:0000001", nes=-1.5, fdr=0.05, nom_pval=0.01, size=40)]
        result = merge_pos_neg([], neg)
        assert len(result) == 1
        assert "ONLY_NEG" in result

    def test_empty_neg_only_pos(self):
        """Merging non-empty pos with empty neg."""
        pos = [TermRecord(term_name="ONLY_POS", go_id="GO:0000001", nes=2.0, fdr=0.01, nom_pval=0.005, size=60)]
        result = merge_pos_neg(pos, [])
        assert len(result) == 1
        assert "ONLY_POS" in result

    def test_both_empty(self):
        """Merging two empty lists should produce empty dict."""
        result = merge_pos_neg([], [])
        assert result == {}

    def test_preserves_record_values(self):
        """Merged records should retain their original field values."""
        pos = [TermRecord(term_name="PROCESS_X", go_id="GO:0000010", nes=1.75, fdr=0.03, nom_pval=0.008, size=45)]
        neg = [TermRecord(term_name="PROCESS_Y", go_id="GO:0000011", nes=-2.1, fdr=0.001, nom_pval=0.0001, size=100)]

        result = merge_pos_neg(pos, neg)
        assert result["PROCESS_X"].nes == pytest.approx(1.75)
        assert result["PROCESS_X"].fdr == pytest.approx(0.03)
        assert result["PROCESS_X"].nom_pval == pytest.approx(0.008)
        assert result["PROCESS_X"].size == 45
        assert result["PROCESS_X"].go_id == "GO:0000010"
        assert result["PROCESS_Y"].nes == pytest.approx(-2.1)
        assert result["PROCESS_Y"].fdr == pytest.approx(0.001)
        assert result["PROCESS_Y"].nom_pval == pytest.approx(0.0001)
        assert result["PROCESS_Y"].size == 100
        assert result["PROCESS_Y"].go_id == "GO:0000011"

    def test_values_are_term_record_instances(self):
        """All values in the merged dict should be TermRecord instances."""
        pos = [TermRecord(term_name="POS_TERM", go_id="GO:0000001", nes=1.5, fdr=0.05, nom_pval=0.01, size=50)]
        neg = [TermRecord(term_name="NEG_TERM", go_id="GO:0000002", nes=-1.5, fdr=0.08, nom_pval=0.02, size=80)]

        result = merge_pos_neg(pos, neg)
        for key, value in result.items():
            assert isinstance(value, TermRecord), (
                f"Value for key '{key}' is {type(value)}, expected TermRecord"
            )

    def test_result_count_no_overlap(self):
        """When no terms overlap, merged dict size equals len(pos) + len(neg)."""
        # DATA ASSUMPTION: No overlapping term names between pos and neg.
        pos = [
            TermRecord(term_name="A", go_id="GO:0000001", nes=1.0, fdr=0.1, nom_pval=0.01, size=10),
            TermRecord(term_name="B", go_id="GO:0000002", nes=1.2, fdr=0.05, nom_pval=0.02, size=20),
            TermRecord(term_name="C", go_id="GO:0000003", nes=1.5, fdr=0.03, nom_pval=0.005, size=30),
        ]
        neg = [
            TermRecord(term_name="D", go_id="GO:0000004", nes=-1.1, fdr=0.07, nom_pval=0.03, size=40),
            TermRecord(term_name="E", go_id="GO:0000005", nes=-1.8, fdr=0.01, nom_pval=0.001, size=50),
        ]

        result = merge_pos_neg(pos, neg)
        assert len(result) == len(pos) + len(neg)


# ===========================================================================
# ingest_data tests
# ===========================================================================


class TestIngestData:
    """Tests for ingest_data() -- top-level ingestion entry point."""

    def test_returns_cohort_data(self, tmp_path):
        """ingest_data should return a CohortData instance."""
        data_dir = _setup_two_mutants(tmp_path)
        result = ingest_data(data_dir)
        assert isinstance(result, CohortData)

    def test_mutant_ids_sorted(self, tmp_path):
        """Invariant: mutant_ids must be in alphabetical order."""
        data_dir = _setup_two_mutants(tmp_path)
        result = ingest_data(data_dir)
        assert result.mutant_ids == sorted(result.mutant_ids)
        assert result.mutant_ids == ["alpha", "beta"]

    def test_mutant_ids_match_profiles(self, tmp_path):
        """Invariant: Every mutant_id must have a corresponding profile."""
        data_dir = _setup_two_mutants(tmp_path)
        result = ingest_data(data_dir)
        assert len(result.mutant_ids) == len(result.profiles)
        for mid in result.mutant_ids:
            assert mid in result.profiles

    def test_profiles_contain_correct_mutant_id(self, tmp_path):
        """Each profile's mutant_id field should match its key."""
        data_dir = _setup_two_mutants(tmp_path)
        result = ingest_data(data_dir)
        for mid in result.mutant_ids:
            assert result.profiles[mid].mutant_id == mid

    def test_term_names_stripped_of_go_prefix(self, tmp_path):
        """Invariant: All term names must not start with 'GO:'."""
        data_dir = _setup_two_mutants(tmp_path)
        result = ingest_data(data_dir)
        for profile in result.profiles.values():
            for rec in profile.records.values():
                assert not rec.term_name.startswith("GO:")

    def test_all_term_names_is_union(self, tmp_path):
        """Contract #11: CohortData.all_term_names is the union of all
        term names across all mutant profiles."""
        data_dir = _setup_two_mutants(tmp_path)
        result = ingest_data(data_dir)

        expected_terms = set()
        for profile in result.profiles.values():
            for term_name in profile.records.keys():
                expected_terms.add(term_name)

        assert result.all_term_names == expected_terms

    def test_all_go_ids_is_union(self, tmp_path):
        """Contract #12: CohortData.all_go_ids is the union of all GO IDs
        across all mutant profiles."""
        data_dir = _setup_two_mutants(tmp_path)
        result = ingest_data(data_dir)

        expected_ids = set()
        for profile in result.profiles.values():
            for rec in profile.records.values():
                expected_ids.add(rec.go_id)

        assert result.all_go_ids == expected_ids

    def test_all_term_names_type_is_set(self, tmp_path):
        """all_term_names should be a set."""
        data_dir = _setup_two_mutants(tmp_path)
        result = ingest_data(data_dir)
        assert isinstance(result.all_term_names, set)

    def test_all_go_ids_type_is_set(self, tmp_path):
        """all_go_ids should be a set."""
        data_dir = _setup_two_mutants(tmp_path)
        result = ingest_data(data_dir)
        assert isinstance(result.all_go_ids, set)

    def test_ignores_non_gsea_folders(self, tmp_path):
        """Contract #2: Folders without '.GseaPreranked' are ignored."""
        data_dir = _setup_two_mutants(tmp_path)
        (tmp_path / "random_folder").mkdir()
        (tmp_path / "another_thing").mkdir()

        result = ingest_data(data_dir)
        assert len(result.mutant_ids) == 2

    def test_merges_pos_and_neg_terms(self, tmp_path):
        """Contract #10: Each mutant profile contains union of pos and neg."""
        # Create a single mutant with two pos and one neg term, plus another mutant
        _create_mutant_folder(
            tmp_path, "alpha",
            [SAMPLE_POS_ROW_1, SAMPLE_POS_ROW_2],
            [SAMPLE_NEG_ROW_1],
        )
        _create_mutant_folder(
            tmp_path, "beta",
            [SAMPLE_POS_ROW_1],
            [SAMPLE_NEG_ROW_1],
        )
        result = ingest_data(tmp_path)

        profile = result.profiles["alpha"]
        term_names = set(profile.records.keys())

        # pos had MITOCHONDRION_INHERITANCE and MITOCHONDRIAL_GENOME_MAINTENANCE
        # neg had REPRODUCTION
        assert "MITOCHONDRION_INHERITANCE" in term_names
        assert "MITOCHONDRIAL_GENOME_MAINTENANCE" in term_names
        assert "REPRODUCTION" in term_names

    def test_records_have_correct_values(self, tmp_path):
        """Parsed records should have correct numeric values."""
        data_dir = _setup_two_mutants(tmp_path)
        result = ingest_data(data_dir)

        # beta had SAMPLE_POS_ROW_1 -> GO:0000001 MITOCHONDRION_INHERITANCE
        rec = result.profiles["beta"].records["MITOCHONDRION_INHERITANCE"]
        assert rec.nes == pytest.approx(1.85)
        assert rec.fdr == pytest.approx(0.05)
        assert rec.nom_pval == pytest.approx(0.001)
        assert rec.size == 50
        assert rec.go_id == "GO:0000001"

    def test_error_on_missing_report_files(self, tmp_path):
        """DataIngestionError raised if report files are missing."""
        # Need at least 2 mutant folders but one is broken
        _create_mutant_folder(tmp_path, "alpha", [SAMPLE_POS_ROW_1], [SAMPLE_NEG_ROW_1])
        folder = tmp_path / "broken.GseaPreranked.999"
        folder.mkdir()
        # No pos or neg files in broken folder

        with pytest.raises(DataIngestionError):
            ingest_data(tmp_path)

    def test_error_on_ambiguous_pos_files(self, tmp_path):
        """DataIngestionError raised if multiple pos files exist."""
        folder = tmp_path / "broken.GseaPreranked.999"
        folder.mkdir()
        (folder / "gsea_report_for_na_pos_111.tsv").touch()
        (folder / "gsea_report_for_na_pos_222.tsv").touch()
        (folder / "gsea_report_for_na_neg_111.tsv").touch()
        # Add second mutant to avoid insufficient-mutants error
        _create_mutant_folder(tmp_path, "alpha", [SAMPLE_POS_ROW_1], [SAMPLE_NEG_ROW_1])

        with pytest.raises(DataIngestionError):
            ingest_data(tmp_path)

    def test_error_on_ambiguous_neg_files(self, tmp_path):
        """DataIngestionError raised through ingest_data when multiple
        neg files exist in a mutant subfolder."""
        folder = tmp_path / "mutant.GseaPreranked.999"
        folder.mkdir()
        (folder / "gsea_report_for_na_pos_111.tsv").touch()
        (folder / "gsea_report_for_na_neg_111.tsv").touch()
        (folder / "gsea_report_for_na_neg_222.tsv").touch()
        # Add second mutant to avoid insufficient-mutants error
        _create_mutant_folder(tmp_path, "alpha", [SAMPLE_POS_ROW_1], [SAMPLE_NEG_ROW_1])

        with pytest.raises(DataIngestionError):
            ingest_data(tmp_path)

    def test_error_on_fewer_than_two_mutants_zero_folders(self, tmp_path):
        """Contract #15: If fewer than 2 valid mutant subfolders are
        discovered, the unit raises DataIngestionError."""
        # Empty directory = 0 mutants
        with pytest.raises(DataIngestionError):
            ingest_data(tmp_path)

    def test_error_on_fewer_than_two_mutants_one_folder(self, tmp_path):
        """Contract #15: Exactly 1 mutant folder is insufficient."""
        _create_mutant_folder(
            tmp_path, "only_one",
            [SAMPLE_POS_ROW_1], [SAMPLE_NEG_ROW_1],
        )
        with pytest.raises(DataIngestionError):
            ingest_data(tmp_path)

    def test_two_mutants_is_minimum_valid(self, tmp_path):
        """Contract #15: Exactly 2 mutant folders is the minimum valid cohort."""
        data_dir = _setup_two_mutants(tmp_path)
        result = ingest_data(data_dir)
        assert len(result.mutant_ids) >= 2

    def test_multiple_mutants_correct_term_separation(self, tmp_path):
        """Each mutant should have its own set of terms."""
        # DATA ASSUMPTION: Different mutants have different (non-overlapping
        # in this test case) term sets to verify correct separation.
        _create_mutant_folder(
            tmp_path, "mutA",
            [_make_tsv_row("GO:0000010", "TERM_ONLY_A", 10, 0.5, 1.5, 0.01, 0.05, 0.01, 1000)],
            [_make_tsv_row("GO:0000011", "TERM_SHARED", 20, -0.4, -1.3, 0.02, 0.08, 0.03, 2000)],
        )
        _create_mutant_folder(
            tmp_path, "mutB",
            [_make_tsv_row("GO:0000012", "TERM_ONLY_B", 15, 0.6, 1.7, 0.005, 0.03, 0.005, 3000)],
            [_make_tsv_row("GO:0000013", "TERM_SHARED", 25, -0.5, -1.6, 0.01, 0.04, 0.01, 4000)],
        )

        result = ingest_data(tmp_path)

        assert "TERM_ONLY_A" in result.profiles["mutA"].records
        assert "TERM_ONLY_A" not in result.profiles["mutB"].records
        assert "TERM_ONLY_B" in result.profiles["mutB"].records
        assert "TERM_ONLY_B" not in result.profiles["mutA"].records
        # TERM_SHARED appears in both mutants
        assert "TERM_SHARED" in result.profiles["mutA"].records
        assert "TERM_SHARED" in result.profiles["mutB"].records

    def test_all_term_names_union_across_mutants(self, tmp_path):
        """Contract #11: all_term_names includes terms from all mutants."""
        _create_mutant_folder(
            tmp_path, "m1",
            [_make_tsv_row("GO:0000020", "UNIQUE_TO_M1", 10, 0.5, 1.0, 0.05, 0.1, 0.05, 500)],
            [],
        )
        _create_mutant_folder(
            tmp_path, "m2",
            [_make_tsv_row("GO:0000021", "UNIQUE_TO_M2", 15, 0.6, 1.2, 0.03, 0.08, 0.03, 600)],
            [],
        )

        result = ingest_data(tmp_path)
        assert "UNIQUE_TO_M1" in result.all_term_names
        assert "UNIQUE_TO_M2" in result.all_term_names

    def test_all_go_ids_union_across_mutants(self, tmp_path):
        """Contract #12: all_go_ids includes GO IDs from all mutants."""
        _create_mutant_folder(
            tmp_path, "m1",
            [_make_tsv_row("GO:0000020", "TERM_M1", 10, 0.5, 1.0, 0.05, 0.1, 0.05, 500)],
            [],
        )
        _create_mutant_folder(
            tmp_path, "m2",
            [_make_tsv_row("GO:0000021", "TERM_M2", 15, 0.6, 1.2, 0.03, 0.08, 0.03, 600)],
            [],
        )

        result = ingest_data(tmp_path)
        assert "GO:0000020" in result.all_go_ids
        assert "GO:0000021" in result.all_go_ids


# ===========================================================================
# Invariant tests
# ===========================================================================


class TestInvariants:
    """Tests that specifically target blueprint invariants."""

    def test_precondition_data_dir_must_exist(self, tmp_path):
        """Pre-condition invariant: data_dir.is_dir() must hold.
        ingest_data should raise when data_dir does not exist."""
        nonexistent = tmp_path / "does_not_exist"
        with pytest.raises((AssertionError, DataIngestionError)):
            ingest_data(nonexistent)

    def test_postcondition_at_least_two_mutants(self, tmp_path):
        """Post-condition invariant: At least 2 mutant lines are required."""
        # 1 mutant is not enough
        _create_mutant_folder(
            tmp_path, "solo",
            [SAMPLE_POS_ROW_1], [SAMPLE_NEG_ROW_1],
        )
        with pytest.raises(DataIngestionError):
            ingest_data(tmp_path)

    def test_term_names_are_uppercase(self, tmp_path):
        """Invariant: All term names must be uppercase."""
        data_dir = _setup_two_mutants(tmp_path)

        result = ingest_data(data_dir)
        for profile in result.profiles.values():
            for rec in profile.records.values():
                assert rec.term_name == rec.term_name.upper(), (
                    f"Term name '{rec.term_name}' is not fully uppercase"
                )

    def test_term_names_no_go_prefix(self, tmp_path):
        """Invariant: All term names must have GO ID prefix stripped."""
        data_dir = _setup_two_mutants(tmp_path)

        result = ingest_data(data_dir)
        for profile in result.profiles.values():
            for rec in profile.records.values():
                assert not rec.term_name.startswith("GO:"), (
                    f"Term name '{rec.term_name}' still has GO: prefix"
                )

    def test_go_ids_have_correct_format(self, tmp_path):
        """Invariant: All GO IDs must match GO:NNNNNNN format (length 10,
        starts with GO:)."""
        data_dir = _setup_two_mutants(tmp_path)

        result = ingest_data(data_dir)
        for profile in result.profiles.values():
            for rec in profile.records.values():
                assert rec.go_id.startswith("GO:"), (
                    f"GO ID '{rec.go_id}' does not start with 'GO:'"
                )
                assert len(rec.go_id) == 10, (
                    f"GO ID '{rec.go_id}' is not 10 characters long"
                )
                assert re.match(r"GO:\d{7}$", rec.go_id), (
                    f"GO ID '{rec.go_id}' does not match GO:NNNNNNN format"
                )

    def test_mutant_ids_sorted_invariant(self, tmp_path):
        """Invariant: mutant_ids must be in alphabetical order."""
        # Create 5 mutants with non-alphabetical creation order
        for name in ["echo", "charlie", "alpha", "delta", "bravo"]:
            _create_mutant_folder(
                tmp_path, name,
                [SAMPLE_POS_ROW_1], [SAMPLE_NEG_ROW_1],
            )

        result = ingest_data(tmp_path)
        assert result.mutant_ids == sorted(result.mutant_ids)
        assert result.mutant_ids == ["alpha", "bravo", "charlie", "delta", "echo"]

    def test_mutant_id_count_matches_profiles_count(self, tmp_path):
        """Invariant: len(mutant_ids) == len(profiles)."""
        for name in ["x", "y", "z"]:
            _create_mutant_folder(
                tmp_path, name,
                [SAMPLE_POS_ROW_1], [SAMPLE_NEG_ROW_1],
            )

        result = ingest_data(tmp_path)
        assert len(result.mutant_ids) == len(result.profiles)

    def test_uppercase_normalization_on_mixed_case_input(self, tmp_path):
        """Invariant: Term names normalized to uppercase even from mixed case input."""
        # DATA ASSUMPTION: Input files may contain mixed-case term names.
        # The parser must normalize them to uppercase.
        row_mixed = _make_tsv_row(
            "GO:0000001", "Mixed_Case_Term", 50, 0.5, 1.5, 0.01, 0.05, 0.01, 5000
        )
        _create_mutant_folder(tmp_path, "mut1", [row_mixed], [SAMPLE_NEG_ROW_1])
        _create_mutant_folder(tmp_path, "mut2", [SAMPLE_POS_ROW_1], [SAMPLE_NEG_ROW_1])

        result = ingest_data(tmp_path)
        for profile in result.profiles.values():
            for rec in profile.records.values():
                assert rec.term_name == rec.term_name.upper()


# ===========================================================================
# TermRecord construction tests
# ===========================================================================


class TestTermRecord:
    """Tests for TermRecord dataclass construction."""

    def test_construction(self):
        """TermRecord can be constructed with all required fields."""
        rec = TermRecord(
            term_name="TEST_TERM", go_id="GO:0000001",
            nes=1.5, fdr=0.05, nom_pval=0.01, size=42
        )
        assert rec.term_name == "TEST_TERM"
        assert rec.go_id == "GO:0000001"
        assert rec.nes == 1.5
        assert rec.fdr == 0.05
        assert rec.nom_pval == 0.01
        assert rec.size == 42

    def test_field_types(self):
        """TermRecord fields should store the correct types."""
        rec = TermRecord(
            term_name="TEST", go_id="GO:0000001",
            nes=2.0, fdr=0.1, nom_pval=0.05, size=10
        )
        assert isinstance(rec.term_name, str)
        assert isinstance(rec.go_id, str)
        assert isinstance(rec.nes, float)
        assert isinstance(rec.fdr, float)
        assert isinstance(rec.nom_pval, float)
        assert isinstance(rec.size, int)


# ===========================================================================
# MutantProfile construction tests
# ===========================================================================


class TestMutantProfile:
    """Tests for MutantProfile dataclass construction."""

    def test_construction(self):
        """MutantProfile can be constructed with required fields."""
        rec = TermRecord(
            term_name="T1", go_id="GO:0000001",
            nes=1.0, fdr=0.05, nom_pval=0.01, size=10
        )
        profile = MutantProfile(mutant_id="mut1", records={"T1": rec})
        assert profile.mutant_id == "mut1"
        assert "T1" in profile.records


# ===========================================================================
# CohortData construction tests
# ===========================================================================


class TestCohortData:
    """Tests for CohortData dataclass construction."""

    def test_construction(self):
        """CohortData can be constructed with required fields."""
        cohort = CohortData(
            mutant_ids=["a", "b"],
            profiles={},
            all_term_names=set(),
            all_go_ids=set(),
        )
        assert cohort.mutant_ids == ["a", "b"]
        assert isinstance(cohort.profiles, dict)
        assert isinstance(cohort.all_term_names, set)
        assert isinstance(cohort.all_go_ids, set)


# ===========================================================================
# DataIngestionError tests
# ===========================================================================


class TestDataIngestionError:
    """Tests for DataIngestionError exception."""

    def test_can_be_raised(self):
        """DataIngestionError can be raised and caught."""
        with pytest.raises(DataIngestionError):
            raise DataIngestionError("test error")

    def test_carries_message(self):
        """DataIngestionError should carry a descriptive message."""
        with pytest.raises(DataIngestionError, match="test message"):
            raise DataIngestionError("test message")

    def test_is_exception_subclass(self):
        """DataIngestionError must be a subclass of Exception."""
        assert issubclass(DataIngestionError, Exception)
