"""
Acceptance tests for the architect-target thinking token analysis pipeline.

These tests are written BEFORE implementation and are expected to FAIL until
the following production files exist:
  - evaluation/scripts/analyze_architect_target_thinking_tokens.py  (Subtask 3)
  - evaluation/scripts/validate_architect_target_tokens.py          (Subtask 4)
  - evaluation/eval_deepseek_judge/analysis/architect_target_thinking_tokens/
      per_row_iteration_long.csv
      aggregate_by_run.csv
      manifest.json
      DATA_LIMITATIONS.md

Tests are organized into four groups:
  1. Schema tests — verify output column names and manifest fields
  2. CLI tests    — verify script existence and argument acceptance
  3. Output validation tests — verify data correctness (skipped if outputs absent)
  4. Validation script tests — verify the validate script exists and works

All tests are deterministic, isolated (no live API calls, no live model runs),
and tagged with @pytest.mark.unit or @pytest.mark.skipif as appropriate.

Run with:
    cd /home/md2292/inception-eval && uv run python -m pytest tests/test_architect_target_token_analysis.py -v
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
_EVAL_SCRIPTS = _PROJECT_ROOT / "evaluation" / "scripts"
_ANALYSIS_DIR = (
    _PROJECT_ROOT
    / "evaluation"
    / "eval_deepseek_judge"
    / "analysis"
    / "architect_target_thinking_tokens"
)

_ANALYZE_SCRIPT = _EVAL_SCRIPTS / "analyze_architect_target_thinking_tokens.py"
_VALIDATE_SCRIPT = _EVAL_SCRIPTS / "validate_architect_target_tokens.py"

_PER_ROW_CSV = _ANALYSIS_DIR / "per_row_iteration_long.csv"
_AGGREGATE_CSV = _ANALYSIS_DIR / "aggregate_by_run.csv"
_MANIFEST_JSON = _ANALYSIS_DIR / "manifest.json"
_DATA_LIMITATIONS_MD = _ANALYSIS_DIR / "DATA_LIMITATIONS.md"

# ---------------------------------------------------------------------------
# Conditions used in skipif markers
# ---------------------------------------------------------------------------

_outputs_exist = (
    _PER_ROW_CSV.exists()
    and _AGGREGATE_CSV.exists()
    and _MANIFEST_JSON.exists()
    and _DATA_LIMITATIONS_MD.exists()
)

_per_row_csv_exists = _PER_ROW_CSV.exists()
_aggregate_csv_exists = _AGGREGATE_CSV.exists()
_manifest_json_exists = _MANIFEST_JSON.exists()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_script(script: Path, *args: str) -> subprocess.CompletedProcess:
    """Run a Python script via uv in the project root and return the result."""
    return subprocess.run(
        ["uv", "run", "python", str(script), *args],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )


def _load_per_row_csv():
    """Load per_row_iteration_long.csv as a pandas DataFrame (lazy import)."""
    import pandas as pd
    return pd.read_csv(_PER_ROW_CSV)


def _load_aggregate_csv():
    """Load aggregate_by_run.csv as a pandas DataFrame (lazy import)."""
    import pandas as pd
    return pd.read_csv(_AGGREGATE_CSV)


# ===========================================================================
# 1. Schema tests
# ===========================================================================

class TestPerRowCsvSchema:
    """
    Verify that per_row_iteration_long.csv has all required columns.

    These tests fail immediately if the CSV does not exist.
    Once the pipeline runs and produces output, they verify schema compliance.
    """

    # Required columns as frozen in the plan (Subtask 1 schema definition)
    REQUIRED_COLUMNS = [
        "branch",
        "run_label",
        "model_name",
        "df_index",
        "category",
        "iteration",
        "role",
        "target_thinking_tokens",
        "architect_post_think_tail_tokens",
        "chars",
        "had_think_close_delimiter",
        "source_pickle",
    ]

    @pytest.mark.unit
    def test_PerRowCsv_FileExists(self):
        # Arrange / Act / Assert
        assert _PER_ROW_CSV.exists(), (
            f"per_row_iteration_long.csv does not exist at:\n  {_PER_ROW_CSV}\n"
            f"Run the CLI pipeline to generate it:\n"
            f"  uv run python evaluation/scripts/analyze_architect_target_thinking_tokens.py"
        )

    @pytest.mark.skipif(
        not _per_row_csv_exists,
        reason="per_row_iteration_long.csv not yet generated — run the CLI pipeline first",
    )
    @pytest.mark.unit
    def test_PerRowCsv_HasAllRequiredColumns(self):
        # Arrange
        import pandas as pd
        df = pd.read_csv(_PER_ROW_CSV, nrows=0)  # header only for speed

        # Act
        actual_columns = set(df.columns.tolist())
        missing = set(self.REQUIRED_COLUMNS) - actual_columns

        # Assert
        assert not missing, (
            f"per_row_iteration_long.csv is missing required columns:\n"
            f"  Missing: {sorted(missing)}\n"
            f"  Present: {sorted(actual_columns)}\n"
            f"  Required: {sorted(self.REQUIRED_COLUMNS)}"
        )

    @pytest.mark.skipif(
        not _per_row_csv_exists,
        reason="per_row_iteration_long.csv not yet generated",
    )
    @pytest.mark.unit
    @pytest.mark.parametrize("col", REQUIRED_COLUMNS)
    def test_PerRowCsv_HasColumn(self, col: str):
        # Arrange
        import pandas as pd
        df = pd.read_csv(_PER_ROW_CSV, nrows=0)

        # Act / Assert
        assert col in df.columns, (
            f"per_row_iteration_long.csv is missing required column: {col!r}.\n"
            f"Present columns: {sorted(df.columns.tolist())}"
        )


class TestAggregateByRunCsvSchema:
    """
    Verify that aggregate_by_run.csv has all required columns.
    """

    REQUIRED_COLUMNS = [
        "branch",
        "run_label",
        "model_name",
        "n_rows",
        "n_nonempty_cells",
        "total_target_thinking_tokens",
        "total_architect_post_think_tail_tokens",
        "mean_target_thinking_tokens_per_row",
        "mean_architect_post_think_tail_tokens_per_row",
        "pct_target_of_total",
    ]

    @pytest.mark.unit
    def test_AggregateByRunCsv_FileExists(self):
        assert _AGGREGATE_CSV.exists(), (
            f"aggregate_by_run.csv does not exist at:\n  {_AGGREGATE_CSV}\n"
            f"Run the CLI pipeline to generate it."
        )

    @pytest.mark.skipif(
        not _aggregate_csv_exists,
        reason="aggregate_by_run.csv not yet generated — run the CLI pipeline first",
    )
    @pytest.mark.unit
    def test_AggregateByRunCsv_HasAllRequiredColumns(self):
        # Arrange
        import pandas as pd
        df = pd.read_csv(_AGGREGATE_CSV, nrows=0)

        # Act
        actual_columns = set(df.columns.tolist())
        missing = set(self.REQUIRED_COLUMNS) - actual_columns

        # Assert
        assert not missing, (
            f"aggregate_by_run.csv is missing required columns:\n"
            f"  Missing: {sorted(missing)}\n"
            f"  Present: {sorted(actual_columns)}"
        )

    @pytest.mark.skipif(
        not _aggregate_csv_exists,
        reason="aggregate_by_run.csv not yet generated",
    )
    @pytest.mark.unit
    @pytest.mark.parametrize("col", REQUIRED_COLUMNS)
    def test_AggregateByRunCsv_HasColumn(self, col: str):
        # Arrange
        import pandas as pd
        df = pd.read_csv(_AGGREGATE_CSV, nrows=0)

        # Act / Assert
        assert col in df.columns, (
            f"aggregate_by_run.csv is missing required column: {col!r}.\n"
            f"Present columns: {sorted(df.columns.tolist())}"
        )


class TestManifestJsonSchema:
    """
    Verify that manifest.json has all required top-level fields.
    """

    REQUIRED_FIELDS = [
        "script_version",
        "tiktoken_version",
        "encoding_name",
        "encode_method",
        "repo_commit",
        "python_version",
        "timestamp_utc",
        "inception_data_root",
        "runs_discovered",
        "runs_with_pickles",
        "total_rows_emitted",
        "column_definitions",
        "limitations",
    ]

    @pytest.mark.unit
    def test_ManifestJson_FileExists(self):
        assert _MANIFEST_JSON.exists(), (
            f"manifest.json does not exist at:\n  {_MANIFEST_JSON}\n"
            f"Run the CLI pipeline to generate it."
        )

    @pytest.mark.skipif(
        not _manifest_json_exists,
        reason="manifest.json not yet generated — run the CLI pipeline first",
    )
    @pytest.mark.unit
    def test_ManifestJson_IsValidJson(self):
        # Arrange
        raw = _MANIFEST_JSON.read_text(encoding="utf-8")

        # Act / Assert
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"manifest.json is not valid JSON: {exc}\n"
                f"First 200 chars: {raw[:200]!r}"
            )
        assert isinstance(data, dict), "manifest.json must be a JSON object (dict)"

    @pytest.mark.skipif(
        not _manifest_json_exists,
        reason="manifest.json not yet generated",
    )
    @pytest.mark.unit
    def test_ManifestJson_HasAllRequiredFields(self):
        # Arrange
        data = json.loads(_MANIFEST_JSON.read_text(encoding="utf-8"))

        # Act
        missing = [f for f in self.REQUIRED_FIELDS if f not in data]

        # Assert
        assert not missing, (
            f"manifest.json is missing required fields:\n"
            f"  Missing: {missing}\n"
            f"  Present: {sorted(data.keys())}"
        )

    @pytest.mark.skipif(
        not _manifest_json_exists,
        reason="manifest.json not yet generated",
    )
    @pytest.mark.unit
    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    def test_ManifestJson_HasField(self, field: str):
        # Arrange
        data = json.loads(_MANIFEST_JSON.read_text(encoding="utf-8"))

        # Act / Assert
        assert field in data, (
            f"manifest.json is missing required field: {field!r}.\n"
            f"Present fields: {sorted(data.keys())}"
        )

    @pytest.mark.skipif(
        not _manifest_json_exists,
        reason="manifest.json not yet generated",
    )
    @pytest.mark.unit
    def test_ManifestJson_ColumnDefinitionsIsDict(self):
        # Arrange
        data = json.loads(_MANIFEST_JSON.read_text(encoding="utf-8"))

        # Act / Assert
        assert isinstance(data.get("column_definitions"), dict), (
            "manifest.json 'column_definitions' must be a dict mapping column names "
            "to their descriptions"
        )

    @pytest.mark.skipif(
        not _manifest_json_exists,
        reason="manifest.json not yet generated",
    )
    @pytest.mark.unit
    def test_ManifestJson_LimitationsIsList(self):
        # Arrange
        data = json.loads(_MANIFEST_JSON.read_text(encoding="utf-8"))

        # Act / Assert
        assert isinstance(data.get("limitations"), list), (
            "manifest.json 'limitations' must be a list of strings"
        )

    @pytest.mark.skipif(
        not _manifest_json_exists,
        reason="manifest.json not yet generated",
    )
    @pytest.mark.unit
    def test_ManifestJson_LimitationsIsNonEmpty(self):
        # Arrange
        data = json.loads(_MANIFEST_JSON.read_text(encoding="utf-8"))

        # Act / Assert
        limitations = data.get("limitations", [])
        assert len(limitations) > 0, (
            "manifest.json 'limitations' must be a non-empty list documenting "
            "known caveats (e.g. architect CoT is unrecoverable)"
        )


class TestDataLimitationsMd:
    """
    Verify that DATA_LIMITATIONS.md exists and documents the critical architect CoT caveat.
    """

    @pytest.mark.unit
    def test_DataLimitationsMd_FileExists(self):
        assert _DATA_LIMITATIONS_MD.exists(), (
            f"DATA_LIMITATIONS.md does not exist at:\n  {_DATA_LIMITATIONS_MD}\n"
            f"This file must be created by the CLI pipeline (Subtask 1)."
        )

    @pytest.mark.skipif(
        not _DATA_LIMITATIONS_MD.exists(),
        reason="DATA_LIMITATIONS.md not yet generated",
    )
    @pytest.mark.unit
    def test_DataLimitationsMd_MentionsArchitectCot(self):
        # Arrange
        content = _DATA_LIMITATIONS_MD.read_text(encoding="utf-8").lower()

        # Act / Assert
        # Must contain some mention of "architect" and "cot" or "chain-of-thought"
        # or "thinking" combined with "not available" / "dropped" / "unrecoverable"
        has_architect = "architect" in content
        has_cot_language = any(
            phrase in content
            for phrase in [
                "cot",
                "chain-of-thought",
                "chain of thought",
                "thinking",
                "pre-think",
                "pre think",
            ]
        )
        assert has_architect and has_cot_language, (
            "DATA_LIMITATIONS.md must explicitly mention that architect CoT "
            "(chain-of-thought / thinking tokens) is not available because it is "
            "dropped at save time by src/main.py. "
            f"Found 'architect': {has_architect}, found CoT language: {has_cot_language}"
        )

    @pytest.mark.skipif(
        not _DATA_LIMITATIONS_MD.exists(),
        reason="DATA_LIMITATIONS.md not yet generated",
    )
    @pytest.mark.unit
    def test_DataLimitationsMd_MentionsEncodingName(self):
        # Arrange
        content = _DATA_LIMITATIONS_MD.read_text(encoding="utf-8")

        # Act / Assert
        assert "cl100k_base" in content, (
            "DATA_LIMITATIONS.md must mention the encoding used (cl100k_base) "
            "so readers understand the tokenization methodology"
        )


# ===========================================================================
# 2. CLI tests
# ===========================================================================

class TestAnalyzeScriptCli:
    """
    Tests for evaluation/scripts/analyze_architect_target_thinking_tokens.py.

    These tests use subprocess to invoke the script so they exercise the real
    CLI interface without importing the module directly.
    """

    @pytest.mark.unit
    def test_AnalyzeScript_FileExists(self):
        assert _ANALYZE_SCRIPT.exists(), (
            f"CLI script does not exist at:\n  {_ANALYZE_SCRIPT}\n"
            f"This file must be created as part of Subtask 3."
        )

    @pytest.mark.unit
    def test_AnalyzeScript_AcceptsHelpFlag(self):
        # Arrange / Act
        result = _run_script(_ANALYZE_SCRIPT, "--help")

        # Assert
        assert result.returncode == 0, (
            f"Script exited with code {result.returncode} for --help.\n"
            f"stderr: {result.stderr[:500]}"
        )

    @pytest.mark.unit
    def test_AnalyzeScript_HelpOutputMentionsEvalRoot(self):
        # Arrange / Act
        result = _run_script(_ANALYZE_SCRIPT, "--help")

        # Assert
        assert result.returncode == 0
        assert "--eval-root" in result.stdout, (
            "--eval-root argument must be documented in --help output. "
            f"stdout: {result.stdout[:500]}"
        )

    @pytest.mark.unit
    def test_AnalyzeScript_HelpOutputMentionsInceptionData(self):
        # Arrange / Act
        result = _run_script(_ANALYZE_SCRIPT, "--help")

        # Assert
        assert result.returncode == 0
        assert "--inception-data" in result.stdout, (
            "--inception-data argument must be documented in --help output. "
            f"stdout: {result.stdout[:500]}"
        )

    @pytest.mark.unit
    def test_AnalyzeScript_HelpOutputMentionsOutDir(self):
        # Arrange / Act
        result = _run_script(_ANALYZE_SCRIPT, "--help")

        # Assert
        assert result.returncode == 0
        assert "--out-dir" in result.stdout, (
            "--out-dir argument must be documented in --help output. "
            f"stdout: {result.stdout[:500]}"
        )

    @pytest.mark.unit
    def test_AnalyzeScript_HelpOutputMentionsEncoding(self):
        # Arrange / Act
        result = _run_script(_ANALYZE_SCRIPT, "--help")

        # Assert
        assert result.returncode == 0
        assert "--encoding" in result.stdout, (
            "--encoding argument must be documented in --help output. "
            f"stdout: {result.stdout[:500]}"
        )

    @pytest.mark.unit
    def test_AnalyzeScript_HelpOutputMentionsOverwrite(self):
        # Arrange / Act
        result = _run_script(_ANALYZE_SCRIPT, "--help")

        # Assert
        assert result.returncode == 0
        assert "--overwrite" in result.stdout, (
            "--overwrite flag must be documented in --help output. "
            f"stdout: {result.stdout[:500]}"
        )


class TestValidateScriptCli:
    """
    Tests for evaluation/scripts/validate_architect_target_tokens.py.
    """

    @pytest.mark.unit
    def test_ValidateScript_FileExists(self):
        assert _VALIDATE_SCRIPT.exists(), (
            f"Validation script does not exist at:\n  {_VALIDATE_SCRIPT}\n"
            f"This file must be created as part of Subtask 4."
        )

    @pytest.mark.unit
    def test_ValidateScript_AcceptsHelpFlag(self):
        # Arrange / Act
        result = _run_script(_VALIDATE_SCRIPT, "--help")

        # Assert
        assert result.returncode == 0, (
            f"Validation script exited with code {result.returncode} for --help.\n"
            f"stderr: {result.stderr[:500]}"
        )


# ===========================================================================
# 3. Output validation tests (skipped until outputs exist)
# ===========================================================================

class TestPerRowCsvData:
    """
    Verify the correctness of values in per_row_iteration_long.csv.

    All tests are skipped if the file does not exist yet.
    """

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _per_row_csv_exists,
        reason="per_row_iteration_long.csv not yet generated — run the CLI pipeline first",
    )
    def test_PerRowCsv_HasMoreThanZeroRows(self):
        # Arrange
        df = _load_per_row_csv()

        # Act / Assert
        assert len(df) > 0, (
            "per_row_iteration_long.csv must contain at least one data row. "
            f"Got {len(df)} rows."
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _per_row_csv_exists,
        reason="per_row_iteration_long.csv not yet generated",
    )
    def test_PerRowCsv_TargetThinkingTokensAreNonNegative(self):
        # Arrange
        df = _load_per_row_csv()

        # Act
        negative_mask = df["target_thinking_tokens"] < 0
        negative_count = negative_mask.sum()

        # Assert
        assert negative_count == 0, (
            f"target_thinking_tokens must be >= 0 for all rows. "
            f"Found {negative_count} rows with negative values:\n"
            f"{df[negative_mask][['run_label', 'model_name', 'df_index', 'iteration', 'role', 'target_thinking_tokens']].head(5).to_string()}"
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _per_row_csv_exists,
        reason="per_row_iteration_long.csv not yet generated",
    )
    def test_PerRowCsv_ArchitectPostThinkTailTokensAreNonNegative(self):
        # Arrange
        df = _load_per_row_csv()

        # Act
        negative_mask = df["architect_post_think_tail_tokens"] < 0
        negative_count = negative_mask.sum()

        # Assert
        assert negative_count == 0, (
            f"architect_post_think_tail_tokens must be >= 0 for all rows. "
            f"Found {negative_count} rows with negative values."
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _per_row_csv_exists,
        reason="per_row_iteration_long.csv not yet generated",
    )
    def test_PerRowCsv_ArchitectRows_TargetThinkingTokensIsZero(self):
        # Arrange
        df = _load_per_row_csv()
        architect_rows = df[df["role"] == "architect"]

        # Act
        nonzero_mask = architect_rows["target_thinking_tokens"] != 0
        nonzero_count = nonzero_mask.sum()

        # Assert
        assert nonzero_count == 0, (
            f"For architect rows, target_thinking_tokens must always be 0 "
            f"(architect columns contain post-think tail only, not CoT). "
            f"Found {nonzero_count} architect rows with non-zero target_thinking_tokens."
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _per_row_csv_exists,
        reason="per_row_iteration_long.csv not yet generated",
    )
    def test_PerRowCsv_TargetRows_ArchitectPostThinkTailTokensIsZero(self):
        # Arrange
        df = _load_per_row_csv()
        target_rows = df[df["role"] == "target"]

        # Act
        nonzero_mask = target_rows["architect_post_think_tail_tokens"] != 0
        nonzero_count = nonzero_mask.sum()

        # Assert
        assert nonzero_count == 0, (
            f"For target rows, architect_post_think_tail_tokens must always be 0 "
            f"(target columns contain CoT only, not architect post-think content). "
            f"Found {nonzero_count} target rows with non-zero architect_post_think_tail_tokens."
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _per_row_csv_exists,
        reason="per_row_iteration_long.csv not yet generated",
    )
    def test_PerRowCsv_RoleColumn_ContainsOnlyKnownValues(self):
        # Arrange
        df = _load_per_row_csv()
        known_roles = {"architect", "target", "simple_inject_think", "simple_inject_instruct"}

        # Act
        actual_roles = set(df["role"].unique())
        unknown_roles = actual_roles - known_roles

        # Assert
        assert not unknown_roles, (
            f"per_row_iteration_long.csv contains unexpected role values: {unknown_roles}.\n"
            f"Allowed roles: {known_roles}"
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _per_row_csv_exists,
        reason="per_row_iteration_long.csv not yet generated",
    )
    def test_PerRowCsv_BranchColumn_ContainsOnlyKnownValues(self):
        # Arrange
        df = _load_per_row_csv()
        known_branches = {"max_iterations_5", "ablation", "simple_inject"}

        # Act
        actual_branches = set(df["branch"].unique())
        unknown_branches = actual_branches - known_branches

        # Assert
        assert not unknown_branches, (
            f"per_row_iteration_long.csv contains unexpected branch values: {unknown_branches}.\n"
            f"Allowed branches: {known_branches}"
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _per_row_csv_exists,
        reason="per_row_iteration_long.csv not yet generated",
    )
    def test_PerRowCsv_HadThinkCloseDelimiter_IsBooleanLike(self):
        # Arrange
        df = _load_per_row_csv()

        # Act
        # CSV will store booleans as True/False strings or 0/1
        unique_values = set(df["had_think_close_delimiter"].astype(str).unique())
        allowed = {"True", "False", "0", "1", "true", "false"}
        unexpected = unique_values - allowed

        # Assert
        assert not unexpected, (
            f"had_think_close_delimiter column contains unexpected values: {unexpected}.\n"
            f"Expected boolean-like values: {allowed}"
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _per_row_csv_exists,
        reason="per_row_iteration_long.csv not yet generated",
    )
    def test_PerRowCsv_CharsColumn_IsNonNegative(self):
        # Arrange
        df = _load_per_row_csv()

        # Act
        negative_count = (df["chars"] < 0).sum()

        # Assert
        assert negative_count == 0, (
            f"chars column must be >= 0 for all rows. "
            f"Found {negative_count} rows with negative char counts."
        )


class TestAggregateByRunData:
    """
    Verify the correctness of values in aggregate_by_run.csv.
    """

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _aggregate_csv_exists,
        reason="aggregate_by_run.csv not yet generated — run the CLI pipeline first",
    )
    def test_AggregateByRunCsv_HasMoreThanZeroRows(self):
        # Arrange
        df = _load_aggregate_csv()

        # Act / Assert
        assert len(df) > 0, (
            "aggregate_by_run.csv must contain at least one data row."
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _aggregate_csv_exists,
        reason="aggregate_by_run.csv not yet generated",
    )
    def test_AggregateByRunCsv_PctTargetOfTotal_BetweenZeroAndHundred(self):
        # Arrange
        df = _load_aggregate_csv()

        # Act
        out_of_range = df[
            (df["pct_target_of_total"] < 0) | (df["pct_target_of_total"] > 100)
        ]

        # Assert
        assert len(out_of_range) == 0, (
            f"pct_target_of_total must be between 0 and 100 for all runs. "
            f"Found {len(out_of_range)} out-of-range rows:\n"
            f"{out_of_range[['run_label', 'model_name', 'pct_target_of_total']].to_string()}"
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _aggregate_csv_exists,
        reason="aggregate_by_run.csv not yet generated",
    )
    def test_AggregateByRunCsv_TotalTargetThinkingTokens_IsNonNegative(self):
        # Arrange
        df = _load_aggregate_csv()

        # Act
        negative_count = (df["total_target_thinking_tokens"] < 0).sum()

        # Assert
        assert negative_count == 0, (
            f"total_target_thinking_tokens must be >= 0. "
            f"Found {negative_count} rows with negative values."
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _aggregate_csv_exists,
        reason="aggregate_by_run.csv not yet generated",
    )
    def test_AggregateByRunCsv_TotalArchitectPostThinkTailTokens_IsNonNegative(self):
        # Arrange
        df = _load_aggregate_csv()

        # Act
        negative_count = (df["total_architect_post_think_tail_tokens"] < 0).sum()

        # Assert
        assert negative_count == 0, (
            f"total_architect_post_think_tail_tokens must be >= 0. "
            f"Found {negative_count} rows with negative values."
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _aggregate_csv_exists,
        reason="aggregate_by_run.csv not yet generated",
    )
    def test_AggregateByRunCsv_HasAtMostOneRowPerRunLabel(self):
        # Arrange
        df = _load_aggregate_csv()

        # Act
        duplicates = df.groupby(["run_label", "model_name"]).size()
        multi = duplicates[duplicates > 1]

        # Assert
        assert len(multi) == 0, (
            f"aggregate_by_run.csv should have exactly one row per (run_label, model_name). "
            f"Found duplicates:\n{multi.to_string()}"
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _aggregate_csv_exists,
        reason="aggregate_by_run.csv not yet generated",
    )
    def test_AggregateByRunCsv_NRowsIsPositive(self):
        # Arrange
        df = _load_aggregate_csv()

        # Act
        zero_or_negative = (df["n_rows"] <= 0).sum()

        # Assert
        assert zero_or_negative == 0, (
            f"n_rows must be positive for all runs (each pickle has 800 rows). "
            f"Found {zero_or_negative} runs with n_rows <= 0."
        )


class TestManifestJsonData:
    """
    Verify the correctness of values inside manifest.json.
    """

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _manifest_json_exists,
        reason="manifest.json not yet generated — run the CLI pipeline first",
    )
    def test_ManifestJson_EncodingName_IsCl100kBase(self):
        # Arrange
        data = json.loads(_MANIFEST_JSON.read_text(encoding="utf-8"))

        # Act / Assert
        assert data.get("encoding_name") == "cl100k_base", (
            f"manifest.json encoding_name must be 'cl100k_base', "
            f"got {data.get('encoding_name')!r}"
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _manifest_json_exists,
        reason="manifest.json not yet generated",
    )
    def test_ManifestJson_EncodeMethod_IsEncodeOrdinary(self):
        # Arrange
        data = json.loads(_MANIFEST_JSON.read_text(encoding="utf-8"))

        # Act / Assert
        assert data.get("encode_method") == "encode_ordinary", (
            f"manifest.json encode_method must be 'encode_ordinary' to confirm "
            f"no special tokens were injected. Got: {data.get('encode_method')!r}"
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _manifest_json_exists,
        reason="manifest.json not yet generated",
    )
    def test_ManifestJson_RunsDiscovered_IsPositiveInteger(self):
        # Arrange
        data = json.loads(_MANIFEST_JSON.read_text(encoding="utf-8"))

        # Act / Assert
        runs_discovered = data.get("runs_discovered")
        assert isinstance(runs_discovered, int) and runs_discovered > 0, (
            f"manifest.json runs_discovered must be a positive integer. "
            f"Got: {runs_discovered!r}"
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _manifest_json_exists,
        reason="manifest.json not yet generated",
    )
    def test_ManifestJson_TotalRowsEmitted_IsPositiveInteger(self):
        # Arrange
        data = json.loads(_MANIFEST_JSON.read_text(encoding="utf-8"))

        # Act / Assert
        total_rows = data.get("total_rows_emitted")
        assert isinstance(total_rows, int) and total_rows > 0, (
            f"manifest.json total_rows_emitted must be a positive integer. "
            f"Got: {total_rows!r}"
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _manifest_json_exists,
        reason="manifest.json not yet generated",
    )
    def test_ManifestJson_ScriptVersion_IsNonEmptyString(self):
        # Arrange
        data = json.loads(_MANIFEST_JSON.read_text(encoding="utf-8"))

        # Act / Assert
        version = data.get("script_version")
        assert isinstance(version, str) and len(version) > 0, (
            f"manifest.json script_version must be a non-empty string. "
            f"Got: {version!r}"
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _manifest_json_exists,
        reason="manifest.json not yet generated",
    )
    def test_ManifestJson_TimestampUtc_IsIso8601Like(self):
        # Arrange
        data = json.loads(_MANIFEST_JSON.read_text(encoding="utf-8"))

        # Act
        ts = data.get("timestamp_utc", "")

        # Assert — basic ISO 8601 sanity: contains 'T' and '-' and ':'
        assert isinstance(ts, str) and "T" in ts and "-" in ts and ":" in ts, (
            f"manifest.json timestamp_utc must be an ISO 8601 datetime string. "
            f"Got: {ts!r}"
        )


# ===========================================================================
# 4. Cross-reference: per_row totals must align with aggregate_by_run
# ===========================================================================

class TestCrossFileConsistency:
    """
    Verify that aggregate_by_run.csv totals are consistent with per_row_iteration_long.csv.
    """

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _outputs_exist,
        reason="One or more output files not yet generated — run the CLI pipeline first",
    )
    def test_AggregateTargetTokens_MatchesPerRowSum(self):
        # Arrange
        import pandas as pd
        per_row = _load_per_row_csv()
        agg = _load_aggregate_csv()

        # Act
        per_row_totals = (
            per_row.groupby(["run_label", "model_name"])["target_thinking_tokens"]
            .sum()
            .reset_index()
            .rename(columns={"target_thinking_tokens": "computed_total"})
        )
        merged = agg.merge(per_row_totals, on=["run_label", "model_name"], how="inner")
        mismatches = merged[
            merged["total_target_thinking_tokens"] != merged["computed_total"]
        ]

        # Assert
        assert len(mismatches) == 0, (
            f"aggregate_by_run.csv total_target_thinking_tokens does not match "
            f"the sum from per_row_iteration_long.csv for {len(mismatches)} run(s):\n"
            f"{mismatches[['run_label', 'model_name', 'total_target_thinking_tokens', 'computed_total']].to_string()}"
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _outputs_exist,
        reason="One or more output files not yet generated — run the CLI pipeline first",
    )
    def test_AggregateArchitectTokens_MatchesPerRowSum(self):
        # Arrange
        import pandas as pd
        per_row = _load_per_row_csv()
        agg = _load_aggregate_csv()

        # Act
        per_row_totals = (
            per_row.groupby(["run_label", "model_name"])["architect_post_think_tail_tokens"]
            .sum()
            .reset_index()
            .rename(columns={"architect_post_think_tail_tokens": "computed_total"})
        )
        merged = agg.merge(per_row_totals, on=["run_label", "model_name"], how="inner")
        mismatches = merged[
            merged["total_architect_post_think_tail_tokens"] != merged["computed_total"]
        ]

        # Assert
        assert len(mismatches) == 0, (
            f"aggregate_by_run.csv total_architect_post_think_tail_tokens does not match "
            f"the sum from per_row_iteration_long.csv for {len(mismatches)} run(s)."
        )

    @pytest.mark.unit
    @pytest.mark.skipif(
        not _outputs_exist,
        reason="One or more output files not yet generated — run the CLI pipeline first",
    )
    def test_ManifestTotalRowsEmitted_MatchesPerRowCsvRowCount(self):
        # Arrange
        per_row = _load_per_row_csv()
        data = json.loads(_MANIFEST_JSON.read_text(encoding="utf-8"))

        # Act
        csv_row_count = len(per_row)
        manifest_total = data.get("total_rows_emitted", -1)

        # Assert
        assert manifest_total == csv_row_count, (
            f"manifest.json total_rows_emitted ({manifest_total}) must equal the "
            f"actual row count in per_row_iteration_long.csv ({csv_row_count})."
        )
