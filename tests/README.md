# StrongREJECT Implementation Tests

This directory contains comprehensive tests for the StrongREJECT safety evaluation implementation in inception-eval.

## Test Structure

### Unit Tests (No API Calls)

Fast, deterministic tests that run entirely in-memory:

1. **`TestComputeStrongRejectScore`** - Tests for `compute_strongreject_score()` in `evaluation/danger_score.py`
   - Refusal handling (always returns 0.0)
   - Score computation with various quality combinations
   - Edge cases (missing values, out-of-range inputs, invalid types)
   - Row-based computation from DataFrame

2. **`TestIterationScoreComputeScore`** - Tests for `IterationScore.compute_score()` in `evaluation/iteration_evaluator.py`
   - Refusal handling
   - Score normalization and clamping
   - Mixed quality value computation

3. **`TestComputeIterationSummary`** - Tests for `compute_iteration_summary()`
   - Aggregation statistics across iterations
   - Missing column handling
   - NaN value handling

4. **`TestDetectQualityDegradation`** - Tests for `detect_quality_degradation()`
   - Degradation detection when scores drop
   - Trend analysis (increasing/decreasing/stable)
   - Edge cases (empty data)

### Integration Tests (Real API Calls)

Tests that make actual API calls to verify end-to-end functionality. These are marked with `@pytest.mark.integration` and require `OPENAI_API_KEY` to be set.

1. **`test_safety_evaluator_with_real_api`** - Tests `SafetyEvaluator` class
   - Makes 1 API call using `gpt-4o-mini`
   - Verifies response contains `specificity`, `convincingness`, and `complied` fields
   - Validates field ranges and StrongREJECT score computation

2. **`test_iteration_evaluator_with_real_api`** - Tests `IterationQualityEvaluator` class
   - Makes 4 API calls (2 samples × 2 iterations) using `gpt-4o-mini`
   - Verifies per-iteration evaluation and score tracking
   - Validates peak iteration detection

## Running Tests

### Prerequisites

```bash
# Install pytest if not already installed
uv pip install pytest

# For integration tests, set your OpenAI API key
export OPENAI_API_KEY="your-key-here"
```

### Run All Unit Tests (Default)

```bash
source .venv/bin/activate
pytest tests/test_strongreject_integration.py -v -m "not integration"
```

Output:
```
29 passed, 2 deselected in 1.07s
```

### Run Only Integration Tests

```bash
source .venv/bin/activate
pytest tests/test_strongreject_integration.py -v -m integration
```

**Note**: Integration tests make real API calls and will incur minimal costs (~$0.001-0.002 with gpt-4o-mini).

### Run All Tests (Unit + Integration)

```bash
source .venv/bin/activate
pytest tests/test_strongreject_integration.py -v
```

### Run Specific Test Class

```bash
# Run only StrongREJECT score tests
pytest tests/test_strongreject_integration.py::TestComputeStrongRejectScore -v

# Run only iteration evaluator tests
pytest tests/test_strongreject_integration.py::TestIterationScoreComputeScore -v
```

### Run Specific Test

```bash
pytest tests/test_strongreject_integration.py::TestComputeStrongRejectScore::test_refusal_returns_zero -v
```

## Test Coverage

The test suite covers:

- **Score Computation Logic**: All edge cases for StrongREJECT formula implementation
- **Input Validation**: Missing values, out-of-range inputs, invalid types
- **DataFrame Operations**: Row-based computation and aggregation
- **Iteration Analysis**: Summary statistics and degradation detection
- **API Integration**: Real API calls with response validation
- **Schema Validation**: Fields `specificity` and `convincingness` are present and in valid ranges

## Integration Test API Usage

Integration tests are designed to minimize API costs:

- `test_safety_evaluator_with_real_api`: **1 API call**
- `test_iteration_evaluator_with_real_api`: **4 API calls**

Total cost estimate: < $0.01 using `gpt-4o-mini`

## Troubleshooting

### ModuleNotFoundError: No module named 'pandas'

Make sure you're running pytest from within the virtual environment:

```bash
source .venv/bin/activate
pytest tests/test_strongreject_integration.py -v
```

### Integration tests skipped

If you see:

```
2 deselected
```

This means integration tests were skipped (default behavior). This is expected when running with `-m "not integration"` or when `OPENAI_API_KEY` is not set.

To run integration tests, ensure `OPENAI_API_KEY` is set and run:

```bash
pytest tests/test_strongreject_integration.py -v -m integration
```

### OPENAI_API_KEY not set

Integration tests will be automatically skipped if the API key is not available:

```python
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping integration test"
)
```

## Files Tested

| File | Functions Tested |
|------|------------------|
| `evaluation/danger_score.py` | `compute_strongreject_score()`, `compute_strongreject_score_row()` |
| `evaluation/iteration_evaluator.py` | `IterationScore.compute_score()`, `compute_iteration_summary()`, `detect_quality_degradation()`, `IterationQualityEvaluator.evaluate_iterations()` |
| `evaluation/safety-judge.py` | `SafetyEvaluator` class (integration test only) |
| `core/openaiapi.py` | `SAFETY_SCHEMA`, `validate_safety_response()` (indirectly through integration tests) |

## Test Philosophy

Following Test-Architect principles:

- **AAA Pattern**: All tests follow Arrange-Act-Assert structure
- **Isolation**: Unit tests have no external dependencies
- **Determinism**: No randomness, time dependencies, or shared state
- **Clarity**: Test names describe scenario and expected behavior
- **Edge Case Coverage**: Comprehensive testing of boundary conditions
- **Minimal Integration**: Real API calls only where necessary, using cost-efficient models
