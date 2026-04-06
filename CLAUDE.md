# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Arthur Common is a shared utility library for the Arthur platform services. It provides common data models, metric aggregation functions, and data processing tools used across Arthur's AI/ML monitoring and governance products.

This library is distributed as a PyPI package: `arthur-common`

## Technologies

- **Language**: Python 3.12+
- **Package Manager**: uv
- **Key Dependencies**:
  - Data Processing: Pandas, DuckDB, fsspec
  - Probabilistic Data Structures: datasketches
  - Data Validation: Pydantic v2
  - API Framework: FastAPI
  - Token Cost Calculations: tokencost
  - Observability: OpenInference semantic conventions

## Installation

```bash
# Via uv
uv add arthur-common

# Via pip
pip install arthur-common
```

## Common Commands

### Setup Development Environment

```bash
uv python pin 3.13
uv sync --all-groups
uv run pre-commit install
```

### Testing

```bash
uv run pytest                    # Run all tests
uv run pytest tests/unit/        # Run unit tests only
uv run pytest --cov             # Run with coverage
```

### Code Quality

```bash
uv run autoflake src/                           # Remove unused imports
uv run isort src/ --profile black               # Sort imports
uv run black src/                               # Format code
uv run mypy --config-file pyproject.toml src/   # Type checking
```

### Pre-commit Checks

```bash
pre-commit run --all-files    # Run all pre-commit hooks manually
```

### Release Process

1. Merge changes to `main` branch
2. Trigger GitHub Actions: **Arthur Common Version Bump**
3. Select bump type (patch/minor/major)
4. Workflow automatically:
   - Bumps version in pyproject.toml
   - Creates git tag
   - Pushes to PyPI

## Architecture

The library is organized into four main modules:

### 1. Models (`src/arthur_common/models/`)

Pydantic-based data models for API contracts and internal schemas.

**Key Files:**
- [enums.py](src/arthur_common/models/enums.py) - Core enumerations (API roles, metric types, model problem types)
- [schema_definitions.py](src/arthur_common/models/schema_definitions.py) - DType enum, ScopeSchemaTag enum
- [metrics.py](src/arthur_common/models/metrics.py) - Base metric models (NumericMetric, SketchMetric)
- [datasets.py](src/arthur_common/models/datasets.py) - Dataset-related enums
- [constants.py](src/arthur_common/models/constants.py) - RBAC roles, password policies, thresholds
- [connectors.py](src/arthur_common/models/connectors.py) - Connector specifications
- [*_schemas.py](src/arthur_common/models/) - Request/response models

**Important Concepts:**
- **RBAC Roles**: ORG_ADMIN, TASK_ADMIN, DEFAULT_RULE_ADMIN, VALIDATION_USER, ORG_AUDITOR
- **Data Types**: INT, FLOAT, BOOL, STRING, UUID, TIMESTAMP, DATE, JSON, IMAGE
- **Schema Tags**: LLM_CONTEXT, LLM_PROMPT, LLM_RESPONSE, PRIMARY_TIMESTAMP, CATEGORICAL, CONTINUOUS, PREDICTION, GROUND_TRUTH
- **Problem Types**: REGRESSION, BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, ARTHUR_SHIELD, CUSTOM, AGENTIC_TRACE

### 2. Aggregations (`src/arthur_common/aggregations/`)

Pluggable aggregation functions for computing metrics from datasets using a **plugin pattern**.

**Architecture:**
- [aggregator.py](src/arthur_common/aggregations/aggregator.py) - Abstract base class `AggregationFunction`
- [functions/\_\_init\_\_.py](src/arthur_common/aggregations/functions/__init__.py) - Auto-discovery and dynamic loading

**Available Aggregation Functions:**
- **Error Metrics**: mean_squared_error, mean_absolute_error
- **Counting**: inference_count, inference_count_by_class, categorical_count, inference_null_count
- **Matrices**: confusion_matrix, multiclass_confusion_matrix
- **Numeric Stats**: numeric_stats, numeric_sum
- **Shield-Specific**: shield_aggregations (toxicity, PII, hallucination)
- **Agentic Systems**: agentic_aggregations

**Plugin Pattern:**
Each aggregation function implements:
- `id()` - UUID identifier
- `display_name()` - User-facing name
- `description()` - Description
- `aggregation_type()` - SketchMetric or NumericMetric
- `reported_aggregations()` - List of BaseReportedAggregation

Functions are automatically discovered via dynamic module loading.

### 3. Tools (`src/arthur_common/tools/`)

Utility functions and data processing tools.

**Key Components:**
- [duckdb_data_loader.py](src/arthur_common/tools/duckdb_data_loader.py) - `DuckDBOperator` class for loading data from various formats
- [schema_inferer.py](src/arthur_common/tools/schema_inferer.py) - `SchemaInferer` for automatic schema detection
- [functions.py](src/arthur_common/tools/functions.py) - Utilities like `uuid_to_base26()`, `hash_nonce()`
- [aggregation_analyzer.py](src/arthur_common/tools/aggregation_analyzer.py) - Aggregation analysis
- [aggregation_loader.py](src/arthur_common/tools/aggregation_loader.py) - Aggregation configuration loading

### 4. Config (`src/arthur_common/config/`)

Configuration management with YAML and environment variable support.

- [config.py](src/arthur_common/config/config.py) - Config class using simple-settings
- [settings.yaml](src/arthur_common/config/settings.yaml) - Default settings (e.g., SEGMENTATION_COL_UNIQUE_VALUE_LIMIT: 100)

## Development Workflow

### Adding New Aggregation Functions

The plugin architecture makes adding new metrics straightforward:

1. Create new file in `src/arthur_common/aggregations/functions/`
2. Implement class inheriting from `AggregationFunction`
3. Define required methods: `id()`, `display_name()`, `description()`, `aggregation_type()`, `reported_aggregations()`
4. The function is automatically discovered - no need to update imports

Example structure:
```python
from arthur_common.aggregations.aggregator import AggregationFunction

class MyNewMetric(AggregationFunction):
    @classmethod
    def id(cls) -> UUID:
        return UUID("your-unique-uuid")

    @classmethod
    def display_name(cls) -> str:
        return "My New Metric"

    # ... implement other required methods
```

### Modifying Schemas

When updating data models:
1. Edit appropriate file in `src/arthur_common/models/`
2. Update Pydantic models with proper type hints
3. Run tests to ensure backward compatibility: `uv run pytest`

### Data Processing

Use the provided utilities:
```python
from arthur_common.tools.duckdb_data_loader import DuckDBOperator
from arthur_common.tools.schema_inferer import SchemaInferer

# Load data into DuckDB
operator = DuckDBOperator(connection, schema)
operator.load_data(data, table_name)

# Infer schema from raw data
inferer = SchemaInferer()
schema = inferer.infer_schema(dataframe)
```

## Testing

**Test Organization:**
- Location: `tests/unit/` with subdirectories mirroring source structure
- Test Data: `tests/test_data/` with sample datasets (balloons, electricity, emails, etc.)
- Fixtures: `conftest.py` provides shared utilities

**Key Test Utilities:**
- `_get_dataset()` - Load test datasets by name
- `create_duckdb_test_data()` - Create test DuckDB connections
- `make_agentic_test_data()` - Generate agentic trace test data

**Coverage Requirements:**
- Minimum 45% code coverage enforced by pre-commit hooks
- Run with: `uv run pytest --cov --cov=src/arthur_common --cov-report term`

## Pre-commit Hooks

Defined in [.pre-commit-config.yaml](.pre-commit-config.yaml):
- **Validation**: trailing-whitespace, end-of-file-fixer, check-yaml, debug-statements
- **Code Quality**: autoflake, isort, black, add-trailing-comma
- **Testing**: pytest with 45% coverage minimum
- **Type Checking**: mypy

Hooks run automatically on `git commit`. To run manually:
```bash
pre-commit run --all-files
```

## Key Configuration Files

| File | Purpose |
|------|---------|
| [pyproject.toml](pyproject.toml) | Project config, dependencies, tool settings (black, mypy, pytest, isort) |
| [.pre-commit-config.yaml](.pre-commit-config.yaml) | Pre-commit hooks for code quality |
| [.bumpversion.cfg](.bumpversion.cfg) | Semantic versioning configuration |
| [uv.lock](uv.lock) | Pinned dependency versions |
| [src/arthur_common/config/settings.yaml](src/arthur_common/config/settings.yaml) | Runtime settings |

## CI/CD

**GitHub Actions Workflows:**
- [arthur-common-workflow.yml](.github/workflows/arthur-common-workflow.yml) - Linting and unit tests on push/PR
- [arthur-common-release.yml](.github/workflows/arthur-common-release.yml) - PyPI build and publish
- [arthur-common-version-bump.yml](.github/workflows/arthur-common-version-bump.yml) - Manual version bump (patch/minor/major)

**Workflow Trigger:**
- Commits to `main` trigger CI/CD
- "Bump version" commits trigger PyPI release

## Important Patterns

### Plugin Pattern
The aggregation system uses dynamic module discovery to automatically load all aggregation functions without manual registration.

### Data Loading
`DuckDBOperator` handles multiple formats with automatic schema application and column aliasing.

### Schema Inference
`SchemaInferer` automatically detects data types and structure from raw data.

### Type Safety
Extensive use of Pydantic models and Python type hints enforced by mypy.

## Quick Reference

**Most Important Modules:**
1. `aggregations/` - Metric computation (plugin architecture)
2. `models/` - Shared data contracts
3. `tools/` - DuckDB data loading and schema inference
4. `config/` - Settings management

**Common Tasks:**
- **Adding Metrics**: Create aggregation function in `aggregations/functions/`
- **Modifying Schemas**: Update Pydantic models in `models/`
- **Data Processing**: Use `DuckDBOperator` and `SchemaInferer` from `tools/`
- **Configuration**: Override settings via `.environ` file or environment variables

**Code Quality Standards:**
- Pre-commit hooks enforce 45% minimum coverage
- mypy strict mode enabled
- Black formatting enforced
- All tests run in CI on every push

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes
# ... edit files ...

# Run tests and formatting
uv run pytest
uv run black src/
uv run isort src/

# Commit (pre-commit hooks run automatically)
git commit -m "Description"

# Push and create PR
git push origin feature/your-feature
```

## Additional Resources

- **PyPI Package**: https://pypi.org/project/arthur-common/
- **Maintainer**: Arthur AI (<engineering@arthur.ai>)
- **License**: MIT
