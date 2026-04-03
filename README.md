# Arthur Common

Arthur Common is a library that contains common operations between Arthur platform services.

## Installation

To install the package, use [uv](https://docs.astral.sh/uv/):

```bash
uv add arthur-common
```

or pip

```bash
pip install arthur-common
```

## Requirements

- Python 3.12+

## Development

To set up the development environment, ensure you have [uv](https://docs.astral.sh/uv/) installed, then run:

```bash
uv sync --all-groups
```

### Running Tests

This project uses [pytest](https://pytest.org/) for testing. To run the tests, execute:

```bash
uv run pytest
```

## Release process
1. Merge changes into `main` branch
2. Go to **Actions** -> **Arthur Common Version Bump**
3. Click **Run workflow**. The workflow will create a new commit with the version bump, push it back to the same branch it is triggered on (default `main`), and start the release process
4. Watch in [GitHub Actions](https://github.com/arthur-ai/arthur-common/actions) for Arthur Common Release to run
5. Update package version in your project (arthur-engine)
