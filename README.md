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
Releases are automatic. Merging into `main` publishes a new patch version once CI passes:
1. **Arthur Common CI** runs on the merge commit.
2. When it passes, **Arthur Common Version Bump** bumps the patch version and pushes a "Bump version" commit to `main`.
3. That commit starts **Arthur Common Release**, which publishes to PyPI and tags the release.
4. Update the package version in the projects that use it (arthur-engine, arthur-scope).

Renovate merges don't release on their own; a weekly run (Mondays 09:00 UTC) releases them if nothing else has.

For a minor or major release, go to **Actions** -> **Arthur Common Version Bump**, click **Run workflow** and pick the bump type.
