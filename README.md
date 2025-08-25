# Arthur Common

Arthur Common is a library that contains common operations between Arthur platform services.

## Installation

To install the package, use [Poetry](https://python-poetry.org/):

```bash
poetry add arthur-common
```

or pip

```bash
pip install arthur-common
```

## Requirements

- Python 3.12

## Development

To set up the development environment, ensure you have [Poetry](https://python-poetry.org/) installed, then run:

```bash
poetry env use 3.12
poetry install
```

### Running Tests

This project uses [pytest](https://pytest.org/) for testing. To run the tests, execute:

```bash
poetry run pytest
```

## Generating OpenAPI spec
Must run via `poetry run python generate_openapi_spec -m` to avoid a stale cache when making changes in the modules.

## Release process
1. Merge changes into **main** branch
2. Go to **Actions** -> **Arthur Common Version Bump**
3. Manually trigger workflow there, it will create a PR with version bumping
4. Go to **Pull requests** and check PR for version bump, accept it if everything is okay
5. Version bump commit will be merged to **main** branch and it will start release process
6. Update package version in your project (arthur-engine)

## License

This project is licensed under the MIT License.

## Authors

- Arthur <engineering@arthur.ai>
