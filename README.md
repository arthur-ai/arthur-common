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

- Python 3.13

## Development

To set up the development environment, ensure you have [Poetry](https://python-poetry.org/) installed, then run:

```bash
poetry env use 3.13
poetry install
```

### Running Tests

This project uses [pytest](https://pytest.org/) for testing. To run the tests, execute:

```bash
poetry run pytest
```

## Release process
1. Merge changes into `main` branch
2. Go to **Actions** -> **Arthur Common Version Bump**
3. Click **Run workflow**. The workflow will create a new commit with the version bump, push it back to the same branch it is triggered on (default `main`), and start the release process
4. Watch in [GitHub Actions](https://github.com/arthur-ai/arthur-common/actions) for Arthur Common Release to run
5. Update package version in your project (arthur-engine)

## Dev Release Process
If you want to create a dev version of your release to test your changes in another project, you can follow the proceeding steps:
1. Commit your changes to a feature branch
2. Go to **Actions** -> **Arthur Common Version Bump**
3. Click **Run workflow**. You'll see a form where you'll need to specify the **Branch** you want to build the dev release from.
   Enter the name of the branch (eg. `feature/my-new-feature`) and click **Run workflow**.
4. A release will be pushed to PyPI with a tag like `2.0.0.dev1`. The first part will be the current version of arthur-common,
   and the second part will be the dev version tag.
5. Check the **Summary** tab of the workflow run to find out the version of your dev package. It will be the value indicated after `dev_version=`. You can install that version in your packages for testing.

## License

This project is licensed under the MIT License.

## Authors

- Arthur <engineering@arthur.ai>
