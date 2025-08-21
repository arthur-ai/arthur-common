import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Generator

from generate_openapi_spec import generate_openapi_components_only

logger = logging.getLogger()


def analyze_openapi_diff(input_data: tuple[str, str, str]) -> str:
    if input_data[0].startswith("error"):
        changelog_message = "- **BREAKING CHANGE**"
    else:
        changelog_message = "- **CHANGE**"

    if "api-schema-removed" in input_data[0]:
        changelog_message += " for Component/Schema:"
    changelog_message += input_data[2]
    changelog_message += "\n"
    return changelog_message


def get_output_of_openapi_diff(
    current_openapi_path: str, new_openapi_path: str
) -> Generator[tuple[str, str, str], None, None] | None:
    logger.info(f"Current openapi path: {current_openapi_path}")
    logger.info(f"New openapi path: {new_openapi_path}")
    output = subprocess.run(
        ["oasdiff", "changelog", current_openapi_path, new_openapi_path],
        capture_output=True,
    )
    logger.info(f"Return code of the command: {output.returncode}")
    if output.returncode != 0:
        raise ValueError("Something wrong happened during diff generation.")

    raw_changelog = [
        line
        for line in output.stdout.decode("utf-8").replace("\t", " ").split("\n")
        if line
    ]

    if not raw_changelog:
        logger.info("No changes in OpenAPI schema.")
        return None

    # Here we are returning the generator that iterates over output of the oasdiff application from shell
    # Output of this application contains summary in first line, and change in every 3 lines, that why
    # we are grouping 3 objects from list, that we could iterate over in generator.
    logger.info(f"OpenAPI schema was updated. Summary: {raw_changelog[0]}")
    return (item for item in zip(*(iter(raw_changelog[1:]),) * 3))


def generate_new_openapi(path: str) -> None:
    logger.info("Generating new OpenAPI schema using Pydantic models.")

    try:
        generate_openapi_components_only(minimize=False, output_path=path)
        logger.info(f"OpenAPI schema generated successfully at: {path}")
    except Exception as e:
        logger.error(f"Failed to generate OpenAPI schema: {e}")
        raise


def main() -> None:
    # Get the current directory (assuming this script is in the repo root)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths relative to the current directory
    old_openapi_path = os.path.join(current_dir, "staging.openapi.json")
    new_openapi_path = os.path.join(current_dir, "new.openapi.json")
    changelog_md_path = os.path.join(current_dir, "api_changelog.md")

    # Check if old schema exists, if not create an empty one
    if not os.path.exists(old_openapi_path):
        logger.info("No existing OpenAPI schema found. Creating empty baseline.")
        empty_schema = {
            "openapi": "3.0.0",
            "info": {
                "title": "Arthur Common Models",
                "description": "Data models and schemas for Arthur Common components",
                "version": "1.0.0",
            },
            "paths": {},
            "components": {"schemas": {}},
        }
        with open(old_openapi_path, "w") as f:
            json.dump(empty_schema, f, indent=2)

    # Generate new schema
    generate_new_openapi(new_openapi_path)

    # Generate changelog
    changelog_md: list[str] = ["\n", f"# {datetime.today().strftime('%m/%d/%Y')}\n"]
    starting_length_of_changelog: int = len(changelog_md)
    index_of_new_changelog: int = 3

    try:
        diff_output = get_output_of_openapi_diff(old_openapi_path, new_openapi_path)
        if diff_output is not None:
            for changelog in diff_output:
                changelog_md.append(analyze_openapi_diff(changelog))
    except Exception as e:
        logger.error(f"Error generating changelog: {e}")
        # If changelog generation fails, still update the schema
        pass

    # Update changelog file if changes were found
    if len(changelog_md) != starting_length_of_changelog:
        # Create changelog file if it doesn't exist
        if not os.path.exists(changelog_md_path):
            with open(changelog_md_path, "w") as f:
                f.write("# API Changelog\n\n")

        with open(changelog_md_path, "r+") as f:
            changelog_md_content = f.readlines()
            new_content = (
                changelog_md_content[:index_of_new_changelog]
                + changelog_md
                + changelog_md_content[index_of_new_changelog:]
            )
            f.seek(0)
            f.writelines(new_content)
            f.truncate()

        # Update the old schema with the new one
        subprocess.run(["cp", new_openapi_path, old_openapi_path])
        logger.info("Schema updated and changelog generated.")
        sys.exit(1)  # Exit with code 1 to indicate changes were made
    else:
        logger.info("No changes detected in OpenAPI schema.")
        # Clean up the temporary new schema file
        if os.path.exists(new_openapi_path):
            os.remove(new_openapi_path)


if __name__ == "__main__":
    main()
