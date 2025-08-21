import json
from typing import Any
from pydantic import BaseModel
import argparse
import logging

logger = logging.getLogger()


def generate_openapi_components_only(
    minimize: bool = False, output_path: str | None = None
) -> None:
    """Generate OpenAPI spec with only component schemas"""

    # If add new models, add them here
    from arthur_common.models import (
        common_schemas,
        metric_schemas,
        request_schemas,
        response_schemas,
        enums,
    )

    # Collect all Pydantic models
    components: dict[str, dict[str, Any]] = {"schemas": {}}

    # Process each module
    modules = [common_schemas, metric_schemas, request_schemas, response_schemas, enums]

    for module in modules:
        for name, obj in module.__dict__.items():
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseModel)
                and obj != BaseModel
            ):
                try:
                    # Generate JSON schema for the model
                    schema = obj.model_json_schema()
                    components["schemas"][name] = schema
                    logger.info(f"Added schema: {name}")
                except Exception as e:
                    logger.error(f"Error processing {name}: {e}")

    # Create minimal OpenAPI spec with only components
    openapi_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Arthur Common Models",
            "description": "Data models and schemas for Arthur Common components",
            "version": "1.0.0",
        },
        "paths": {},
        "components": components,
    }

    # Determine output filename
    if output_path:
        filename = output_path
    else:
        filename = "staging.openapi.min.json" if minimize else "staging.openapi.json"

    # Write to file
    with open(filename, "w") as f:
        if minimize:
            json.dump(openapi_spec, f, separators=(",", ":"))
        else:
            json.dump(openapi_spec, f, indent=2)

    logger.info(f"\nOpenAPI spec generated with {len(components['schemas'])} schemas")
    logger.info(f"File: {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OpenAPI spec for Arthur Common models"
    )
    parser.add_argument(
        "--minimize",
        "-m",
        action="store_true",
        help="Minimize the output JSON (remove whitespace and newlines)",
    )
    parser.add_argument(
        "--output-path", "-o", type=str, help="Path to output the OpenAPI spec"
    )
    args = parser.parse_args()

    # Run the generator with the minimize flag
    generate_openapi_components_only(
        minimize=args.minimize, output_path=args.output_path
    )


if __name__ == "__main__":
    main()
