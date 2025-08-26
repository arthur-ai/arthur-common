import argparse
import importlib
import json
import logging
import os
from enum import Enum
import pkgutil
from types import ModuleType
from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter
from pydantic.json_schema import models_json_schema

logger = logging.getLogger()


def get_all_model_modules() -> list[ModuleType]:
    """Dynamically discover and import all modules in the models subdirectory"""
    import arthur_common.models as models_package

    modules = []
    package_path = models_package.__path__[0]

    # Get all Python files in the models directory
    for _, name, is_pkg in pkgutil.iter_modules([package_path]):
        if not is_pkg and name != "__init__" and name != "shield":
            try:
                # Import the module dynamically
                module = importlib.import_module(f"arthur_common.models.{name}")
                modules.append(module)
                logger.info(f"Successfully imported module: {name}")
            except ImportError as e:
                logger.warning(f"Failed to import module {name}: {e}")
            except Exception as e:
                logger.warning(f"Error processing module {name}: {e}")

    return modules


def generate_openapi_components_only(
    minimize: bool = False, output_path: str | None = None
) -> None:
    modules = get_all_model_modules()

    enum_schemas: dict[str, dict[str, Any]] = {}
    pydantic_models: list[type[BaseModel]] = []
    type_adapter_config = ConfigDict(use_enum_values=False)

    # Extract classes from all modules
    for module in modules:
        for name, obj in module.__dict__.items():
            if isinstance(obj, type):
                logger.info(f"Processing {name} of type {type(obj)}")
                if hasattr(obj, "model_json_schema") and obj != BaseModel:
                    pydantic_models.append(obj)
                elif issubclass(obj, Enum):  # Handle enums separately
                    try:
                        # Only process enums that are defined in this module, not imported ones
                        if obj.__module__ == module.__name__:
                            adapter = TypeAdapter(obj, config=type_adapter_config)
                            enum_schemas[name] = adapter.json_schema()
                    except Exception as e:
                        logger.error(f"Error processing enum {name}: {e}")

    # Generate schemas for all Pydantic models. Handles complex models.
    _, pydantic_schemas = models_json_schema(
        [(model, "validation") for model in pydantic_models],
        ref_template="#/components/schemas/{model}",
    )

    # Create minimal OpenAPI spec with only components
    openapi_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Arthur Common Models",
            "description": "Data models and schemas for Arthur Common components",
            "version": "1.0.1",
        },
        "paths": {},
        "components": {
            "schemas": {
                **enum_schemas,
                **(pydantic_schemas.get("$defs") or {}),
            }
        },
    }

    # Determine output filename
    if output_path:
        path = output_path
    else:
        filename = "staging.openapi.min.json" if minimize else "staging.openapi.json"
        path_directory = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(path_directory, filename)

    logger.info(f"Writing to file: {path}")
    # Write to file
    with open(path, "w") as f:
        if minimize:
            json.dump(openapi_spec, f, separators=(",", ":"))
        else:
            json.dump(openapi_spec, f, indent=2)

    # Avoiding mypy value type error
    schema_count = len(pydantic_schemas) + len(enum_schemas)
    logger.info(f"\nOpenAPI spec generated with {schema_count} schemas in {path}")


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
