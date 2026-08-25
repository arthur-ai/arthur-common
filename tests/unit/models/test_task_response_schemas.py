from arthur_common.models.response_schemas import TaskResponse


def _task_response(**overrides) -> TaskResponse:
    kwargs = {
        "id": "b0a3a5e6-0000-0000-0000-000000000001",
        "name": "my-task",
        "created_at": 1_700_000_000_000,
        "updated_at": 1_700_000_000_000,
        "rules": [],
    }
    kwargs.update(overrides)
    return TaskResponse(**kwargs)


class TestTaskResponseIsAgenticDeprecatedButPresent:
    """`TaskResponse.is_agentic` is deliberately KEPT while `is_agentic` was
    removed from the request schemas. It is a response field that existing API
    clients still read, so it stays and always reports True now that every task
    is agentic. It is marked deprecated in the OpenAPI schema only.

    This is easy to "clean up" by accident -- the request-side removal makes the
    response-side field look like a leftover. It is not.

    REMOVAL CONDITION: delete this test class together with the
    `TaskResponse.is_agentic` field, in the step-5 cleanup that drops the
    deprecated field once no API client reads it.
    """

    def test_field_still_exists(self):
        assert "is_agentic" in TaskResponse.model_fields

    def test_field_is_marked_deprecated_in_the_schema(self):
        schema = TaskResponse.model_json_schema()

        assert schema["properties"]["is_agentic"].get("deprecated") is True

    def test_serializes_as_true(self):
        response = _task_response(is_agentic=True)

        assert response.is_agentic is True
        assert response.model_dump()["is_agentic"] is True

    def test_survives_exclude_none(self):
        """Callers that serialize with exclude_none must still see the field."""
        response = _task_response(is_agentic=True)

        assert "is_agentic" in response.model_dump(exclude_none=True)

    def test_survives_exclude_unset(self):
        """Callers that serialize with exclude_unset must still see the field,
        which requires the producer to set it explicitly rather than relying on
        the None default.
        """
        response = _task_response(is_agentic=True)

        assert "is_agentic" in response.model_dump(exclude_unset=True)

    def test_present_in_emitted_json(self):
        response = _task_response(is_agentic=True)

        assert '"is_agentic":true' in response.model_dump_json()

    def test_round_trips(self):
        response = _task_response(is_agentic=True)

        assert TaskResponse.model_validate_json(response.model_dump_json()) == response
