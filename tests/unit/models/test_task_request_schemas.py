from arthur_common.models.request_schemas import NewTaskRequest, SearchTasksRequest


def _schema_required(model) -> list[str]:
    return model.model_json_schema().get("required", [])


class TestNewTaskRequestIsAgenticDeprecatedButPresent:
    """`is_agentic` stays on the task creation request as a deprecated no-op
    rather than being removed. Removing it would be a breaking API change for
    clients that still send it, so the field is accepted, documented as
    deprecated in the OpenAPI schema, and ignored by consumers -- every task is
    agentic post-consolidation, so there is nothing left for the caller to
    choose.

    REMOVAL CONDITION: delete this test class together with the
    `NewTaskRequest.is_agentic` field, in the cleanup that drops the deprecated
    field once no API client sends it.
    """

    def test_field_still_exists(self):
        assert "is_agentic" in NewTaskRequest.model_fields

    def test_field_is_marked_deprecated_in_the_schema(self):
        schema = NewTaskRequest.model_json_schema()

        assert schema["properties"]["is_agentic"].get("deprecated") is True

    def test_field_is_optional_and_defaults_to_none(self):
        request = NewTaskRequest(name="my-task")

        assert request.is_agentic is None
        assert "is_agentic" not in _schema_required(NewTaskRequest)

    def test_accepts_and_preserves_true(self):
        request = NewTaskRequest.model_validate({"name": "my-task", "is_agentic": True})

        assert request.is_agentic is True
        assert request.model_dump()["is_agentic"] is True

    def test_accepts_and_preserves_false(self):
        request = NewTaskRequest.model_validate(
            {"name": "my-task", "is_agentic": False},
        )

        assert request.is_agentic is False
        assert request.name == "my-task"

    def test_round_trips(self):
        request = NewTaskRequest(name="my-task", is_agentic=True)

        assert NewTaskRequest.model_validate(request.model_dump()) == request


class TestSearchTasksRequestIsAgenticDeprecatedButPresent:
    """The `is_agentic` search filter is likewise kept as a deprecated no-op.
    With every task agentic the filter has no discriminating power, but callers
    that still pass it must not get a validation error, so consumers accept the
    value and do not filter on it.

    REMOVAL CONDITION: delete this test class together with the
    `SearchTasksRequest.is_agentic` field, in the cleanup that drops the
    deprecated field once no API client sends it.
    """

    def test_field_still_exists(self):
        assert "is_agentic" in SearchTasksRequest.model_fields

    def test_field_is_marked_deprecated_in_the_schema(self):
        schema = SearchTasksRequest.model_json_schema()

        assert schema["properties"]["is_agentic"].get("deprecated") is True

    def test_field_is_optional_and_defaults_to_none(self):
        request = SearchTasksRequest()

        assert request.is_agentic is None
        assert "is_agentic" not in _schema_required(SearchTasksRequest)

    def test_accepts_every_legacy_value(self):
        for value in (True, False, None):
            request = SearchTasksRequest.model_validate({"is_agentic": value})

            assert request.is_agentic is value

    def test_round_trips(self):
        request = SearchTasksRequest(task_name="needle", is_agentic=True)

        assert SearchTasksRequest.model_validate(request.model_dump()) == request


class TestLegacyPayloadTolerance:
    """Rollout guard, mirroring the `task_type` guard in test_task_job_specs.py.
    During rollout an older arthur-scope may send fields this version of
    arthur-common no longer declares. Pydantic's default `extra` behavior is
    "ignore", so those payloads deserialize cleanly.

    Setting `extra="forbid"` on either model would fail these tests and would be
    a real rollout break -- old producers would get validation errors mid-deploy
    -- not a test to update.
    """

    def test_new_task_request_extra_is_not_forbidden(self):
        assert NewTaskRequest.model_config.get("extra") != "forbid"

    def test_search_tasks_request_extra_is_not_forbidden(self):
        assert SearchTasksRequest.model_config.get("extra") != "forbid"

    def test_new_task_request_ignores_undeclared_fields(self):
        request = NewTaskRequest.model_validate(
            {"name": "my-task", "task_type": "agentic"},
        )

        assert not hasattr(request, "task_type")
        assert request.name == "my-task"

    def test_search_tasks_request_ignores_undeclared_fields(self):
        request = SearchTasksRequest.model_validate({"task_type": "agentic"})

        assert not hasattr(request, "task_type")
        assert "task_type" not in request.model_dump()
