from arthur_common.models.request_schemas import NewTaskRequest, SearchTasksRequest


class TestNewTaskRequestWithoutIsAgentic:
    """`is_agentic` was removed from the task creation request when
    traditional/agentic tasks were consolidated -- every task is agentic, so
    there is nothing for the caller to choose.
    """

    def test_is_agentic_field_is_gone(self):
        assert "is_agentic" not in NewTaskRequest.model_fields

    def test_constructs_and_serializes_without_is_agentic(self):
        request = NewTaskRequest(name="my-task")

        dumped = request.model_dump()
        assert "is_agentic" not in dumped
        assert dumped["name"] == "my-task"

    def test_round_trips_without_is_agentic(self):
        request = NewTaskRequest(name="my-task")

        assert NewTaskRequest.model_validate(request.model_dump()) == request


class TestSearchTasksRequestWithoutIsAgentic:
    """The `is_agentic` search filter was removed alongside the field it
    filtered on. With every task agentic, the filter has no discriminating
    power.
    """

    def test_is_agentic_field_is_gone(self):
        assert "is_agentic" not in SearchTasksRequest.model_fields

    def test_constructs_and_serializes_without_is_agentic(self):
        request = SearchTasksRequest()

        assert "is_agentic" not in request.model_dump()

    def test_round_trips_without_is_agentic(self):
        request = SearchTasksRequest(task_name="needle")

        assert SearchTasksRequest.model_validate(request.model_dump()) == request


class TestLegacyIsAgenticPayloadTolerance:
    """Rollout guard, mirroring the `task_type` guard in
    test_task_job_specs.py. During rollout an older arthur-scope still sends
    `is_agentic` on these requests while this version of arthur-common no longer
    declares it. Pydantic's default `extra` behavior is "ignore", so those
    payloads deserialize cleanly.

    Setting `extra="forbid"` on either model would fail these tests and would be
    a real rollout break -- old producers would get validation errors mid-deploy
    -- not a test to update.
    """

    def test_new_task_request_extra_is_not_forbidden(self):
        assert NewTaskRequest.model_config.get("extra") != "forbid"

    def test_search_tasks_request_extra_is_not_forbidden(self):
        assert SearchTasksRequest.model_config.get("extra") != "forbid"

    def test_new_task_request_ignores_legacy_is_agentic_true(self):
        request = NewTaskRequest.model_validate({"name": "my-task", "is_agentic": True})

        assert not hasattr(request, "is_agentic")
        assert "is_agentic" not in request.model_dump()

    def test_new_task_request_ignores_legacy_is_agentic_false(self):
        request = NewTaskRequest.model_validate(
            {"name": "my-task", "is_agentic": False},
        )

        assert not hasattr(request, "is_agentic")
        assert request.name == "my-task"

    def test_search_tasks_request_ignores_legacy_is_agentic(self):
        for value in (True, False, None):
            request = SearchTasksRequest.model_validate({"is_agentic": value})

            assert not hasattr(request, "is_agentic")
            assert "is_agentic" not in request.model_dump()
