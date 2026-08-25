from uuid import uuid4

from arthur_common.models.enums import MetricType
from arthur_common.models.request_schemas import NewMetricRequest, NewRuleRequest
from arthur_common.models.task_job_specs import CreateModelTaskJobSpec


def _rule() -> NewRuleRequest:
    return NewRuleRequest(
        name="SSN Regex Rule",
        type="RegexRule",
        apply_to_prompt=True,
        apply_to_response=False,
        config={"regex_patterns": [r"\d{3}-\d{2}-\d{4}"]},
    )


def _metric() -> NewMetricRequest:
    return NewMetricRequest(
        type=MetricType.QUERY_RELEVANCE,
        name="My User Query Relevance",
        metric_metadata="test metric metadata",
    )


class TestCreateModelTaskJobSpecWithoutTaskType:
    """`task_type` and the `initial_metric_required` validator were removed when
    traditional/agentic tasks were consolidated. Every task is agentic now, so
    there is no task type to branch on and no reason to reject initial metrics.
    """

    def test_task_type_field_is_gone(self):
        assert "task_type" not in CreateModelTaskJobSpec.model_fields

    def test_round_trips_without_task_type(self):
        spec = CreateModelTaskJobSpec(
            connector_id=uuid4(),
            task_name="my-task",
            initial_rules=[_rule()],
        )

        dumped = spec.model_dump()
        assert "task_type" not in dumped

        assert CreateModelTaskJobSpec.model_validate(dumped) == spec

    def test_json_round_trips_without_task_type(self):
        spec = CreateModelTaskJobSpec(
            connector_id=uuid4(),
            task_name="my-task",
            initial_rules=[_rule()],
            initial_metrics=[_metric()],
        )

        assert (
            CreateModelTaskJobSpec.model_validate_json(spec.model_dump_json()) == spec
        )

    def test_initial_metrics_defaults_to_empty_list(self):
        spec = CreateModelTaskJobSpec(
            connector_id=uuid4(),
            task_name="my-task",
            initial_rules=[_rule()],
        )

        assert spec.initial_metrics == []

    def test_initial_metrics_accepted_unconditionally(self):
        """The retired `initial_metric_required` validator raised whenever
        metrics were supplied for a TRADITIONAL task. Metrics must now be
        accepted with no task type to gate them.
        """
        spec = CreateModelTaskJobSpec(
            connector_id=uuid4(),
            task_name="my-task",
            initial_rules=[_rule()],
            initial_metrics=[_metric()],
        )

        assert len(spec.initial_metrics) == 1
        assert spec.initial_metrics[0].name == "My User Query Relevance"

    def test_job_type_discriminator_is_unchanged(self):
        spec = CreateModelTaskJobSpec(
            connector_id=uuid4(),
            task_name="my-task",
            initial_rules=[_rule()],
        )

        assert spec.job_type == "create_model_task"
        assert spec.model_dump()["job_type"] == "create_model_task"


class TestCreateModelTaskJobSpecLegacyPayloadTolerance:
    """Rollout guard. During the arthur-common -> arthur-scope rollout an older
    arthur-scope still sends `task_type` on this spec while this version of
    arthur-common no longer declares the field. Pydantic's default `extra`
    behavior is "ignore", so those payloads deserialize cleanly.

    If someone sets `extra="forbid"` on this model, these tests fail and that is
    a real rollout break, not a test to update: old producers would start
    getting validation errors mid-deploy.
    """

    def test_extra_is_not_forbidden(self):
        assert CreateModelTaskJobSpec.model_config.get("extra") != "forbid"

    def test_legacy_traditional_task_type_is_ignored(self):
        payload = {
            "job_type": "create_model_task",
            "connector_id": str(uuid4()),
            "task_name": "my-task",
            "initial_rules": [],
            "initial_metrics": [],
            "task_type": "traditional",
        }

        spec = CreateModelTaskJobSpec.model_validate(payload)

        assert not hasattr(spec, "task_type")
        assert "task_type" not in spec.model_dump()

    def test_legacy_agentic_task_type_with_metrics_is_ignored(self):
        payload = {
            "job_type": "create_model_task",
            "connector_id": str(uuid4()),
            "task_name": "my-task",
            "initial_rules": [],
            "initial_metrics": [_metric().model_dump(mode="json")],
            "task_type": "agentic",
        }

        spec = CreateModelTaskJobSpec.model_validate(payload)

        assert not hasattr(spec, "task_type")
        assert len(spec.initial_metrics) == 1

    def test_legacy_traditional_task_type_with_metrics_is_accepted(self):
        """The old validator rejected exactly this combination. A legacy
        producer sending it must no longer trip a 422.
        """
        payload = {
            "job_type": "create_model_task",
            "connector_id": str(uuid4()),
            "task_name": "my-task",
            "initial_rules": [],
            "initial_metrics": [_metric().model_dump(mode="json")],
            "task_type": "traditional",
        }

        spec = CreateModelTaskJobSpec.model_validate(payload)

        assert len(spec.initial_metrics) == 1
