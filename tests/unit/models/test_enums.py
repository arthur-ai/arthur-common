from arthur_common.models import enums
from arthur_common.models.enums import ModelProblemType


class TestTaskTypeRemoved:
    """`TaskType` (TRADITIONAL / AGENTIC) was removed when traditional and
    agentic tasks were consolidated. Re-introducing it would reintroduce the
    branch this consolidation deleted.
    """

    def test_task_type_enum_is_gone(self):
        assert not hasattr(enums, "TaskType")


class TestModelProblemTypeMembersAreLoadBearing:
    """`ModelProblemType` is the DATASET-level problem type, which is a
    different axis from the retired task type. Both ARTHUR_SHIELD and
    AGENTIC_TRACE remain live dataset types: a consolidated application carries
    an AGENTIC_TRACE dataset and, when Shield is in use, an ARTHUR_SHIELD
    dataset alongside it.

    "Shield is deprecated" invites deleting ARTHUR_SHIELD as part of a
    pattern-matched cleanup. It is still the problem type that selects the
    Shield aggregations, so removing it would silently drop every Shield metric.
    """

    def test_arthur_shield_still_exists(self):
        assert ModelProblemType.ARTHUR_SHIELD.value == "arthur_shield"

    def test_agentic_trace_still_exists(self):
        assert ModelProblemType.AGENTIC_TRACE.value == "agentic_trace"

    def test_both_are_members_of_the_enum(self):
        values = ModelProblemType.values()

        assert "arthur_shield" in values
        assert "agentic_trace" in values
