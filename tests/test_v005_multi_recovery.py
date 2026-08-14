from lily_contact_planner.multi_contact_v005 import (
    V005MultiRecoveryMixin,
    V005MultiSettings,
    event_nodes,
)
from lily_contact_planner.staged_search_v006 import V006StagedSearchMixin
from lily_contact_planner.planner_search import DfsSearchMixin
from lily_contact_planner.unified_planner import UnifiedContactPlanner


def test_recovered_v005_event_nodes():
    assert event_nodes(2, 2) == ((3, 4), (5, 6))
    assert event_nodes(2, 1) == ((3, 4), (5,))
    assert event_nodes(1, 2) == ((3,), (4, 5))


def test_recovered_v005_baseline_parameters():
    cfg = V005MultiSettings()
    assert cfg.n_nodes == 11
    assert cfg.checker_samples == 101
    assert cfg.touchdown_seeds_per_leg == 5
    assert cfg.seed_samples_per_leg == 256
    assert cfg.initial_touchdown_arc_m == 0.02
    assert cfg.initial_liftoff_clearance_m == 0.02
    assert V005MultiRecoveryMixin.v005_multi_seed == 1303


def test_v006_search_uses_existing_dfs_and_restored_mixins():
    assert issubclass(V006StagedSearchMixin, DfsSearchMixin)
    mro = UnifiedContactPlanner.mro()
    assert mro.index(V005MultiRecoveryMixin) < mro.index(V006StagedSearchMixin)
    assert mro.index(V006StagedSearchMixin) < mro.index(DfsSearchMixin)
