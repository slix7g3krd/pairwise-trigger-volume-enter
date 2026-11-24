import pytest
import gameplay.trigger_volume_enter as mod

CASES = [
    ("swept_detect", (1,)),
    ("pair_ordered", (1,)),
    ("enable_recompute", (1,)),
    ("unload_reference", (1,)),
    ("step_sequence", (1,)),
    ("determinism", (1,)),
    ("teleport_ccd", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("overlap_set", (1,)),
    ("metrics", (1,)),
    ("enter_event", (1,)),
    ("exit_event", (1,)),
    ("volume_bounds", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
