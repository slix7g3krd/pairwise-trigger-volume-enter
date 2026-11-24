"""针对六类缺陷修复的端到端场景测试。"""

import gameplay.trigger_volume_enter as tv


# --- 修复 1：扫掠跨帧补交，瞬移穿薄墙不再“既不进也不出” ---------------------


def test_swept_detect_catches_teleport_through_thin_wall():
    # 薄墙体积：x∈[0.49,0.51]，角色从 0 瞬移到 10，两端点都在墙外
    volume = {"id": "thin", "min": (0.49, 0, 0), "max": (0.51, 1, 1)}
    actor = {"id": "hero",
             "previous_position": (0.0, 0.5, 0.5),
             "position": (10.0, 0.5, 0.5)}

    result = tv.swept_detect(volume, actor, step=3)

    assert result.crossed is True
    assert result.inside_before is False
    assert result.inside_after is False
    kinds = [e.type for e in result.events]
    assert kinds == ["enter", "exit"]
    assert result.events[0].step == 3
    assert result.events[0].t < result.events[1].t


def test_teleport_ccd_flags_events_and_never_skips():
    volume = tv.volume_bounds(
        {"id": "v", "center": (0, 0, 0), "half_extents": (1, 1, 1)}
    )
    actor = {"id": "a",
             "previous_position": (-50.0, 0.0, 0.0),
             "position": (50.0, 0.0, 0.0)}

    ccd = tv.teleport_ccd(volume, actor, step=9)
    kinds = [e.type for e in ccd.events]
    assert kinds == ["enter", "exit"]
    assert all(e.ccd and e.teleport for e in ccd.events)
    assert all(e.step == 9 for e in ccd.events)


def test_sweep_partial_paths():
    volume = {"id": "v", "min": (0.0,) * 3, "max": (1.0,) * 3}

    # 外 -> 内：只补 enter
    entering = tv.swept_detect(volume, {
        "id": "a", "previous_position": (-1, 0.5, 0.5),
        "position": (0.5, 0.5, 0.5)})
    assert [e.type for e in entering.events] == ["enter"]

    # 内 -> 外：只补 exit
    leaving = tv.swept_detect(volume, {
        "id": "b", "previous_position": (0.5, 0.5, 0.5),
        "position": (3.0, 0.5, 0.5)})
    assert [e.type for e in leaving.events] == ["exit"]

    # 静止在内部：稳态不重复发事件
    steady = tv.swept_detect(volume, {
        "id": "c", "previous_position": (0.5,) * 3,
        "position": (0.5,) * 3})
    assert steady.events == []
    assert steady.overlap is True


# --- 修复 2：同帧 enter/exit 按顺序配对，exit 不再丢失 -----------------------


def test_pair_ordered_commits_both_events():
    enter = tv.enter_event("hero", "zone", step=1, order=0, t=0.2)
    exit_ = tv.exit_event("hero", "zone", step=1, order=1, t=0.8)

    # 故意以乱序传入
    result = tv.pair_ordered([exit_], [enter])

    assert [e.type for e in result.ordered] == ["enter", "exit"]
    assert result.count == 2
    assert result.paired == 1
    assert result.pairs[0].complete is True
    assert result.pairs[0].enter is enter
    assert result.dangling_enters == []


def test_pair_ordered_unpaired_enter_kept():
    enter = tv.enter_event("hero", "zone")
    result = tv.pair_ordered([enter])
    assert len(result.ordered) == 1
    assert result.dangling_enters == [enter]
    assert result.pairs == []


# --- 修复 3：启停时重算重叠集，已站在里面的角色会重新触发 --------------------


def test_enable_recompute_retriggers_standing_actor():
    volume = {"id": "room", "min": (0.0,) * 3, "max": (10.0,) * 3}
    actors = [
        {"id": "inside_guy", "position": (5.0, 5.0, 5.0)},
        {"id": "outside_guy", "position": (20.0, 20.0, 20.0)},
    ]

    # 启用且旧集合为空：内部角色补 enter
    enabled = tv.enable_recompute(volume, actors, set(), enabled=True, step=4)
    assert enabled.overlap == {"inside_guy"}
    assert enabled.entered == ["inside_guy"]
    assert [e.type for e in enabled.events] == ["enter"]

    # 停用：内部角色补 exit，重叠集清空
    disabled = tv.enable_recompute(volume, actors, enabled.overlap,
                                   enabled=False, step=5)
    assert disabled.overlap == set()
    assert disabled.exited == ["inside_guy"]
    assert [e.type for e in disabled.events] == ["exit"]

    # 重新启用：仍站在里面的角色再次触发
    reenabled = tv.enable_recompute(volume, actors, set(), enabled=True, step=6)
    assert reenabled.overlap == {"inside_guy"}
    assert [e.type for e in reenabled.events] == ["enter"]


def test_enable_recompute_tracks_leaver():
    volume = {"id": "room", "min": (0.0,) * 3, "max": (10.0,) * 3}
    actors = [{"id": "mover", "position": (50.0, 50.0, 50.0)}]
    result = tv.enable_recompute(volume, actors, {"mover"}, enabled=True)
    assert result.exited == ["mover"]
    assert "mover" not in result.overlap


# --- 修复 4：卸载清理 + 存活校验 + 审计 + 保守兜底 ---------------------------


def test_unload_clears_dangling_reference_and_emits_exit():
    state = {"room": {"a", "b"}}
    registry = {"a": {"alive": True}, "b": {"alive": False}}

    result = tv.unload_reference(state, "b", registry=registry, step=7)

    assert result.overlaps["room"] == {"a"}
    assert result.removed == [("room", "b")]
    assert result.events and result.events[0].type == "exit"
    assert result.events[0].actor == "b"
    # 原状态不被修改
    assert state["room"] == {"a", "b"}


def test_unload_evicts_unknown_registry_entries():
    state = {"room": {"a", "ghost"}}
    registry = {"a": True}  # ghost 已不在注册表中
    result = tv.unload_reference(state, registry=registry)
    assert result.evicted == [("room", "ghost")]
    assert result.overlaps["room"] == {"a"}


def test_validation_partitions_refs():
    registry = {"a": True, "b": True}
    report = tv.validation(["a", "b", "c"], registry)
    assert set(report.valid) == {"a", "b"}
    assert report.invalid == ["c"]
    assert report.ok is False


def test_audit_flags_stale_and_disabled():
    overlaps = {"room": {"a", "ghost"}, "old_room": {"a"}}
    registry = {"a": True}
    volumes = {
        "room": {"enabled": True},
        "old_room": {"enabled": False},
    }
    report = tv.audit(overlaps, registry, volumes)
    assert report.stale == {"room": ["ghost"]}
    assert report.disabled == {"old_room": ["a"]}
    assert report.clean is False
    assert report.counts.stale == 1


def test_fallback_is_fail_closed():
    result = tv.fallback("ghost", volume="room", reason="streaming_unload")
    assert result.safe is True
    assert result.action == "evict"
    assert result.event.type == "exit"
    assert result.event.actor == "ghost"


# --- 修复 5：物理步序号排序，exit 不再早于更早 step 的 enter 派发 ------------


def test_step_sequence_orders_across_threads():
    physics_queue = [tv.exit_event("hero", "zone", step=2, order=1)]
    main_queue = [tv.enter_event("hero", "zone", step=1, order=0)]

    result = tv.step_sequence(physics_queue, main_queue)

    assert [e.step for e in result.ordered] == [1, 2]
    assert [e.type for e in result.ordered] == ["enter", "exit"]
    assert result.steps == [1, 2]
    assert set(result.by_step.keys()) == {1, 2}


def test_determinism_signature_stable_under_shuffle():
    events = [
        tv.exit_event("a", "v", step=2, order=1),
        tv.enter_event("a", "v", step=1, order=0),
        tv.enter_event("b", "v", step=1, order=0),
    ]
    first = tv.determinism(events)
    second = tv.determinism(list(reversed(events)))

    assert first.signature == second.signature
    assert first.stable is True
    assert first.ordered[0].step == 1
    assert len(first.signature) == 3


# --- 基础原语 ----------------------------------------------------------------


def test_volume_bounds_and_overlap_set():
    volume = {"id": "v", "center": (0, 0, 0), "half_extents": (2, 2, 2)}
    bounds = tv.volume_bounds(volume)
    assert bounds.min == (-2.0, -2.0, -2.0)
    assert bounds.max == (2.0, 2.0, 2.0)

    actors = [
        {"id": "in", "position": (0, 0, 0)},
        {"id": "out", "position": (9, 9, 9)},
        {"id": "edge", "position": (2, 0, 0)},  # 闭区间，表面算在内
    ]
    assert tv.overlap_set(volume, actors) == {"in", "edge"}


def test_metrics_counts():
    events = [
        tv.enter_event("a", "v"),
        tv.exit_event("a", "v"),
        tv.enter_event("b", "v", ccd=True),
    ]
    m = tv.metrics(events)
    assert m.enter == 2
    assert m.exit == 1
    assert m.ccd == 1
    assert m.swept == 2
    assert m.queued == 3


def test_loose_input_forms_never_raise():
    # 标量、裸 id、tuple 形态都应被容错处理
    for name in ("swept_detect", "pair_ordered", "enable_recompute",
                 "unload_reference", "step_sequence", "determinism",
                 "teleport_ccd", "validation", "audit", "fallback",
                 "overlap_set", "metrics", "enter_event", "exit_event",
                 "volume_bounds"):
        assert getattr(tv, name)(1) is not None
