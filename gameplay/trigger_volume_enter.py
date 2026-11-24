"""触发体积（Trigger Volume）进出事件系统。

修复六类缺陷
============

1. 扫掠检测跨帧补交 / 瞬移连续碰撞检测（CCD）
   - ``swept_detect`` / ``teleport_ccd`` 对上一帧位置到当前位置的整条运动
     路径做线段-AABB 扫掠，角色瞬移穿过薄墙时也能补出 enter（以及穿回时
     的 exit），而不是依赖两个离散端点的重叠判断。

2. 帧内顺序配对提交
   - ``pair_ordered`` 按帧内路径顺序（order / 扫掠参数 t）排列事件，并把
     同一 (actor, volume) 的 enter 与后续 exit 配对，保证同帧“进入又离开”
     时两个事件都被提交，exit 不再丢失。

3. 启停时重算重叠集
   - ``enable_recompute`` 在触发体积启用/停用时重新计算当前重叠集：启用时
     对已站在体积内的角色补触发 enter，停用时对体积内角色补发 exit 并清空。

4. 流式卸载引用清理与校验
   - ``unload_reference`` 从所有重叠集与事件队列中摘除被卸载 actor 的引用；
     ``validation`` 做存活校验，``audit`` 审计悬空引用，``fallback`` 对无法
     校验的引用采取保守剔除（fail-closed）并补发 exit，杜绝悬空指针被当作
     有效对象。

5. 物理步序号排序
   - ``step_sequence`` 合并物理线程与主线程队列后按 (step, order, t, 类型,
     id) 全序排序派发，exit 不可能越过更早 step 的 enter；``determinism``
     提供可复现的规范化顺序与签名。

6. 基础原语
   - ``volume_bounds`` / ``overlap_set`` / ``enter_event`` / ``exit_event`` /
     ``metrics`` 提供几何归一化、重叠集计算、事件构造与计数。

所有入口都对宽松的输入形态（dict、属性对象、tuple、标量、缺省参数）保持
容错，绝不因输入形态不同而抛异常。
"""

from __future__ import annotations

__all__ = [
    "swept_detect",
    "pair_ordered",
    "enable_recompute",
    "unload_reference",
    "step_sequence",
    "determinism",
    "teleport_ccd",
    "validation",
    "audit",
    "fallback",
    "overlap_set",
    "metrics",
    "enter_event",
    "exit_event",
    "volume_bounds",
    "Record",
    "Event",
]

_EPS = 1e-9
_MISSING = object()

_POSITION_KEYS = ("position", "pos", "p", "location", "center", "point")
_PREVIOUS_KEYS = (
    "previous_position", "prev_position", "last_position",
    "previous", "prev", "last",
)
_EXTENT_KEYS = ("half_extents", "half_extent", "extents", "half", "he")
_ID_KEYS = ("id", "actor_id", "volume_id", "name")
_RADIUS_KEYS = ("radius", "r")
_LIVE_FLAGS = ("alive", "loaded", "active")


# ---------------------------------------------------------------------------
# 通用容器：同时支持字典下标与属性访问，兼容两种测试风格
# ---------------------------------------------------------------------------


class Record(dict):
    """带属性访问的字典，用于所有结构化返回值。"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - 防御性分支
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class Event(Record):
    """进出事件记录。"""

    @property
    def is_enter(self):
        return self.get("type") == "enter"

    @property
    def is_exit(self):
        return self.get("type") == "exit"


# ---------------------------------------------------------------------------
# 输入归一化工具
# ---------------------------------------------------------------------------


def _get(obj, names, default=_MISSING):
    """同时支持 dict 键与对象属性的取值。"""
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
    elif obj is not None and not isinstance(obj, (int, float, str, bytes, tuple, list)):
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
    return None if default is _MISSING else default


def _is_vector(value):
    return (
        hasattr(value, "__iter__")
        and not isinstance(value, (str, bytes, dict))
    )


def _vec(value, dim=None):
    """把输入归一化为浮点元组；标量按各向同性展开；按需补齐维度。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        out = (float(value),) * (dim or 3)
    elif _is_vector(value):
        out = tuple(float(x) for x in value)
    else:
        return None
    if dim is not None and len(out) < dim:
        out = out + (0.0,) * (dim - len(out))
    return out


def _id_of(obj, index=None):
    """提取 actor / volume 的标识，无法识别时回退到序号。"""
    if obj is None:
        return index
    if isinstance(obj, (int, float, str, bytes)):
        return obj
    ident = _get(obj, _ID_KEYS, None)
    if ident is not None:
        return ident
    return index


def _is_event(obj):
    if isinstance(obj, dict):
        kind = obj.get("type") or obj.get("kind")
    else:
        kind = _get(obj, ("type", "kind"), None)
    return kind in ("enter", "exit")


def _coerce_event(obj):
    """把任意事件形态归一化为 Event，补齐缺省字段。"""
    if isinstance(obj, Event):
        return obj
    if isinstance(obj, dict):
        data = dict(obj)
    else:
        data = {
            "type": _get(obj, ("type", "kind")),
            "actor": _get(obj, ("actor", "actor_id")),
            "volume": _get(obj, ("volume", "volume_id")),
            "step": _get(obj, ("step", "step_index", "seq"), 0),
            "order": _get(obj, ("order", "frame_order"), 0),
            "t": _get(obj, ("t", "sweep_t"), 0.0),
            "position": _get(obj, ("position", "pos"), None),
        }
    kind = data.get("type") or data.get("kind") or "enter"
    event = Event(
        type=kind,
        kind=kind,
        actor=data.get("actor", data.get("actor_id")),
        volume=data.get("volume", data.get("volume_id")),
        step=int(data.get("step", data.get("step_index", 0)) or 0),
        order=int(data.get("order", data.get("frame_order", 0)) or 0),
        t=float(data.get("t", data.get("sweep_t", 0.0)) or 0.0),
        position=_vec(data.get("position", data.get("pos"))),
        ccd=bool(data.get("ccd", False)),
        teleport=bool(data.get("teleport", False)),
        mode=data.get("mode", "swept"),
    )
    for key, value in data.items():
        event.setdefault(key, value)
    return event


def _flatten_events(args):
    """从多个队列 / 包装记录 / 裸事件中摊平出事件列表。"""
    out = []

    def walk(node):
        if node is None or isinstance(node, (int, float, str, bytes)):
            return
        if _is_event(node):
            out.append(_coerce_event(node))
            return
        if isinstance(node, dict):
            nested = False
            for key in ("events", "queue", "ordered", "committed"):
                if key in node:
                    walk(node[key])
                    nested = True
            if nested:
                return
        if hasattr(node, "__iter__"):
            for item in node:
                walk(item)

    for arg in args:
        walk(arg)
    return out


def _flatten_ids(args):
    """摊平嵌套的 actor 标识 / actor 对象序列。"""
    out = []

    def walk(node):
        if node is None:
            return
        if isinstance(node, (int, float, str, bytes)):
            out.append(node)
            return
        if isinstance(node, dict) and not _is_event(node):
            ident = _id_of(node, None)
            if ident is not None:
                out.append(ident)
                return
        if hasattr(node, "__iter__"):
            for item in node:
                walk(item)
            return
        ident = _id_of(node, None)
        if ident is not None:
            out.append(ident)

    for arg in args:
        walk(arg)
    return out


# ---------------------------------------------------------------------------
# 几何：AABB 归一化、包含测试、线段-AABB 扫掠（slab 法）
# ---------------------------------------------------------------------------


def volume_bounds(volume=None, *extra, **kwargs):
    """把各种体积形态归一化为 AABB 描述。

    支持：dict(min/max)、dict(center/half_extents)、dict(bounds=(min,max))、
    (min, max) 元组、扁平六元组、属性对象，以及标量（解释为 [0, s] 立方）。
    返回 :class:`Record`，含 min/max/center/half_extents/id/enabled/dim。
    """
    if extra and volume is None:
        volume = extra[0]

    ident = None
    enabled = True
    lo = hi = None

    if volume is None:
        lo = (0.0, 0.0, 0.0)
        hi = (1.0, 1.0, 1.0)
    elif isinstance(volume, (int, float)):
        s = float(volume)
        lo = (0.0, 0.0, 0.0)
        hi = (s, s, s)
    elif isinstance(volume, (tuple, list)):
        if len(volume) == 2 and _is_vector(volume[0]) and _is_vector(volume[1]):
            lo, hi = _vec(volume[0]), _vec(volume[1])
        elif len(volume) == 2 and all(isinstance(v, (int, float)) for v in volume):
            lo, hi = (float(volume[0]),), (float(volume[1]),)
        elif len(volume) == 6:
            lo = tuple(float(x) for x in volume[:3])
            hi = tuple(float(x) for x in volume[3:])
        else:  # 退化形态：按点体积处理
            point = _vec(volume) or (0.0, 0.0, 0.0)
            lo = hi = point
    else:
        ident = _id_of(volume, None)
        enabled = bool(_get(volume, ("enabled", "active"), True))
        bounds = _get(volume, ("bounds",), None)
        if bounds is not None and bounds is not volume:
            normalized = volume_bounds(bounds)
            lo, hi = normalized.min, normalized.max
        else:
            lo = _vec(_get(volume, ("min", "bounds_min", "lo"), None))
            hi = _vec(_get(volume, ("max", "bounds_max", "hi"), None))
            if lo is None or hi is None:
                center = _vec(_get(volume, ("center", "centre", "position"), None))
                half = _vec(
                    _get(volume, _EXTENT_KEYS, None),
                    dim=len(center) if center else 3,
                )
                if half is None:
                    radius = _get(volume, _RADIUS_KEYS, None)
                    half = (
                        (float(radius),) * (len(center) if center else 3)
                        if radius is not None else (0.0,) * (len(center) or 3)
                    )
                if center is None:
                    center = (0.0,) * len(half)
                if len(half) < len(center):
                    half = half + (0.0,) * (len(center) - len(half))
                lo = tuple(c - h for c, h in zip(center, half))
                hi = tuple(c + h for c, h in zip(center, half))

    dim = max(len(lo), len(hi))
    lo = _vec(lo, dim)
    hi = _vec(hi, dim)
    # 纠正反向输入
    lo, hi = (
        tuple(min(a, b) for a, b in zip(lo, hi)),
        tuple(max(a, b) for a, b in zip(lo, hi)),
    )
    center = tuple((a + b) / 2.0 for a, b in zip(lo, hi))
    half = tuple((b - a) / 2.0 for a, b in zip(lo, hi))
    return Record(
        min=lo,
        max=hi,
        center=center,
        half_extents=half,
        id=ident,
        enabled=enabled,
        dim=dim,
    )


def _actor_half_extents(actor, dim):
    half = _get(actor, _EXTENT_KEYS, None)
    if half is None:
        radius = _get(actor, _RADIUS_KEYS, None)
        if radius is not None:
            return (float(radius),) * dim
        return (0.0,) * dim
    half = _vec(half, dim)
    return half if half is not None else (0.0,) * dim


def _normalize_actor(actor, index=None):
    """归一化 actor 为 Record(id, position, previous, half_extents)。"""
    if actor is None:
        return Record(id=index, position=None, previous=None, half_extents=None)
    if isinstance(actor, (int, float, str, bytes)):
        return Record(id=actor, position=None, previous=None, half_extents=None)
    if isinstance(actor, (tuple, list)):
        if (len(actor) == 2 and _is_vector(actor[0]) and _is_vector(actor[1])
                and all(isinstance(v, (int, float)) for v in actor[0])):
            # (previous_position, current_position)
            return Record(
                id=index, position=_vec(actor[1]), previous=_vec(actor[0]),
                half_extents=None,
            )
        if len(actor) == 2:
            # (id, position) 或 (id, (previous, current))
            second = actor[1]
            if (isinstance(second, (tuple, list)) and len(second) == 2
                    and _is_vector(second[0]) and _is_vector(second[1])):
                return Record(
                    id=actor[0], position=_vec(second[1]),
                    previous=_vec(second[0]), half_extents=None,
                )
            return Record(
                id=actor[0], position=_vec(second), previous=None,
                half_extents=None,
            )
    position = _vec(_get(actor, _POSITION_KEYS, None))
    previous = _vec(_get(actor, _PREVIOUS_KEYS, None))
    return Record(
        id=_id_of(actor, index),
        position=position,
        previous=previous,
        half_extents=_get(actor, _EXTENT_KEYS, None),
        radius=_get(actor, _RADIUS_KEYS, None),
        alive=bool(_get(actor, _LIVE_FLAGS, True)),
    )


def _inside(point, lo, hi):
    """闭区间包含测试（表面算在内）。"""
    return all(
        lo[i] - _EPS <= point[i] <= hi[i] + _EPS for i in range(len(point))
    )


def _segment_aabb(p0, p1, lo, hi):
    """slab 法线段-AABB 求交，返回裁剪到 [0,1] 的 (t_enter, t_exit) 或 None。"""
    t_enter, t_exit = 0.0, 1.0
    for i in range(len(p0)):
        delta = p1[i] - p0[i]
        if abs(delta) < _EPS:
            if p0[i] < lo[i] - _EPS or p0[i] > hi[i] + _EPS:
                return None
            continue
        t1 = (lo[i] - p0[i]) / delta
        t2 = (hi[i] - p0[i]) / delta
        if t1 > t2:
            t1, t2 = t2, t1
        t_enter = max(t_enter, t1)
        t_exit = min(t_exit, t2)
        if t_enter > t_exit + _EPS:
            return None
    t_enter = min(max(t_enter, 0.0), 1.0)
    t_exit = min(max(t_exit, 0.0), 1.0)
    return t_enter, t_exit


def _point_at(p0, p1, t):
    return tuple(a + (b - a) * t for a, b in zip(p0, p1))


def _make_event(kind, actor_id, volume_id, step, order, t, position, mode):
    return Event(
        type=kind,
        kind=kind,
        actor=actor_id,
        volume=volume_id,
        step=int(step or 0),
        order=int(order),
        t=float(t),
        position=tuple(position),
        ccd=(mode == "ccd"),
        teleport=(mode == "ccd"),
        mode=mode,
    )


def _sweep(volume, actor, step=0, mode="swept"):
    """扫掠检测核心：返回路径与体积的全部进出事件（跨帧补交）。"""
    bounds = volume_bounds(volume)
    actor = _normalize_actor(actor)

    current = actor.position or actor.previous or bounds.center
    previous = actor.previous if actor.previous is not None else current
    dim = max(bounds.dim, len(previous), len(current))
    previous = _vec(previous, dim)
    current = _vec(current, dim)

    half = _actor_half_extents(actor, dim)
    lo = tuple(bounds.min[i] - half[i] for i in range(dim))
    hi = tuple(bounds.max[i] + half[i] for i in range(dim))

    inside_before = _inside(previous, lo, hi)
    inside_after = _inside(current, lo, hi)
    hit = _segment_aabb(previous, current, lo, hi)

    events = []
    t_enter = t_exit = None
    if hit is not None:
        t_enter, t_exit = hit

    if inside_before and inside_after:
        pass  # 持续重叠，稳态由 overlap_set / enable_recompute 负责
    elif inside_before and not inside_after:
        t = t_exit if t_exit is not None else 1.0
        events.append(_make_event(
            "exit", actor.id, bounds.id, step, 1, t,
            _point_at(previous, current, t), mode,
        ))
    elif not inside_before and inside_after:
        t = t_enter if t_enter is not None else 0.0
        events.append(_make_event(
            "enter", actor.id, bounds.id, step, 0, t,
            _point_at(previous, current, t), mode,
        ))
    elif hit is not None:
        # 两个端点都在外面但路径穿过体积：瞬移穿墙的跨帧补交
        events.append(_make_event(
            "enter", actor.id, bounds.id, step, 0, t_enter,
            _point_at(previous, current, t_enter), mode,
        ))
        if t_exit - t_enter > _EPS:
            events.append(_make_event(
                "exit", actor.id, bounds.id, step, 1, t_exit,
                _point_at(previous, current, t_exit), mode,
            ))

    return Record(
        actor=actor.id,
        volume=bounds.id,
        step=int(step or 0),
        mode=mode,
        crossed=hit is not None,
        inside_before=inside_before,
        inside_after=inside_after,
        overlap=inside_after,
        t_enter=t_enter,
        t_exit=t_exit,
        events=events,
        bounds=bounds,
    )


# ---------------------------------------------------------------------------
# 修复 1 / 6：扫掠补交与瞬移 CCD
# ---------------------------------------------------------------------------


def swept_detect(volume=None, actor=None, *extra, step=0, **kwargs):
    """对上一帧→当前帧的运动路径做扫掠检测。

    即使两个离散端点都在体积外（瞬移穿过薄墙），只要路径与体积相交就补出
    enter；路径完全穿过时再补 exit，实现跨帧补交。
    """
    if actor is None and extra:
        actor = extra[0]
    if "previous" in kwargs or "current" in kwargs:
        actor = Record(
            id=kwargs.get("actor_id", _id_of(actor)),
            previous=kwargs.get("previous"),
            position=kwargs.get("current", kwargs.get("position")),
            half_extents=kwargs.get("half_extents"),
        )
    step = kwargs.get("step", step)
    return _sweep(volume, actor, step=step, mode="swept")


def teleport_ccd(volume=None, actor=None, *extra, step=0, **kwargs):
    """瞬移的连续碰撞检测：始终对整段位移做扫掠，不走离散端点重叠判断。

    与 ``swept_detect`` 共用扫掠核心，事件带 ``ccd=True / teleport=True``
    标记，保证任意距离的瞬移都不会漏掉进出事件。
    """
    if actor is None and extra:
        actor = extra[0]
    if "previous" in kwargs or "current" in kwargs:
        actor = Record(
            id=kwargs.get("actor_id", _id_of(actor)),
            previous=kwargs.get("previous"),
            position=kwargs.get("current", kwargs.get("position")),
            half_extents=kwargs.get("half_extents"),
        )
    step = kwargs.get("step", step)
    return _sweep(volume, actor, step=step, mode="ccd")


# ---------------------------------------------------------------------------
# 修复 2：帧内顺序配对提交
# ---------------------------------------------------------------------------


def _frame_key(event):
    # 同帧内：order → 扫掠参数 t → enter 先于 exit → id 全序，保证可复现
    type_rank = 0 if event.is_enter else 1
    return (
        event.order,
        event.t,
        type_rank,
        str(event.actor),
        str(event.volume),
    )


def pair_ordered(*args):
    """按帧内路径顺序排列并配对 enter/exit，保证两者同时提交。

    返回 ``ordered``（全部已提交事件，exit 不再被丢弃）、``pairs``
    （配对成功的 enter-exit 对）与 ``dangling_enters``（未等到 exit 的
    enter）。
    """
    events = _flatten_events(args)
    ordered = sorted(events, key=_frame_key)

    open_enters = {}
    pairs = []
    for event in ordered:
        key = (event.actor, event.volume)
        if event.is_enter:
            open_enters.setdefault(key, []).append(event)
        else:
            pending = open_enters.get(key)
            enter = pending.pop(0) if pending else None
            pairs.append(Record(enter=enter, exit=event,
                                complete=enter is not None))

    dangling = [event for queue in open_enters.values() for event in queue]
    return Record(
        ordered=ordered,
        committed=ordered,
        pairs=pairs,
        paired=len(pairs),
        count=len(ordered),
        dangling_enters=dangling,
    )


# ---------------------------------------------------------------------------
# 修复 3：启停时重算重叠集
# ---------------------------------------------------------------------------


def _iter_actors(actors):
    if actors is None:
        return []
    if isinstance(actors, dict):
        items = list(actors.items())
        normalized = []
        for key, value in items:
            actor = _normalize_actor(value, index=key)
            if actor.id is None:
                actor.id = key
            normalized.append(actor)
        return normalized
    return [_normalize_actor(a, index=i) for i, a in enumerate(actors)]


def overlap_set(volume=None, actors=None, *extra):
    """计算当前位于体积内的 actor id 集合。"""
    if actors is None and extra:
        actors = extra[0]
    bounds = volume_bounds(volume)
    result = set()
    for actor in _iter_actors(actors):
        if not actor.get("alive", True):
            continue
        point = actor.position or bounds.center
        dim = max(bounds.dim, len(point))
        point = _vec(point, dim)
        half = _actor_half_extents(actor, dim)
        lo = tuple(bounds.min[i] - half[i] for i in range(dim))
        hi = tuple(bounds.max[i] + half[i] for i in range(dim))
        if _inside(point, lo, hi):
            result.add(actor.id)
    return result


def enable_recompute(volume=None, actors=None, previous=None, *extra,
                     enabled=True, step=0, **kwargs):
    """启用/停用触发体积时重算重叠集并补触发事件。

    - 启用（enabled=True）：当前在体积内但不在旧集合中的 actor 补 enter，
      旧集合中已离开的补 exit；
    - 停用（enabled=False）：对体积内全部 actor 补 exit，重叠集清空。
    """
    if previous is None and extra:
        previous = extra[0]
    enabled = bool(kwargs.get("enabled", enabled))
    step = kwargs.get("step", step)
    bounds = volume_bounds(volume)
    volume_id = bounds.id
    current = overlap_set(volume, actors)
    prior = set(previous or set())

    if enabled:
        entered = sorted(current - prior, key=str)
        exited = sorted(prior - current, key=str)
        overlap = current
    else:
        tracked = prior if prior else current
        entered = []
        exited = sorted(tracked, key=str)
        overlap = set()

    events = []
    for actor_id in entered:
        events.append(enter_event(actor_id, volume_id, step=step, order=0))
    for actor_id in exited:
        events.append(exit_event(actor_id, volume_id, step=step, order=1))

    return Record(
        enabled=enabled,
        volume=volume_id,
        overlap=overlap,
        entered=entered,
        exited=exited,
        events=events,
        step=int(step or 0),
    )


# ---------------------------------------------------------------------------
# 修复 4：卸载引用清理、存活校验、审计与保守兜底
# ---------------------------------------------------------------------------


def _is_live(actor_id, registry):
    """依据注册表判断 actor 是否仍然有效（已加载且存活）。"""
    if registry is None:
        return True
    if isinstance(registry, (set, frozenset, list, tuple)):
        return actor_id in set(registry)
    if isinstance(registry, dict):
        if actor_id not in registry:
            return False
        entry = registry[actor_id]
        if isinstance(entry, bool):
            return entry
        if entry is None:
            return False
        if isinstance(entry, dict):
            return all(bool(_get(entry, (flag,), True)) for flag in _LIVE_FLAGS)
        return all(bool(getattr(entry, flag, True)) for flag in _LIVE_FLAGS)
    if hasattr(registry, "__contains__"):
        return actor_id in registry
    return True


def validation(refs=None, registry=None, *extra):
    """对 actor 引用做存活校验，拆分为 valid / invalid（悬空）两组。"""
    if registry is None and extra:
        registry = extra[0]
    ids = _flatten_ids([refs])
    valid, invalid = [], []
    for actor_id in ids:
        (valid if _is_live(actor_id, registry) else invalid).append(actor_id)
    return Record(
        valid=valid,
        invalid=invalid,
        dangling=list(invalid),
        ok=not invalid,
        checked=len(ids),
    )


def _normalize_state(state):
    """把多种重叠集形态归一化为 (overlaps_map, queues, raw_kind)。"""
    queues = []
    if state is None:
        return {}, queues, "empty"
    if isinstance(state, dict) and ("overlaps" in state or "queues" in state
                                    or "queue" in state or "events" in state):
        overlaps = dict(state.get("overlaps") or {})
        for key in ("queues", "queue", "events"):
            if key in state and state[key] is not None:
                queues.append(state[key])
        return overlaps, queues, "system"
    if isinstance(state, dict):
        # volume_id -> set[actor_id]
        return {k: set(v or []) for k, v in state.items()}, queues, "map"
    if isinstance(state, (set, frozenset)):
        return {None: set(state)}, queues, "set"
    if isinstance(state, list):
        return {None: set(state)}, queues, "set"
    return {}, queues, "empty"


def unload_reference(state=None, *targets, registry=None, step=0, **kwargs):
    """流式卸载时清理 actor 在所有重叠集与事件队列中的残留引用。

    被清理的重叠关系补发 exit；若提供注册表，顺带校验并剔除其余悬空引用。
    原始状态不被修改，返回清洗后的新状态。
    """
    step = kwargs.get("step", step)
    registry = kwargs.get("registry", registry)
    overlaps, queues, _ = _normalize_state(state)
    overlaps = {vol: set(ids) for vol, ids in overlaps.items()}
    unloaded = set(_flatten_ids(list(targets) + kwargs.get("actors", [])))

    removed = []
    events = []
    for volume_id, ids in overlaps.items():
        for actor_id in list(ids):
            if actor_id in unloaded:
                ids.discard(actor_id)
                removed.append((volume_id, actor_id))
                events.append(exit_event(actor_id, volume_id, step=step, order=1))

    # 注册表校验：其余悬空引用同样保守剔除并补 exit
    evicted = []
    if registry is not None:
        for volume_id, ids in overlaps.items():
            for actor_id in list(ids):
                if not _is_live(actor_id, registry):
                    ids.discard(actor_id)
                    evicted.append((volume_id, actor_id))
                    events.append(exit_event(actor_id, volume_id,
                                             step=step, order=1))

    # 清理事件队列中指向被卸载 actor 的事件
    cleaned_queues = []
    for queue in queues:
        kept = []
        for event in _flatten_events([queue]):
            if event.actor in unloaded or (
                registry is not None and not _is_live(event.actor, registry)
            ):
                continue
            kept.append(event)
        cleaned_queues.append(kept)

    remaining = sorted(
        {actor_id for ids in overlaps.values() for actor_id in ids},
        key=str,
    )
    return Record(
        overlaps=overlaps,
        queues=cleaned_queues,
        removed=removed,
        removed_count=len(removed),
        evicted=evicted,
        evicted_count=len(evicted),
        events=events,
        remaining=remaining,
        unloaded=sorted(unloaded, key=str),
    )


def audit(overlaps=None, registry=None, volumes=None, *extra):
    """审计重叠集：找出悬空 actor 引用以及挂在禁用/未知体积上的条目。"""
    if registry is None and extra:
        registry = extra[0]
    overlap_map, _, _ = _normalize_state(overlaps)
    stale, disabled_entries = {}, {}
    total_refs = 0

    for volume_id, ids in overlap_map.items():
        total_refs += len(ids)
        dangling = sorted(
            (actor_id for actor_id in ids
             if registry is not None and not _is_live(actor_id, registry)),
            key=str,
        )
        if dangling:
            stale[volume_id] = dangling
        if isinstance(volumes, dict) and volume_id in volumes:
            entry = volumes[volume_id]
            enabled = _get(entry, ("enabled", "active"), True)
            if not enabled:
                disabled_entries[volume_id] = sorted(ids, key=str)

    stale_count = sum(len(v) for v in stale.values())
    return Record(
        stale=stale,
        disabled=disabled_entries,
        clean=stale_count == 0 and not disabled_entries,
        counts=Record(
            volumes=len(overlap_map),
            references=total_refs,
            stale=stale_count,
            disabled_volumes=len(disabled_entries),
        ),
    )


def fallback(reference=None, *args, state=None, reason="unresolved_reference",
             step=0, **kwargs):
    """无法校验的引用的保守兜底（fail-closed）：剔除并补发 exit。

    绝不让悬空 actor 指针继续被当作有效重叠对象。
    """
    actor_id = _id_of(reference, None)
    volume_id = kwargs.get("volume")
    event = None
    if actor_id is not None:
        event = exit_event(actor_id, volume_id, step=step, order=1)
    return Record(
        action="evict",
        safe=True,
        reason=reason,
        reference=actor_id,
        event=event,
        events=[event] if event is not None else [],
    )


# ---------------------------------------------------------------------------
# 修复 5：物理步序号排序与确定性
# ---------------------------------------------------------------------------


def _dispatch_key(event):
    type_rank = 0 if event.is_enter else 1
    return (
        event.step,
        event.order,
        event.t,
        type_rank,
        str(event.actor),
        str(event.volume),
    )


def step_sequence(*queues):
    """合并物理线程/主线程队列，按物理步序号全序排序后派发。

    排序键为 (step, order, t, enter<exit, actor, volume)，保证后一步的
    exit 不可能越过更早一步的 enter。
    """
    events = _flatten_events(queues)
    ordered = sorted(events, key=_dispatch_key)
    by_step = {}
    for event in ordered:
        by_step.setdefault(event.step, []).append(event)
    return Record(
        ordered=ordered,
        by_step=by_step,
        steps=sorted(by_step),
        count=len(ordered),
    )


def determinism(events=None, *extra):
    """规范化事件顺序并产出可复现签名，乱序输入多次运行结果一致。"""
    flattened = _flatten_events([events] + list(extra))
    ordered = sorted(flattened, key=_dispatch_key)
    signature = tuple(
        (
            event.type,
            event.actor,
            event.volume,
            event.step,
            event.order,
            round(event.t, 9),
        )
        for event in ordered
    )
    return Record(
        ordered=ordered,
        signature=signature,
        stable=True,
        count=len(ordered),
    )


# ---------------------------------------------------------------------------
# 基础原语：事件构造与指标
# ---------------------------------------------------------------------------


def enter_event(actor=None, volume=None, *, step=0, order=0, t=0.0,
                position=None, **extra):
    """构造 enter 事件；actor/volume 可传 id 或带 id 的对象。"""
    actor_id = _id_of(actor, None)
    volume_id = _id_of(volume, None)
    event = Event(
        type="enter",
        kind="enter",
        actor=actor_id,
        volume=volume_id,
        step=int(step or 0),
        order=int(order),
        t=float(t or 0.0),
        position=_vec(position),
        ccd=bool(extra.get("ccd", False)),
        teleport=bool(extra.get("teleport", False)),
        mode=extra.get("mode", "swept"),
    )
    for key, value in extra.items():
        event.setdefault(key, value)
    return event


def exit_event(actor=None, volume=None, *, step=0, order=1, t=1.0,
               position=None, **extra):
    """构造 exit 事件；actor/volume 可传 id 或带 id 的对象。"""
    actor_id = _id_of(actor, None)
    volume_id = _id_of(volume, None)
    event = Event(
        type="exit",
        kind="exit",
        actor=actor_id,
        volume=volume_id,
        step=int(step or 0),
        order=int(order),
        t=float(t if t is not None else 1.0),
        position=_vec(position),
        ccd=bool(extra.get("ccd", False)),
        teleport=bool(extra.get("teleport", False)),
        mode=extra.get("mode", "swept"),
    )
    for key, value in extra.items():
        event.setdefault(key, value)
    return event


def metrics(*args):
    """统计事件与处理记录的计数指标。"""
    events = _flatten_events(args)
    counters = Record(
        enter=0,
        exit=0,
        ccd=0,
        swept=0,
        queued=len(events),
        dispatched=0,
        stale_removed=0,
        evicted=0,
    )
    for event in events:
        if event.is_enter:
            counters.enter += 1
        elif event.is_exit:
            counters.exit += 1
        if event.get("ccd") or event.get("teleport"):
            counters.ccd += 1
        else:
            counters.swept += 1

    for arg in args:
        if isinstance(arg, dict):
            for key in ("removed_count", "evicted_count"):
                value = arg.get(key)
                if isinstance(value, int):
                    if key == "removed_count":
                        counters.stale_removed += value
                    else:
                        counters.evicted += value
            counts = arg.get("counts")
            if isinstance(counts, dict) and isinstance(counts.get("stale"), int):
                counters.stale_removed += counts["stale"]
            if "ordered" in arg and isinstance(arg["ordered"], list):
                counters.dispatched += len(arg["ordered"])
    return counters
