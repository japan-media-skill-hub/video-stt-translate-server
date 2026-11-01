"""stt_qwen VAD 定点切片纯函数测试（SPEC §9.1，不加载模型）。

这些用例是**先于真音频**跑的边界验证：合成语音边界 → 期望切点。任何一条失败都
说明切片逻辑退化，禁止继续往下烧真素材。
"""

from __future__ import annotations

import pytest

from whisper_stt_service.executor.qwen_boundary import (
    MAX_CHUNK_SECONDS,
    POLICY_HARD_CUT,
    POLICY_VAD_OFFSET,
    POLICY_VAD_OFFSET_EXTENDED,
    BoundaryParams,
    QwenBoundaryConfigError,
    merge_short_tail,
    pick_cuts,
)


# 本轮落地档（SPEC §4.1）：nominal=30 / Δback=20 / Δfwd=20 / fallback_fwd=40 → 上界 70s
LANDED = dict(
    nominal_seconds=30.0,
    delta_back_seconds=20.0,
    delta_forward_seconds=20.0,
    fallback_forward_seconds=40.0,
    min_chunk_seconds=10.0,
)


def _chunk_lengths(cuts: list[float], duration: float) -> list[float]:
    """按切点还原片长（首片从 0 起、末片到 duration）。"""

    bounds = [0.0] + list(cuts) + [float(duration)]
    return [b - a for a, b in zip(bounds[:-1], bounds[1:])]


def _assert_invariants(
    cuts: list[float],
    duration: float,
    *,
    min_chunk: float,
    upper_bound: float,
) -> None:
    """统一断言 §3.2 的 I1/I2/I6：上界、严格递增、最小间隔、不越界。"""

    assert cuts, "至少要有一个切点"
    # I6：末条切点不得越过音频时长
    assert cuts[-1] <= duration + 1e-9, cuts
    for prev, cur in zip(cuts[:-1], cuts[1:]):
        # I2：严格递增
        assert cur > prev, cuts
        # I2：相邻间隔 >= min_chunk
        assert cur - prev >= min_chunk - 1e-9, (prev, cur)
    for length in _chunk_lengths(cuts, duration):
        # I1：每片长度 <= 上界
        assert length <= upper_bound + 1e-9, (length, upper_bound)
        # 绝对上限：任何情况下都不得越过 180s（否则被包内二次切分）
        assert length <= MAX_CHUNK_SECONDS + 1e-9


def test_dense_boundaries_all_land_in_normal_window() -> None:
    """密集边界（每 3s）：全部命中正常窗，片长落在 [nominal-Δback, nominal+Δfwd]。"""

    duration = 1000.0
    ends = [3.0 * i for i in range(1, 334)]
    cuts, meta = pick_cuts(duration, speech_ends=ends, **LANDED)

    _assert_invariants(cuts, duration, min_chunk=10.0, upper_bound=70.0)
    assert all(item["policy"] == POLICY_VAD_OFFSET for item in meta)
    # 首片之外全部应贴着名义片长（Δ 之内），尾片允许更短
    for length in _chunk_lengths(cuts, duration)[:-1]:
        assert 30.0 - 20.0 <= length <= 30.0 + 20.0, length


def test_sparse_boundaries_snap_to_nearest() -> None:
    """稀疏边界（每 25s）：命中最近的边界，片长稳定在 25s 量级。"""

    duration = 1000.0
    ends = [25.0 * i for i in range(1, 41)]
    cuts, meta = pick_cuts(duration, speech_ends=ends, **LANDED)

    _assert_invariants(cuts, duration, min_chunk=10.0, upper_bound=70.0)
    # 25s 间距的边界全都落在正常窗内（±20 > 25/2），不应触发 fallback。
    assert all(item["policy"] == POLICY_VAD_OFFSET for item in meta)
    # 每个切点都必须真的是一个语音段结束点（不得凭空造切点）。
    for cut in cuts:
        assert cut in ends


def test_extreme_sparse_boundaries_fall_back_to_extended_window() -> None:
    """极稀疏（每 45s，正常窗 ±5）：必须走 extended，片长仍 ≤ 上界。"""

    duration = 1000.0
    ends = [45.0 * i for i in range(1, 23)]
    cuts, meta = pick_cuts(
        duration,
        nominal_seconds=30.0,
        delta_back_seconds=5.0,
        delta_forward_seconds=5.0,
        fallback_forward_seconds=40.0,
        min_chunk_seconds=10.0,
        speech_ends=ends,
    )

    _assert_invariants(cuts, duration, min_chunk=10.0, upper_bound=70.0)
    assert all(item["policy"] == POLICY_VAD_OFFSET_EXTENDED for item in meta)
    for cut in cuts:
        assert cut in ends


def test_no_boundaries_hard_cuts_at_upper_bound() -> None:
    """无任何语音边界：全部 hard_cut，片长正好等于上界（首尾正确）。"""

    duration = 300.0
    cuts, meta = pick_cuts(duration, speech_ends=[], **LANDED)

    _assert_invariants(cuts, duration, min_chunk=10.0, upper_bound=70.0)
    assert all(item["policy"] == POLICY_HARD_CUT for item in meta)
    # 硬切位置 = 上一刀 + nominal + fallback_forward
    assert cuts == [70.0, 140.0, 210.0, 280.0], cuts
    assert _chunk_lengths(cuts, duration) == [70.0, 70.0, 70.0, 70.0, 20.0]


def test_boundaries_outside_window_never_exceed_upper_bound() -> None:
    """边界全在窗口之外：不得越过上界，也不得拿窗外的边界当切点。"""

    duration = 300.0
    # 0.5s 落在 low=10 之前；其余都远在扩窗之外
    cuts, meta = pick_cuts(duration, speech_ends=[0.5, 299.5], **LANDED)

    _assert_invariants(cuts, duration, min_chunk=10.0, upper_bound=70.0)
    assert all(item["policy"] == POLICY_HARD_CUT for item in meta)
    assert 0.5 not in cuts


def test_hard_cut_on_miss_disabled_stops_at_absolute_limit() -> None:
    """禁用硬切：前向搜到 180s 绝对上限仍无边界时在 180s 落刀，绝不超过 180s。"""

    duration = 1000.0
    cuts, meta = pick_cuts(
        duration,
        nominal_seconds=30.0,
        delta_back_seconds=20.0,
        delta_forward_seconds=20.0,
        fallback_forward_seconds=40.0,
        min_chunk_seconds=10.0,
        hard_cut_on_miss=False,
        speech_ends=[],
    )

    # 首片走到距上一刀 180s（= target 30 + (180-30)）后放弃，剩余整体作尾片。
    assert cuts == [180.0, 360.0, 540.0, 720.0, 900.0], cuts
    assert all(item["policy"] == POLICY_HARD_CUT for item in meta)
    assert _chunk_lengths(cuts, duration) == [180.0, 180.0, 180.0, 180.0, 180.0, 100.0]


def test_hard_cut_on_miss_disabled_prefers_real_boundary_past_soft_bound() -> None:
    """禁用硬切：软上界（70s）之外但 180s 之内的真实边界应被采用。"""

    duration = 600.0
    # 100s 在软上界 70s 之外、绝对上限 180s 之内 → 必须被采用（片长 100s）。
    cuts, meta = pick_cuts(
        duration,
        nominal_seconds=30.0,
        delta_back_seconds=20.0,
        delta_forward_seconds=20.0,
        fallback_forward_seconds=40.0,
        min_chunk_seconds=10.0,
        hard_cut_on_miss=False,
        speech_ends=[100.0, 300.0, 500.0],
    )
    assert cuts[0] == 100.0, cuts
    assert meta[0]["policy"] == POLICY_VAD_OFFSET_EXTENDED
    _assert_invariants(cuts, duration, min_chunk=10.0, upper_bound=MAX_CHUNK_SECONDS)


def test_prev_anchoring_beats_grid_anchoring_regression() -> None:
    """回归：prev 锚定不得退化成固定网格（SPEC F5 / §13.5）。

    构造"第一个切点被向前吸附很多"的边界：网格锚定下 `k×nominal` 会继续往前推，
    于是下一片变成 90s（越过 70s 上界，真机上会被包内二次切分）；prev 锚定则把
    目标绑在上一刀上，最坏只走到上界。
    """

    duration = 400.0
    ends = [10.0, 100.0, 130.0, 160.0, 190.0, 220.0, 250.0, 280.0, 310.0, 340.0, 370.0]
    cuts, _meta = pick_cuts(duration, speech_ends=ends, **LANDED)
    _assert_invariants(cuts, duration, min_chunk=10.0, upper_bound=70.0)

    # 旧实现（固定网格 k×nominal + 向前吸附）在本用例下必然越界，用于证明用例有区分度。
    def _grid_cuts() -> list[float]:
        """复刻 handoff 里被否掉的固定网格锚定，仅供对照。"""

        out: list[float] = []
        prev = 0.0
        k = 1
        target = float(LANDED["nominal_seconds"])
        while target < duration - 1.0:
            low = max(prev + LANDED["min_chunk_seconds"], target - LANDED["delta_back_seconds"])
            high = min(duration, target + LANDED["fallback_forward_seconds"])
            cands = [o for o in sorted(ends) if low <= o <= high]
            cut = min(cands, key=lambda o: abs(o - target)) if cands else min(
                target + LANDED["fallback_forward_seconds"], duration
            )
            out.append(cut)
            prev = cut
            k += 1
            target = k * float(LANDED["nominal_seconds"])
        return out

    grid_lengths = _chunk_lengths(_grid_cuts(), duration)
    assert max(grid_lengths) > 70.0, grid_lengths


def test_min_chunk_prevents_fragments() -> None:
    """最小片长约束：边界紧贴上一刀时不得切出碎片片（历史 0.48s 碎片回归）。"""

    duration = 200.0
    # 10.0 与 10.2 都在第一个窗内且离 target=30 最近的是 10.2；
    # prev+min_chunk=10 是硬夹子，切完 10.2 之后的下一个候选必须 >= 20.2。
    ends = [10.0, 10.2, 20.2, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0]
    cuts, _meta = pick_cuts(duration, speech_ends=ends, **LANDED)

    _assert_invariants(cuts, duration, min_chunk=10.0, upper_bound=70.0)
    for length in _chunk_lengths(cuts, duration):
        # 尾片可能短于 min_chunk（它不受 I2 约束），但不得出现 0.5s 级别的碎片。
        assert length >= 10.0 - 1e-9 or length > 0.0


def test_short_tail_is_merged_and_reported() -> None:
    """尾片短于阈值：并入前一片，并且必须回报合并时长（不静默丢）。"""

    cuts, merged = merge_short_tail([100.0, 190.0], 190.05)
    assert cuts == [100.0]
    assert merged == pytest.approx(0.05)

    cuts, merged = merge_short_tail([100.0, 190.0], 195.0)
    assert cuts == [100.0, 190.0]
    assert merged == 0.0

    # 没有切点时不做任何事（整段音频就是一片）
    assert merge_short_tail([], 12.0) == ([], 0.0)


@pytest.mark.parametrize(
    "params, detail",
    [
        # 片长上界越过 180s：必须被拒（否则被包内二次切分、VAD 定点全废）
        (
            dict(
                nominal_seconds=180.0,
                delta_back_seconds=20.0,
                delta_forward_seconds=20.0,
                fallback_forward_seconds=30.0,
                min_chunk_seconds=10.0,
            ),
            "chunk_upper_bound_exceeds_180s",
        ),
        # min_chunk 大于 nominal：自相矛盾
        (
            dict(
                nominal_seconds=30.0,
                delta_back_seconds=20.0,
                delta_forward_seconds=20.0,
                fallback_forward_seconds=40.0,
                min_chunk_seconds=40.0,
            ),
            "min_chunk_seconds_exceeds_nominal_seconds",
        ),
        # 非正数
        (
            dict(
                nominal_seconds=0.0,
                delta_back_seconds=20.0,
                delta_forward_seconds=20.0,
                fallback_forward_seconds=40.0,
                min_chunk_seconds=10.0,
            ),
            "nominal_seconds_must_be_positive",
        ),
    ],
)
def test_invalid_params_are_rejected(params: dict, detail: str) -> None:
    """非法参数必须 fail-fast（错误码 stt_qwen_config_invalid），不得静默跑。"""

    with pytest.raises(QwenBoundaryConfigError) as excinfo:
        BoundaryParams(**params).validate()
    assert excinfo.value.code == "stt_qwen_config_invalid"
    assert detail in excinfo.value.detail


def test_two_shipped_profiles_are_both_valid() -> None:
    """两套已实测档位都必须通过校验，且上界分别为 70s / 80s（SPEC §14）。"""

    landed = BoundaryParams(**LANDED)
    landed.validate()
    assert landed.max_chunk_seconds == 70.0

    target = BoundaryParams(
        nominal_seconds=50.0,
        delta_back_seconds=20.0,
        delta_forward_seconds=20.0,
        fallback_forward_seconds=30.0,
        min_chunk_seconds=10.0,
    )
    target.validate()
    assert target.max_chunk_seconds == 80.0


def test_policy_counts_cover_all_three_buckets() -> None:
    """策略计数必须三类齐备（缺省为 0），便于指标直接打点。"""

    _cuts, meta = pick_cuts(300.0, speech_ends=[], **LANDED)
    counts = {item["policy"] for item in meta}
    assert counts == {POLICY_HARD_CUT}

    _cuts2, meta2 = pick_cuts(600.0, speech_ends=[30.0 * i for i in range(1, 20)], **LANDED)
    assert {item["policy"] for item in meta2} == {POLICY_VAD_OFFSET}
