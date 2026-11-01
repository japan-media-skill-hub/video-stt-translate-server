"""stt_qwen 的 VAD 定点切片纯函数（无模型依赖，可单测）。

本模块是切片算法的**唯一事实源**：执行器 `stt_qwen.py` 只负责"读音频 → 跑 VAD →
调用本模块 → 把切点变成批次"，所有"切在哪里"的判定全部收在这里，便于用合成用例
先把边界条件跑通，再烧真音频（SPEC §12 执行顺序第 2 步）。

算法与不变量见 `docs/superpowers/specs/2026-11-01-stt-qwen-execution-spec.md` §3：

    第 1 级（正常窗）：候选 = { o ∈ 语音段结束点 | max(prev+min_chunk, target-Δback) ≤ o ≤ target+Δfwd }
                     命中 → cut = argmin|o - target|，policy=vad_offset
    第 2 级（fallback）：候选 = { o | max(prev+min_chunk, target-Δback) ≤ o ≤ target+fallback_fwd }
                     命中 → cut = argmin|o - target|，policy=vad_offset_extended
    第 3 级（硬切）：cut = min(target + fallback_fwd, duration)，policy=hard_cut

关键取舍（都由实测决定，不得凭直觉改回）：

- **prev 锚定**：`target = 上一刀 + nominal`。若改成固定网格 `k × nominal`，切点向前
  吸附后下一片会变成 `(cut_k → (k+1)×nominal)`，实测 5/10 片超过 180s（最长 182.14s），
  超限片会被 qwen_asr 包内 `split_audio_into_chunks` 二次切分，VAD 定点全部作废。
- **取最接近 target 的候选**：取 `max(cands)`（窗内最靠后）会让片长系统漂到窗上沿
  （30±20 档实测均值 ~45s）；取 `argmin|o - target|` 才落在名义片长上。
- **两级搜索窗**：第一级只认 `target ± Δ`，找不到才放宽到 `target + fallback_forward`；
  `fallback_forward` 同时就是**片长上界的来源**（上界 = nominal + fallback_forward）。
- **严格递增 + 最小片长**：候选被 `prev + min_chunk` 夹住，保证切点严格递增且相邻间隔
  不小于 `min_chunk`（历史上出现过 0.48s 碎片片）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


# 片长的绝对硬上限（秒）。来源：Qwen3-ForcedAligner 单次调用上限
# `MAX_FORCE_ALIGN_INPUT_SECONDS = 180`（qwen_asr/inference/utils.py:34-35）；
# 超过该值的切片会被包内 `split_audio_into_chunks` 二次切分（qwen3_asr.py:366），
# 外部 VAD 定点立即失效。任何参数组合都不得越过这条线。
MAX_CHUNK_SECONDS = 180.0

# 尾片最小可识别时长（秒）：短于该值的尾巴并入前一片，避免产生退化短片
# （16k 采样下 0.2s = 3200 采样点，低于模型可稳定处理的下限）。
MIN_TAIL_SECONDS = 0.2

# 切片策略标签：进日志/指标，禁止出现未登记的取值（不静默降级）。
POLICY_VAD_OFFSET = "vad_offset"
POLICY_VAD_OFFSET_EXTENDED = "vad_offset_extended"
POLICY_HARD_CUT = "hard_cut"

# 全部合法策略（配置校验与指标标签共用）。
ALL_POLICIES: tuple[str, ...] = (
    POLICY_VAD_OFFSET,
    POLICY_VAD_OFFSET_EXTENDED,
    POLICY_HARD_CUT,
)


class QwenBoundaryConfigError(ValueError):
    """切片参数非法（对应错误码 `stt_qwen_config_invalid`）。"""

    code = "stt_qwen_config_invalid"

    def __init__(self, detail: str) -> None:
        """记录非法原因，便于直接进日志与 error 字段。"""

        self.detail = detail
        super().__init__(f"{self.code}: {detail}")


@dataclass(frozen=True)
class BoundaryParams:
    """切片参数集（全部来自配置，代码里不得硬编码任何一套档位）。

    两套已实测档位（SPEC §14）：
      本轮落地：nominal=30 / Δback=20 / Δfwd=20 / fallback_fwd=40 → 上界 70s / batch=10
      目标档　：nominal=50 / Δback=20 / Δfwd=20 / fallback_fwd=30 → 上界 80s / batch=13
    """

    nominal_seconds: float = 30.0
    delta_back_seconds: float = 20.0
    delta_forward_seconds: float = 20.0
    fallback_forward_seconds: float = 40.0
    min_chunk_seconds: float = 10.0
    # 扩窗仍无语音边界时是否硬切到上界（true）——false 表示"继续前向找边界，
    # 直到绝对硬上限 180s 才放弃"，见 pick_cuts 的 hard_cut_on_miss 分支。
    hard_cut_on_miss: bool = True

    @property
    def max_chunk_seconds(self) -> float:
        """片长上界（秒）= nominal + fallback_forward。"""

        return float(self.nominal_seconds) + float(self.fallback_forward_seconds)

    def validate(self) -> None:
        """校验参数自洽；不合法直接抛 `stt_qwen_config_invalid`（fail-fast）。

        规则（SPEC §3.2 / §6.3）：
        - 所有时长必须为正的有限数；
        - `min_chunk ≤ nominal`（否则最小片长与名义片长互相矛盾）；
        - `nominal + fallback_forward ≤ 180`（I1：越过即被包内二次切分）。
        """

        values = {
            "nominal_seconds": self.nominal_seconds,
            "delta_back_seconds": self.delta_back_seconds,
            "delta_forward_seconds": self.delta_forward_seconds,
            "fallback_forward_seconds": self.fallback_forward_seconds,
            "min_chunk_seconds": self.min_chunk_seconds,
        }
        for name, raw in values.items():
            try:
                number = float(raw)
            except (TypeError, ValueError) as exc:
                raise QwenBoundaryConfigError(f"{name}_not_a_number") from exc
            # NaN/Inf 也必须挡掉：它们会让下面所有比较恒为 False，静默切出怪片。
            if number != number or number in (float("inf"), float("-inf")):
                raise QwenBoundaryConfigError(f"{name}_not_finite")
            if number <= 0:
                raise QwenBoundaryConfigError(f"{name}_must_be_positive")
        if float(self.min_chunk_seconds) > float(self.nominal_seconds):
            raise QwenBoundaryConfigError("min_chunk_seconds_exceeds_nominal_seconds")
        if self.max_chunk_seconds > MAX_CHUNK_SECONDS:
            raise QwenBoundaryConfigError(
                "chunk_upper_bound_exceeds_180s: "
                f"nominal+fallback_forward={self.max_chunk_seconds:.3f}"
            )


@dataclass
class CutPlan:
    """切片结果：切点 + 逐片元数据 + 策略计数。"""

    cuts: list[float] = field(default_factory=list)
    meta: list[dict[str, Any]] = field(default_factory=list)

    @property
    def policy_counts(self) -> dict[str, int]:
        """按策略统计切点数（三类计数都必须能被外部看到，不许隐藏）。"""

        counts = {policy: 0 for policy in ALL_POLICIES}
        for item in self.meta:
            policy = str(item.get("policy", ""))
            counts[policy] = counts.get(policy, 0) + 1
        return counts


def pick_cuts(
    duration: float,
    nominal_seconds: float,
    delta_back_seconds: float,
    delta_forward_seconds: float,
    fallback_forward_seconds: float,
    min_chunk_seconds: float = 10.0,
    hard_cut_on_miss: bool = True,
    speech_ends: Sequence[float] | None = None,
) -> tuple[list[float], list[dict[str, Any]]]:
    """在音频时长内挑选切点，返回 `(cuts, meta)`。

    参数语义见 `BoundaryParams`；`speech_ends` 是 VAD 语音段的**结束点**列表
    （取 offset 不取 onset：offset 落在静音里，切在词与词之间）。

    `hard_cut_on_miss=True`（默认）：扩窗仍无边界即硬切到 `nominal + fallback_forward`
    并记 `hard_cut`。
    `hard_cut_on_miss=False`：不满足于 `nominal + fallback_forward` 这个上界——只要在
    **距上一刀 180s**（绝对硬上限）之内能找到一个真实语音边界，就继续往前找并采用它，
    片长可以超过 nominal+fallback_forward。若连 180s 内都没有边界，仍在 180s 处落刀
    并记 `hard_cut`：**任何分支都不会产出 >180s 的片**（I1 是硬不变量，不可配置放弃）。

    返回值 `meta` 每项含：`nominal`（本片名义目标）、`cut`、`policy`、`vad_prob`
    （调用方回填，纯函数阶段为 None）。`cuts` 严格递增、末项 ≤ duration，且任何
    相邻间隔 ≥ `min_chunk_seconds`。
    """

    params = BoundaryParams(
        nominal_seconds=nominal_seconds,
        delta_back_seconds=delta_back_seconds,
        delta_forward_seconds=delta_forward_seconds,
        fallback_forward_seconds=fallback_forward_seconds,
        min_chunk_seconds=min_chunk_seconds,
        hard_cut_on_miss=hard_cut_on_miss,
    )
    params.validate()

    total = float(duration)
    nominal = float(nominal_seconds)
    back = float(delta_back_seconds)
    forward = float(delta_forward_seconds)
    fallback_forward = float(fallback_forward_seconds)
    min_chunk = float(min_chunk_seconds)

    # 语音段结束点：去重 + 排序 + 丢掉落在时长之外的脏值（VAD 尾帧可能超一点点）。
    offsets = sorted(
        {
            round(float(value), 6)
            for value in (speech_ends or [])
            if 0.0 < float(value) <= total
        }
    )

    cuts: list[float] = []
    meta: list[dict[str, Any]] = []
    prev = 0.0
    # 搜索级别：
    #   ① 正常窗 ±Δ → vad_offset
    #   ② fallback 扩窗（前向到 fallback_forward）→ vad_offset_extended
    #   ③ 仅在 hard_cut_on_miss=false 时追加：继续前向搜到"距上一刀 180s"，
    #      即 `target + (180 - nominal)`，保证任何候选片的长度都不越过绝对上限。
    levels: list[tuple[float, float, str]] = [
        (back, forward, POLICY_VAD_OFFSET),
        (back, fallback_forward, POLICY_VAD_OFFSET_EXTENDED),
    ]
    if not hard_cut_on_miss:
        levels.append(
            (back, MAX_CHUNK_SECONDS - nominal, POLICY_VAD_OFFSET_EXTENDED)
        )

    # 循环条件用"名义目标还没到结尾"判定：目标一旦落到结尾 1s 以内，剩下的尾巴
    # 直接作为最后一片（尾片不受 nominal 约束，但受下面的硬上限与时长约束）。
    target = nominal
    while target < total - 1.0:
        chosen: float | None = None
        policy = ""
        # 逐级放宽搜索窗；每级都取**最接近名义目标**的边界，且都被 prev+min_chunk 夹住。
        for back_window, forward_window, tag in levels:
            low = max(prev + min_chunk, target - back_window)
            high = min(total, target + forward_window)
            if high < low:
                continue
            candidates = [offset for offset in offsets if low <= offset <= high]
            if candidates:
                chosen = min(candidates, key=lambda offset: abs(offset - target))
                policy = tag
                break
        if chosen is None:
            # 连"距上一刀 180s"的绝对上限内都没有语音边界：
            # - 默认（hard_cut_on_miss=true）：硬切到 nominal + fallback_forward 并计数；
            # - 禁用硬切时也必须在绝对上限处落刀——否则剩余音频会变成一个 >180s 的
            #   巨片，被 qwen_asr 包内二次切分，比"切在语音上"更糟（I1 是硬不变量）。
            if not hard_cut_on_miss:
                chosen = min(prev + MAX_CHUNK_SECONDS, total)
            else:
                chosen = min(target + fallback_forward, total)
            policy = POLICY_HARD_CUT
        chosen = min(float(chosen), total)
        # 切点落在音频末尾等于"没有切点"：再切一刀只会造出零长片，直接收尾。
        if chosen >= total - 1e-9:
            break
        # 防御性保护：任何情况下都不能让切点回退或与上一刀靠得太近（I2）。
        if chosen < prev + min_chunk:
            chosen = min(prev + min_chunk, total)
        cuts.append(round(chosen, 6))
        meta.append(
            {
                "nominal": round(target, 6),
                "cut": round(chosen, 6),
                "policy": policy,
                "vad_prob": None,
            }
        )
        prev = chosen
        # prev 锚定：目标随上一刀推进（**不得**改成 k×nominal 网格，见模块 docstring）。
        target = prev + nominal
    return cuts, meta


def merge_short_tail(
    cuts: Sequence[float], duration: float, *, min_tail_seconds: float = MIN_TAIL_SECONDS
) -> tuple[list[float], float]:
    """把过短的尾片并入前一片，返回 `(cuts, merged_seconds)`。

    尾片来自 `[最后一刀, duration]`：它没有"下一个切点"可吸附，可能只剩几十毫秒。
    长于 `min_tail_seconds` 的尾巴保留（ASR 能正常识别短句）；短于该值的合并进
    前一片——**必须回报合并时长**，不许静默丢弃（否则覆盖率悄悄掉了没人知道）。
    """

    if not cuts:
        return list(cuts), 0.0
    tail = float(duration) - float(cuts[-1])
    if tail >= float(min_tail_seconds) or tail < 0.0:
        return list(cuts), 0.0
    return list(cuts[:-1]), tail
