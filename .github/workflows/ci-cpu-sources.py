#!/usr/bin/env python3
"""CI 专用：把 pyproject.toml 的 torch/torchaudio 源临时改成 PyTorch CPU 索引。

背景（SPEC I：CI 用 Python 3.10 + uv sync --group dev，不装 gpu group）：

- `whisperx` 无条件传递依赖 `pyannote-audio -> torch/torchaudio`，所以"不装 gpu group"
  仍会安装 torch；
- 本仓库 `[tool.uv.sources]` 把 torch/torchaudio 固定在 `pytorch-cu128` 索引，
  `UV_TORCH_BACKEND=cpu` **不会**覆盖显式 sources（实测：仍解析出 torch==2.8.0+cu128
  与 14 个 nvidia-* 包）；
- 因此 CI 在 checkout 后用本脚本生成"CPU 源覆盖"版本的 pyproject.toml，再执行
  `uv lock` + `uv sync --group dev`。这只影响 CI 运行环境，**不改变**仓库里部署用的
  gpu 组定义与 cu128 索引。

实测（干净临时目录）：覆盖后 `uv sync --group dev --dry-run` 结果为
`torch==2.8.0+cpu` / `torchaudio==2.8.0+cpu`，待安装列表中 `nvidia-*` 为 0。
"""

from __future__ import annotations

import pathlib
import sys

CU128_URL = "https://download.pytorch.org/whl/cu128"
CPU_URL = "https://download.pytorch.org/whl/cpu"
CU128_INDEX_NAME = "pytorch-cu128"
CPU_INDEX_NAME = "pytorch-cpu"


def main(argv: list[str]) -> int:
    """就地改写 pyproject.toml：cu128 -> cpu 索引，并去掉 gpu 组的 +cu128 本地版本约束。"""

    path = pathlib.Path(argv[1] if len(argv) > 1 else "pyproject.toml")
    text = path.read_text(encoding="utf-8")

    # 1) 索引地址与名称改成 CPU 索引（explicit index 保持显式，不影响其他包）。
    text = text.replace(CU128_URL, CPU_URL)
    text = text.replace(f'name = "{CU128_INDEX_NAME}"', f'name = "{CPU_INDEX_NAME}"')
    text = text.replace(f'index = "{CU128_INDEX_NAME}"', f'index = "{CPU_INDEX_NAME}"')

    # 2) gpu 组里写死的 2.8.0+cu128 在 CPU 索引上没有对应 wheel，
    #    去掉 local version 后由 CPU 索引解析为 2.8.0+cpu。gpu 组本身仍然保留（结构不变）。
    text = text.replace('"torch==2.8.0+cu128"', '"torch==2.8.0"')
    text = text.replace('"torchaudio==2.8.0+cu128"', '"torchaudio==2.8.0"')

    if CPU_URL not in text:
        print("ci-cpu-sources: failed to apply CPU index override", file=sys.stderr)
        return 1
    path.write_text(text, encoding="utf-8")
    print("ci-cpu-sources: pyproject.toml now resolves torch/torchaudio from the CPU index")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
