# tsagentkit v1.1.1 综合优化方案

> 本文档基于 Plan A / B / C 的逐条对表与代码审计，形成一份**可执行的最终方案**。
> 每条优化项均标注"来源计划 → 代码现状 → 采纳决策 → 具体行动"。

---

## 〇、方法论

1. **逐条对表**：将三份计划的所有条目提炼为统一条目编号（A1-A7, B1-B7, C1-C12），与真实代码逐一核实。
2. **归类聚合**：将交叉重叠的条目合并为**7 大工作流**（Workstream, WS）。
3. **优先级排序**：按 `影响面 × 实施成本⁻¹` 排序——先修 bug，再补测试，然后做架构升级。
4. **不做什么**：明确排除超出 v1.1.1 scope 的远期项（如 MCP Server、分布式）。

---

## 一、三份计划逐条对表

### Plan A — "API 易用性与渐进式体验"

| 编号 | Plan A 条目 | 代码现状核实 | 对表结论 |
|------|------------|-------------|---------|
| A1 | TaskSpec 概念密度过高（20+ 字段） | `task_spec.py` (430 行)：TaskSpec 确实有 `h, freq, quantiles, levels, tsfm_policy, backtest, covariates, router_thresholds, hierarchy...` 等 15+ 用户可见字段，多数有合理默认值 | ✅ 问题存在但已有默认值；需要的是**文档简化 + 预设模板**，而非删减字段 |
| A2 | Pattern 1 代码示例有 bug（`validate_contract(df)` 缺 spec） | `skill/README.md:72` → `validate_contract(df)` ← `contracts/schema.py:19` 实际签名 `validate_contract(df, spec=None)` → spec 是可选参数，**代码可运行** | ⚠️ 部分误判：代码可运行但**省略 spec 时行为不明确**，文档应明示 |
| A3 | Error 缺乏修复提示 | `errors.py` (301 行)：28 个错误类只有 `error_code + message + context`，无 `fix_hint` 字段 | ✅ 问题确认：需要增加结构化修复提示 |
| A4 | Covariate 处理复杂度过高 | `covariates/__init__.py` (341 行)：`align_covariates()` → `AlignedDataset` → `build_dataset().with_covariates(aligned, panel_with_covariates=df)` — 确实需要传两次 df | ✅ 问题确认：`with_covariates` 接口应简化 |
| A5 | 缺少 Hello World / quickstart | `main.py` 只是 `print("Hello from tsagentkit!")`；`CLAUDE.md:126-137` 有一个快速示例但不在独立模块中 | ✅ 问题确认 |
| A6 | 建议增加 `repair()` 函数 | `qa/__init__.py` 已有 `apply_repairs` 参数；`run_forecast` 的非 strict 模式自动修复 | ⚠️ 部分满足：已有自动修复，但缺少**独立的 repair() 入口** |
| A7 | 建议增加 `suggest_spec()` 决策助手 | 不存在 | ✅ 有价值但属 P2 |

### Plan B — "安装/策略/插件化"

| 编号 | Plan B 条目 | 代码现状核实 | 对表结论 |
|------|------------|-------------|---------|
| B1 | 默认安装太重（torch, chronos, gluonts） | `pyproject.toml:7-23`：核心依赖直接包含 `torch`, `chronos-forecasting`, `tsagentkit-timesfm`, `tsagentkit-uni2ts`, `gluonts` | ✅ **严重问题**：`pip install tsagentkit` 会拉 PyTorch 全量 → 首次安装 5GB+ |
| B2 | TSFMPolicy.mode 默认 `required`，新手不友好 | `task_spec.py:74`：`TSFMPolicy.mode` 默认 `"required"`；无 TSFM 适配器时 `make_plan()` 直接 raise | ✅ 问题确认：但这是 v1.1 的**有意设计决策**，可通过预设模板缓解 |
| B3 | 路由/模型分发硬编码 | `router.py:92-127`：intermittent→`["Croston","Naive"]`, short→`["HistoricAverage","Naive"]`, default→`["SeasonalNaive","HistoricAverage","Naive"]`；`plan_name` 固定 `"default"` | ✅ 问题确认 |
| B4 | wrapper 隐式降级 | `orchestration.py:212-240`：非 strict 模式下 covariate 错误被 catch → 静默丢弃协变量 → 仅在 qa_report.issues 中记录 | ✅ 问题确认：降级可接受但**应显式返回降级事件** |
| B5 | 缺少 CLI | `main.py` 只有 hello world | ✅ 问题确认 |
| B6 | 文档重复维护 | `skill/` 和 `src/tsagentkit/skill/` 手动镜像，靠 `test_phase6_doc_consistency.py` 的 byte-identical 断言保证一致 | ✅ 问题确认：应改为 symlink 或构建时复制 |
| B7 | mypy 形同虚设 | `pyproject.toml:78`：`ignore_errors = true`，`follow_imports = "skip"`，`check_untyped_defs = false` | ✅ **严重问题**：类型检查完全无效 |

### Plan C — "测试/Schema/可观测性"

| 编号 | Plan C 条目 | 代码现状核实 | 对表结论 |
|------|------------|-------------|---------|
| C1 | eval/ 无测试 | `src/tsagentkit/eval/__init__.py` (286 行)：`evaluate_forecasts`, `MetricFrame`, `ScoreSummary` — **确认无测试文件** | ✅ 问题确认 |
| C2 | calibration/ 无测试 | `src/tsagentkit/calibration/__init__.py` (137 行) — 仅在 `test_packaging.py` 间接触及 | ✅ 问题确认 |
| C3 | time/ 无测试 | `src/tsagentkit/time/__init__.py` (135 行)：`infer_freq`, `make_regular_grid`, `make_future_index` — **确认无测试文件** | ✅ 问题确认 |
| C4 | monitoring/alerts.py 无测试 | `alerts.py` (303 行)：`AlertCondition`, `Alert`, `AlertManager` — **确认无测试文件** | ✅ 问题确认 |
| C5 | utils/temporal.py, utils/signature.py 无测试 | **确认无测试文件** | ✅ 问题确认 |
| C6 | `test_full_pipeline_with_monitoring` 注释掉核心调用 | `test_v02_integration.py:345-366`：`run_forecast(...)` 被注释，只断言 config 创建 | ✅ **严重问题**：假测试 |
| C7 | 硬编码 reconciliation `"bottom_up"` | `orchestration.py:841`：`method_str = "bottom_up"` 写死 | ✅ 问题确认 |
| C8 | bucketing `get_model_for_bucket()` 是占位符 | `bucketing.py:448`：`return "SeasonalNaive"  # Placeholder for TSFM` | ✅ 问题确认 |
| C9 | tsfm_cache.py 文档说 "weak references" 但实际用普通字典 | `tsfm_cache.py:19-25`：docstring 说 "Uses weak references" 但 `_cache` 是 `dict` | ✅ 文档与实现不一致 |
| C10 | 缺少 Schema-Driven API Discovery | 无 `tsagentkit.describe()` 或机器可读 API schema | ✅ 有价值但属 P2 |
| C11 | 缺少 Dry-Run 模式 | `run_forecast` 无 dry_run 参数 | ✅ 有价值但属 P1 |
| C12 | 缺少 MCP Server | 不存在 | 📌 超出 v1.1.1 scope → 放入 roadmap |

---

## 二、条目聚合 → 7 大工作流

通过对表可以看到三份计划有大量交叉重叠。归类聚合后形成 7 个工作流：

```
┌───────────────────────────────────────────────────────────────────────────┐
│  WS-1  修 Bug / 修硬编码 / 修假测试              ← B3,B4,C6,C7,C8,C9,A2 │
│  WS-2  补测试覆盖                                 ← C1,C2,C3,C4,C5       │
│  WS-3  错误体系升级 (结构化修复提示)               ← A3,A6,C11            │
│  WS-4  安装分层 + 预设模板                        ← B1,B2,A1,A5           │
│  WS-5  类型安全加固 (mypy)                        ← B7                    │
│  WS-6  文档/Skill 治理                            ← B6,A2,A5              │
│  WS-7  架构扩展性 (插件化路由/API Discovery)       ← B3,C10,A7,C12        │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 三、最终方案：分优先级详细设计

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### WS-1: 修 Bug / 修硬编码 / 修假测试 [🔴 P0]
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**原则：先止血，不引入新 API**

#### 1.1 修复 reconciliation 硬编码（来源: C7）

**现状**：`orchestration.py:841` → `method_str = "bottom_up"` 写死
**方案**：从 `TaskSpec` 或 `run_forecast()` 参数读取

```python
# --- serving/orchestration.py ---
# Before:
method_str = "bottom_up"

# After: 从 TaskSpec.hierarchy_config 读取，兼容默认值
method_str = (
    task_spec.hierarchy_config.get("reconciliation_method", "bottom_up")
    if task_spec.hierarchy_config
    else "bottom_up"
)
```

同时在 `TaskSpec` 中增加 `hierarchy_config: dict | None = None` 字段，或在 `run_forecast()` 增加 `reconciliation_method` 参数。

#### 1.2 修复 bucketing 占位符（来源: C8）

**现状**：`bucketing.py:448` → `return "SeasonalNaive"  # Placeholder for TSFM`
**方案**：让 `get_model_for_bucket()` 感知 TSFM policy

```python
# --- router/bucketing.py ---
def get_model_for_bucket(bucket: SeriesBucket, tsfm_policy=None) -> str:
    if bucket == SeriesBucket.HEAD and tsfm_policy and tsfm_policy.mode != "disabled":
        return f"tsfm-{tsfm_policy.adapters[0]}" if tsfm_policy.adapters else "SeasonalNaive"
    # ... existing logic for other buckets
```

#### 1.3 修复 tsfm_cache 文档不一致（来源: C9）

**现状**：docstring 说 "Uses weak references" 但用普通 `dict`
**方案**：二选一 — ①修改 docstring 移除 weak reference 描述 ②改用 `weakref.WeakValueDictionary`
**建议**：选 ①（改 docstring），因为 TSFM 模型需要常驻内存，weak ref 会导致意外释放

#### 1.4 修复假测试（来源: C6）

**现状**：`test_v02_integration.py` 中 `test_full_pipeline_with_monitoring` 核心调用被注释
**方案**：恢复 `run_forecast()` 调用或将测试标记为 `@pytest.mark.skip(reason="...")`，不要留"看起来通过但什么也没测"的假测试

#### 1.5 修复 covariate 降级不透明（来源: B4）

**现状**：非 strict 模式静默丢弃协变量，仅在 `qa_report.issues` 中记录
**方案**：在 `RunArtifact` 中增加 `degradation_events: list[dict]` 字段，显式记录所有降级动作

```python
# RunArtifact 或 Provenance 中增加
degradation_events: list[dict] = []
# 每次降级时:
degradation_events.append({
    "step": "covariate_alignment",
    "action": "dropped_covariates",
    "reason": str(e),
    "severity": "warning",
})
```

#### 1.6 Router 候选模型可配置化（来源: B3, 部分）

**现状**：`router.py` 中 intermittent/short/default 候选列表硬编码
**方案**：将候选列表移入 `RouterThresholds`（已有该 dataclass），增加字段：

```python
# --- contracts/task_spec.py (RouterThresholds 中增加) ---
intermittent_candidates: list[str] = ["Croston", "Naive"]
short_history_candidates: list[str] = ["HistoricAverage", "Naive"]
default_candidates: list[str] = ["SeasonalNaive", "HistoricAverage", "Naive"]
```

`router.py` 从 `spec.router_thresholds` 读取，而非硬编码。

---

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### WS-2: 补测试覆盖 [🔴 P0]
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**原则：核心模块 ≥ 80% 行覆盖率**

| 缺失模块 | 行数 | 测试文件待创建 | 测试要点 |
|----------|------|---------------|---------|
| `eval/` | 286 | `tests/eval/test_evaluate.py` | `evaluate_forecasts()` 各指标正确性; `MetricFrame` 聚合; `ScoreSummary` 序列化; 边界: 空 df, 单序列, NaN |
| `calibration/` | 137 | `tests/calibration/test_calibration.py` | `fit_calibrator()` + `apply_calibrator()` 端到端; `CalibratorArtifact` 序列化; 边界: 无分位数输入 |
| `time/` | 135 | `tests/time/test_time_utils.py` | `infer_freq()` 对各频率准确性; `make_regular_grid()` 填充逻辑; `make_future_index()` 生成正确; 边界: 混合频率, 不规则间隔 |
| `monitoring/alerts.py` | 303 | `tests/monitoring/test_alerts.py` | `AlertCondition` 评估; `AlertManager` 触发/静默/恢复; 边界: 空历史, 阈值恰好 |
| `utils/temporal.py` | — | `tests/utils/test_temporal.py` | `drop_future_rows()` 正确裁剪; 时区处理; 边界: 无未来行 |
| `utils/signature.py` | — | `tests/utils/test_signature.py` | `compute_data_signature()` 确定性; 不同数据不同哈希; 边界: 空 df |
| `features/tsfeatures_adapter.py` | — | `tests/features/test_tsfeatures_adapter.py` | 适配器正确提取特征; import 失败时的 graceful fallback |

**预计新增 7 个测试文件，约 40-60 个测试函数**

---

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### WS-3: 错误体系升级 [🟡 P1]
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**原则：让每个错误都告诉 agent "怎么修"**

#### 3.1 为 TSAgentKitError 增加 `fix_hint` 字段（来源: A3, C11 融合）

```python
# --- contracts/errors.py ---
class TSAgentKitError(Exception):
    error_code: str = "E_UNKNOWN"
    fix_hint: str = ""  # 新增：可执行的修复提示

    def __init__(self, message, context=None, fix_hint=None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        if fix_hint:
            self.fix_hint = fix_hint

    def to_agent_dict(self) -> dict:
        """返回 agent 可直接消费的结构化错误信息"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "fix_hint": self.fix_hint,
            "context": self.context,
        }
```

#### 3.2 为高频错误预置修复提示

| 错误类 | 当前 message | 新增 fix_hint |
|--------|-------------|---------------|
| `EDSNotMonotonic` | "Time index not monotonic per series." | `"df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)"` |
| `EContractMissingColumn` | "Missing column: {col}" | `"确保 DataFrame 包含 'unique_id', 'ds', 'y' 列。可用 df.rename(columns={...}) 映射。"` |
| `EContractDuplicateKey` | "Duplicate keys found" | `"df = df.drop_duplicates(subset=['unique_id', 'ds'], keep='last')"` |
| `ECovariateLeakage` | "Covariate leaks into future" | `"将 past-only 协变量标记为 role='past'，或使用 align_covariates() 自动对齐"` |
| `ETSFMRequiredUnavailable` | "TSFM required but unavailable" | `"安装 TSFM: pip install tsagentkit[tsfm]，或设置 tsfm_policy={'mode': 'preferred'} 允许回退"` |
| `EFallbackExhausted` | "All fallback candidates failed" | `"检查数据是否满足最低要求（≥2 个观测值），或放宽 router_thresholds"` |

#### 3.3 增加独立的 `repair()` 入口（来源: A6）

```python
# --- 新增 tsagentkit/repair.py ---
from tsagentkit.contracts import ValidationReport

def repair(df, report: ValidationReport) -> pd.DataFrame:
    """根据 ValidationReport 自动应用修复。返回修复后的 df。"""
    if report.has_error("E_DS_NOT_MONOTONIC"):
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    if report.has_error("E_CONTRACT_DUPLICATE_KEY"):
        df = df.drop_duplicates(subset=["unique_id", "ds"], keep="last")
    # ... 其他可安全自动修复的问题
    return df
```

#### 3.4 增加 Dry-Run 验证模式（来源: C11）

```python
# --- 在 run_forecast() 增加参数 ---
def run_forecast(data, task_spec, mode="standard", *, dry_run=False, ...):
    """
    dry_run=True 时：只执行 validate → QA → make_plan，
    返回 ValidationResult 而非 RunArtifact。
    """
```

---

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### WS-4: 安装分层 + 预设模板 [🟡 P1]
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**原则：让 `pip install tsagentkit` 轻量可用，TSFM 按需安装**

#### 4.1 拆分安装档位（来源: B1）

```toml
# --- pyproject.toml ---
[project]
dependencies = [
    # 核心（轻量）：~50MB
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "pydantic>=2.0.0",
    "typing-extensions>=4.0.0",
    "scipy>=1.11.3,<1.12.0",
    "statsforecast>=1.7.0",
    "utilsforecast>=0.1.0",
]

[project.optional-dependencies]
tsfm = [
    # TSFM 全量：~5GB（含 PyTorch）
    "torch",
    "huggingface-hub",
    "chronos-forecasting>=2.0.0",
    "tsagentkit-timesfm",
    "tsagentkit-uni2ts",
    "gluonts",
]
hierarchy = [
    "hierarchicalforecast>=1.0.0",
]
features = [
    "tsfeatures>=0.4.5",
    "tsfresh>=0.20.0",
    "sktime>=0.24.0",
]
full = [
    "tsagentkit[tsfm,hierarchy,features]",
]
dev = [
    "tsagentkit[full]",
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "mypy>=1.0.0",
    "ruff>=0.1.0",
    "import-linter>=2.0.0",
]
```

**安装命令矩阵**：
```
pip install tsagentkit              # 核心 (~50MB) - 基线模型可用
pip install tsagentkit[tsfm]        # + TSFM 适配器 (~5GB)
pip install tsagentkit[full]        # 全部功能
pip install tsagentkit[dev]         # 开发环境
```

#### 4.2 TSFM 适配器延迟导入保护

确保 `from tsagentkit import ...` 在无 torch 时不报错：

```python
# --- models/adapters/__init__.py ---
def _lazy_import_chronos():
    try:
        from .chronos import ChronosAdapter
        return ChronosAdapter
    except ImportError:
        return None
```

当前代码已有部分延迟导入保护，需审查确保**所有入口路径**在无 torch 时 graceful。

#### 4.3 TaskSpec 场景预设（来源: A1, B2 融合）

```python
# --- contracts/task_spec.py 增加工厂方法 ---
class TaskSpec:
    @classmethod
    def starter(cls, h: int, freq: str = "D") -> "TaskSpec":
        """最小配置预设，tsfm_policy=preferred，适合快速试验"""
        return cls(
            h=h,
            freq=freq,
            tsfm_policy={"mode": "preferred"},
            backtest={"n_windows": 2},
        )

    @classmethod
    def production(cls, h: int, freq: str = "D") -> "TaskSpec":
        """生产配置预设，tsfm_policy=required，完整 backtest"""
        return cls(h=h, freq=freq)  # 默认即 production-grade
```

Agent 使用：
```python
spec = TaskSpec.starter(h=7)     # 5 秒上手
spec = TaskSpec.production(h=7)  # 生产部署
```

#### 4.4 quickstart 便捷函数（来源: A5）

```python
# --- 新增 tsagentkit/quickstart.py ---
def forecast(df, horizon, freq=None):
    """零配置快速预测。自动推断频率、标准化列名、处理常见问题。"""
    from tsagentkit import TaskSpec, run_forecast
    from tsagentkit.time import infer_freq

    # 自动列名映射
    df = _auto_rename_columns(df)
    # 自动排序
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    # 自动推断频率
    if freq is None:
        freq = infer_freq(df)

    spec = TaskSpec.starter(h=horizon, freq=freq)
    return run_forecast(df, spec, mode="quick")

def diagnose(df):
    """数据诊断报告。返回结构化的数据质量和推荐信息。"""
    ...
```

---

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### WS-5: 类型安全加固 [🟡 P1]
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**原则：渐进式启用 mypy，不搞一刀切**

#### 5.1 分模块渐进启用

```toml
# --- pyproject.toml ---
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
show_error_codes = true
ignore_missing_imports = true

# 全局宽松，按模块收紧
[[tool.mypy.overrides]]
module = "tsagentkit.contracts.*"
disallow_untyped_defs = true
check_untyped_defs = true
ignore_errors = false

[[tool.mypy.overrides]]
module = "tsagentkit.errors.*"
disallow_untyped_defs = true
ignore_errors = false

# 逐步扩展到 series, time, utils...
```

#### 5.2 阶段目标

| 阶段 | 覆盖模块 | 目标 |
|------|---------|------|
| v1.1.1 | `contracts/`, `errors/`, `time/`, `utils/` | 核心数据类型安全 |
| v1.2 | `router/`, `eval/`, `calibration/` | 计算逻辑安全 |
| v1.3 | `models/`, `serving/` | 全量覆盖 |

---

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### WS-6: 文档/Skill 治理 [🟡 P1]
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**原则：单一信息源 + 自动同步**

#### 6.1 消除文档重复（来源: B6）

```bash
# 删除 src/tsagentkit/skill/ 中的重复文件，改为 symlink
rm src/tsagentkit/skill/README.md src/tsagentkit/skill/recipes.md src/tsagentkit/skill/tool_map.md
ln -s ../../../../skill/README.md src/tsagentkit/skill/README.md
ln -s ../../../../skill/recipes.md src/tsagentkit/skill/recipes.md
ln -s ../../../../skill/tool_map.md src/tsagentkit/skill/tool_map.md
```

或更好的方案：在 `pyproject.toml` 构建钩子中复制：

```toml
[tool.hatch.build.hooks.custom]
# 构建时从 skill/ 复制到 src/tsagentkit/skill/
```

`test_phase6_doc_consistency.py` 的 byte-identical 测试作为**兜底**保留。

#### 6.2 Skill 文档增补（来源: A5, A1）

在 `skill/` 下新增两个文件：

```
skill/
├── README.md              # 现有（保持）
├── recipes.md             # 现有（保持）
├── tool_map.md            # 现有（保持）
├── QUICKSTART.md          # 新增：3 分钟上手指南
└── TROUBLESHOOTING.md     # 新增：常见错误 → 修复代码速查
```

**QUICKSTART.md** 结构：
```markdown
# 3 分钟快速上手

## 最小示例（5 行代码）
from tsagentkit.quickstart import forecast
result = forecast(df, horizon=7)

## 标准流程（Assembly-First）
... (10 行完整示例)

## 选择 TaskSpec 预设
- TaskSpec.starter(h=7)    → 快速实验
- TaskSpec.production(h=7) → 生产部署
```

**TROUBLESHOOTING.md** 结构：
```markdown
# 常见错误速查

| 错误码 | 含义 | 修复代码 |
|--------|------|---------|
| E_DS_NOT_MONOTONIC | 时间索引未排序 | `df = df.sort_values(...)` |
| E_TSFM_REQUIRED_UNAVAILABLE | 未安装 TSFM | `pip install tsagentkit[tsfm]` 或设置 preferred |
| ... | ... | ... |
```

#### 6.3 CLAUDE.md 更新

- 更新版本号 `1.1.0 → 1.1.1`
- 增加安装档位说明
- 增加 `TaskSpec.starter()` / `TaskSpec.production()` 用法
- 增加 `quickstart` 模块入口

---

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### WS-7: 架构扩展性（远期） [🟢 P2-P3]
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**原则：v1.1.1 只做设计，不做实现**

| 编号 | 条目 | 归属计划 | v1.1.1 行动 |
|------|------|---------|------------|
| 7.1 | 路由插件化 (YAML/JSON plan 模板) | B3 | 设计 RFC 文档，不实现 |
| 7.2 | `tsagentkit.describe()` API Discovery | C10 | 可做简版：返回 `tool_map.md` 的结构化 dict |
| 7.3 | `suggest_spec()` 决策助手 | A7 | 可做简版：基于 `infer_freq()` + 行数给推荐 |
| 7.4 | MCP Server | C12 | 放入 v1.2 roadmap |
| 7.5 | Pipeline checkpoint/resume | C11(部分) | 放入 v1.2 roadmap |
| 7.6 | CLI 三件套 (`init/doctor/run`) | B5 | v1.1.1 做 `tsagentkit doctor`（环境检查） |
| 7.7 | Guardrail 可配置化 | C12 | 与 WS-1 的 `RouterThresholds` 可配置化合并 |
| 7.8 | 可执行 examples 模块 | C11(部分) | v1.1.1 quickstart 模块覆盖基础场景 |

---

## 四、实施路线图

```
v1.1.1-alpha  (Week 1-2)
━━━━━━━━━━━━━━━━━━━━━━━
  [WS-1] 修复所有 P0 bug / 硬编码 / 假测试
  [WS-2] 补全 7 个测试文件

v1.1.1-beta   (Week 3-4)
━━━━━━━━━━━━━━━━━━━━━━━
  [WS-3] 错误体系升级 (fix_hint + repair() + dry_run)
  [WS-4] 安装分层 + TaskSpec 预设 + quickstart 模块
  [WS-5] mypy 渐进启用 (contracts/ + time/ + utils/)

v1.1.1-rc     (Week 5)
━━━━━━━━━━━━━━━━━━━━━━━
  [WS-6] 文档治理 (symlink + QUICKSTART.md + TROUBLESHOOTING.md)
  [WS-7] tsagentkit doctor CLI + describe() 简版
  全量回归测试 + CI 验证

v1.1.1        (Week 6)
━━━━━━━━━━━━━━━━━━━━━━━
  Release
```

---

## 五、变更影响评估

### 破坏性变更 (Breaking Changes)

| 变更 | 影响 | 缓解措施 |
|------|------|---------|
| 核心依赖移入 `[tsfm]` extra | `pip install tsagentkit` 不再默认包含 torch | README + 错误消息引导安装 `[tsfm]` |
| `hierarchicalforecast` 移入 `[hierarchy]` extra | hierarchy 功能需显式安装 | 延迟导入 + 明确 ImportError |

### 非破坏性变更 (Backward Compatible)

| 变更 | 说明 |
|------|------|
| `TaskSpec.starter()` / `.production()` | 新增工厂方法，原 API 不变 |
| `TSAgentKitError.fix_hint` | 新增属性，默认空字符串 |
| `repair()` 函数 | 新增独立入口 |
| `quickstart` 模块 | 全新模块 |
| `dry_run` 参数 | `run_forecast()` 新增可选参数 |
| Router 候选可配置 | `RouterThresholds` 增加字段，有默认值 |
| Reconciliation method 可配置 | 新增参数，默认值 `"bottom_up"` 保持向后兼容 |

---

## 六、不做什么（Explicitly Out of Scope for v1.1.1）

| 条目 | 原计划 | 原因 |
|------|--------|------|
| MCP Server | C12 | 工作量大，需独立设计 → v1.2 |
| Pipeline checkpoint/resume | C11 | 需要重构 orchestration.py → v1.2 |
| 完整路由插件化 (YAML plan 模板) | B3 | 需要 RFC 讨论 → v1.2 |
| `result.plot()` 可视化 | A7 | 不属于核心 agent 工具链 → v1.3 |
| 分布式/流式预测 | Roadmap | 远期 |
| Covariate API 彻底重构 | A4 | 影响面过大 → v1.2 评估 |

---

## 七、成功指标

| 指标 | 当前值 | v1.1.1 目标 |
|------|--------|------------|
| 核心模块测试覆盖 | eval/calibration/time 为 0% | ≥ 80% |
| 假测试数量 | 1 | 0 |
| 硬编码数量 | 5 处 | 0 处 |
| 安装体积（核心） | ~5GB (含 torch) | ~50MB |
| 错误含修复提示的比例 | 0/28 | 10/28 (高频错误全覆盖) |
| mypy 有效覆盖模块 | 0 | ≥ 4 模块 |
| Agent 最小上手代码行数 | 15+ 行 | 2 行 (`forecast(df, 7)`) |
| 文档重复文件 | 6 个 (3对) | 0 (symlink) |

---

## 附录 A：条目溯源矩阵

```
WS-1 ← A2, B3, B4, C6, C7, C8, C9
WS-2 ← C1, C2, C3, C4, C5
WS-3 ← A3, A6, C11
WS-4 ← A1, A5, B1, B2
WS-5 ← B7
WS-6 ← A2, A5, B6
WS-7 ← A7, B3, B5, C10, C12
```

每个原始条目都有归属，无遗漏。

## 附录 B：文件变更清单预估

| 操作 | 文件 | WS |
|------|------|-----|
| 修改 | `src/tsagentkit/contracts/errors.py` | WS-3 |
| 修改 | `src/tsagentkit/contracts/task_spec.py` | WS-1, WS-4 |
| 修改 | `src/tsagentkit/router/router.py` | WS-1 |
| 修改 | `src/tsagentkit/router/bucketing.py` | WS-1 |
| 修改 | `src/tsagentkit/serving/orchestration.py` | WS-1, WS-3 |
| 修改 | `src/tsagentkit/serving/tsfm_cache.py` | WS-1 |
| 修改 | `src/tsagentkit/contracts/results.py` | WS-1 |
| 修改 | `src/tsagentkit/__init__.py` | WS-4 |
| 修改 | `pyproject.toml` | WS-4, WS-5 |
| 修改 | `CLAUDE.md` | WS-6 |
| 修改 | `tests/test_v02_integration.py` | WS-1 |
| 新增 | `src/tsagentkit/quickstart.py` | WS-4 |
| 新增 | `src/tsagentkit/repair.py` | WS-3 |
| 新增 | `tests/eval/test_evaluate.py` | WS-2 |
| 新增 | `tests/calibration/test_calibration.py` | WS-2 |
| 新增 | `tests/time/test_time_utils.py` | WS-2 |
| 新增 | `tests/monitoring/test_alerts.py` | WS-2 |
| 新增 | `tests/utils/test_temporal.py` | WS-2 |
| 新增 | `tests/utils/test_signature.py` | WS-2 |
| 新增 | `tests/features/test_tsfeatures_adapter.py` | WS-2 |
| 新增 | `skill/QUICKSTART.md` | WS-6 |
| 新增 | `skill/TROUBLESHOOTING.md` | WS-6 |
