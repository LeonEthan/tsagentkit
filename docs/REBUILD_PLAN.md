# tsagentkit Phase-by-Phase Rebuild Plan

本计划基于 `docs/ARCHITECTURE.md` 的架构约束，对现有代码进行梳理与差距分析，并给出可执行的分阶段重建方案，作为后续重构的统一参考。

## 目标与边界

目标：
- 保持 `run_forecast()`、`TaskSpec`、`validate_contract()`、`rolling_backtest()` 等稳定 API 语义不变。
- 对齐架构中的模块分层与依赖方向，清理跨层依赖与重复实现。
- 用成熟库替换自研指标/特征/层级算法，实现“thin adapter”策略。

非目标：
- 不在本次计划中确定具体模型选择与业务策略。
- 不承诺对外接口的破坏性变更，所有变更须提供兼容层或迁移路径。

## 现有代码梳理（概要）

代码结构（基于 `src/tsagentkit/`）：
- `contracts/`：`TaskSpec`、`PlanSpec`、错误与校验（Pydantic）
- `series/`：`TSDataset`、稀疏度分析、对齐与填补工具
- `time/`：频率推断、未来索引、规则网格
- `covariates/`：协变量角色识别、覆盖与泄露校验
- `features/`：本地特征工程与特征版本哈希
- `router/`：路由与回退、计划生成
- `models/`：StatsForecast 基线模型 + TSFM 适配器 + sktime 适配
- `backtest/`：滚动回测、指标计算、报告结构
- `eval/`：评估汇总（可选使用 utilsforecast）
- `hierarchy/`：层级结构、对齐与 reconciliation（部分自研矩阵逻辑）
- `calibration/`、`anomaly/`、`monitoring/`、`serving/`：校准、异常、监控与编排
- `skill/`：使用说明与 Agent 文档

现状与架构的主要差距：
- `backtest/metrics.py` 与 `eval/` 仍保留自研指标实现，架构要求优先 `utilsforecast.evaluate`。
- `features/` 采用本地手写特征工程，架构要求默认 `tsfeatures`，`tsfresh` 仅扩展。
- `hierarchy/aggregation.py` 等存在自研层级算法，架构要求只做契约校验与数据转换，算法交给 `hierarchicalforecast`。
- `series/alignment.py` 与 `utilsforecast` 的能力重复，需要明确保留/替换策略。
- repo 内存在 `__pycache__`、空 `adapters/`、空 `reconciliation/` 等残留，应纳入清理计划。

## Phase-by-Phase Rebuild 方案

**Phase 0: 基线冻结与依赖边界校验**

目标：建立“可对齐、可回滚”的重构基线，锁定稳定 API。

工作项：
- 明确稳定 API 与兼容边界（`run_forecast()`、`TaskSpec`、`ForecastResult`、`RunArtifact`）。
- 生成模块依赖关系图，确保层级依赖符合架构。
- 设定 lint/检查规则，禁止下层依赖上层。
- 清理 repo 中的 `__pycache__` 与空包残留（制定执行清单）。

验收标准：
- 稳定 API 清单与兼容承诺落地到 `docs/`。
- 依赖边界检查可在 CI/本地运行。

**Phase 1: 合同层与数据层重建（contracts/series/time/covariates）**

目标：保证底层契约与数据结构的稳定性与可审计性。

工作项：
- 将 `contracts/` 完全隔离为仅依赖 Pydantic/标准库。
- 统一 `PanelContract`/`ForecastContract` 的默认列名与映射策略。
- 统一 `TSDataset` 与 `validate_contract()` 的列规范化逻辑。
- `covariates/` 对齐策略与 `TaskSpec.covariate_policy` 完整对齐。

验收标准：
- `series/` 与 `covariates/` 不再依赖 `serving/`、`models/`。
- 合同校验错误码与上下文稳定。

**Phase 2: 特征工程重构（features/）**

目标：以 `tsfeatures` 为默认实现，保留必要的可扩展接口。

工作项：
- 新增 `features/tsfeatures_adapter.py`，`FeatureFactory` 默认走 `tsfeatures`。
- 将现有手写特征工程迁移至 `features/extra`，标注为非默认路径。
- 统一 `FeatureMatrix` 契约与哈希签名生成逻辑。

验收标准：
- 默认特征管道仅做薄适配，不再维护复杂算法。
- 所有特征输出遵循 `FeatureMatrix` 契约。

**Phase 3: 评估与回测重构（eval/backtest）**

目标：用 `utilsforecast.evaluate` 统一指标计算与汇总。

工作项：
- `eval/` 统一封装 `utilsforecast.evaluate`，产出长表 `MetricFrame` 与 `ScoreSummary`。
- `backtest/` 回测输出统一为 `CVFrame` + `BacktestReport`，指标计算迁移到 `eval/`。
- `backtest/metrics.py` 标记为 deprecated，并逐步移除。

验收标准：
- 任何指标计算路径均经过 `utilsforecast.evaluate`。
- 兼容原有 `BacktestReport` 字段与序列化接口。

**Phase 4: 层级重构（hierarchy/）**

目标：只保留契约校验与数据转换，算法交给 `hierarchicalforecast`。

工作项：
- 定义 `S_df`/`tags` 为标准输入，提供 `HierarchyStructure -> (S_df, tags)` 的转换器。
- 删除或迁移 `hierarchy/aggregation.py` 中的自研算法逻辑。
- `reconcile_forecasts()` 仅调用 `hierarchicalforecast`。

验收标准：
- 层级模块不再包含自研 reconciliation 算法。
- 输入输出严格对齐 `S_df`/`tags` 合同。

**Phase 5: 模型与编排对齐（models/serving）**

目标：让 `models/` 成为纯适配层，`serving/` 成为唯一编排入口。

工作项：
- `models/` 统一适配接口，基线模型继续使用 `statsforecast`，TSFM 仅做薄适配。
- `serving/` 中的步骤顺序与错误策略对齐 `ARCHITECTURE.md`。
- 明确 `run_forecast()` 语义与扩展点（校准、异常、监控）。

验收标准：
- `models/` 不再包含自研算法实现。
- `run_forecast()` 的输入输出结构保持稳定。

**Phase 6: 测试与文档收敛**

目标：确保重构后行为可验证、文档可执行。

工作项：
- 更新测试覆盖（特别是 backtest/eval/hierarchy/features 变化）。
- 更新 `docs/ARCHITECTURE.md` 与 `docs/README.md`，同步新边界与默认实现。
- 在 `skill/` 中补充迁移指南与新接口示例。

验收标准：
- 关键路径测试通过。
- 文档与代码行为一致。

## 风险与注意事项

- 指标与特征迁移会影响回测结果，需保留兼容层或比对开关。
- 层级 reconciliation 行为迁移应提供一致性校验与回归测试。
- `utilsforecast`、`tsfeatures`、`hierarchicalforecast` 依赖需纳入可选依赖管理。

## 预期产出清单

- `docs/REBUILD_PLAN.md`（本文件）
- 依赖边界检查配置
- 模块适配器与契约规范更新
- 回测与评估统一输出格式

