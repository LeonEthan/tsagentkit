# tsagentkit Architecture

> 目的：描述**真实实现的技术架构**、模块边界与依赖方向、扩展点与稳定 API，作为 PRD 的实现视角补充。
> 范围：代码结构、数据流与模块责任；不包含模型选型与业务策略细节。

---

## 1. 架构目标

- 保持**时序预测流程确定性**与可审计性
- 明确**模块边界与依赖方向**，降低耦合
- 提供**清晰扩展点**（模型、特征、校准、异常检测等）
- 面向**Coding Agent**调用，API 稳定、契约严格

## 2. 模块分层与依赖方向

建议并约束以下单向依赖（自下而上）：

1. `contracts/`、`errors/`
2. `series/`、`time/`、`covariates/`、`features/`
3. `router/`、`backtest/`、`eval/`、`calibration/`、`anomaly/`
4. `models/`
5. `serving/`

依赖方向规则：
- 上层可依赖下层，下层不可依赖上层
- `contracts/` 不依赖业务模块（仅 Pydantic/标准库）
- `serving/` 是唯一编排层

依赖细则（强约束）：
1. 禁止 `contracts/`、`series/`、`time/`、`covariates/`、`features/` 依赖 `serving/` 或 `models/`
2. 禁止 `eval/`、`backtest/`、`router/`、`calibration/`、`anomaly/` 依赖 `serving/`
3. 禁止在本库自研以下能力：统计/基线模型实现、层级一致性算法、通用指标函数
4. `hierarchy/` 只接受 `S_df`/`tags` 输入，不保留算法实现；仅做契约校验与数据转换
5. `eval/` 仅封装 `utilsforecast.evaluate`；指标函数不在本库重复实现
6. `features/` 默认使用 `tsfeatures`；`tsfresh` 仅在 `features/extra`
7. `models/` 仅做适配与参数映射，不复制 `statsforecast`/`sktime` 算法实现
8. `serving/` 是流水线封装层；对外推荐 assembly-first 逐步组合，`run_forecast()` 仅作为兼容包装器

## 3. Pipeline 数据流（实现视角）

推荐入口为 step-level 组合；`run_forecast()` 复用同一组 step API 进行包装：

1. `validate_contract()` → 结构与类型校验
2. `run_qa()` → 质量检测与可选修复
3. `align_covariates()` → 协变量对齐与泄露校验
4. `TSDataset.from_dataframe()` → 统一数据结构
5. `make_plan()` → 生成 `PlanSpec`
6. `build_plan_graph()` / `attach_plan_graph()` → 暴露可编排 DAG 节点（可选）
7. `rolling_backtest()` → 可选 CV 评估
8. `models.fit()` → 模型适配与训练
9. `models.predict()` → 预测输出
10. `fit_calibrator()` / `apply_calibrator()` → 置信度校准（可选）
11. `detect_anomalies()` → 异常检测（可选）
12. `package_run()` → 打包 `RunArtifact`

回测策略说明：
回测逻辑保持自研实现，以兼容 TSFM 与自定义预测来源；指标与汇总优先复用 `utilsforecast.evaluate`。

### 流程示意图

```mermaid
flowchart TD
    A[Input Panel Data] --> B[Validate Contract]
    B --> C[QA Checks & Repairs]
    C --> D[Align Covariates]
    D --> E[Build TSDataset]
    E --> F[Make Plan]
    F --> G[Backtest (optional)]
    G --> H[Fit Model]
    H --> I[Predict]
    I --> J[Calibrate (optional)]
    J --> K[Anomaly (optional)]
    K --> L[Package RunArtifact]
```

## 4. 核心模块职责

- `contracts/`：输入输出契约、`TaskSpec`/`PlanSpec` 等配置模型
- `series/`：`TSDataset` 统一数据对象、稀疏度分析、时间对齐
- `time/`：频率推断、未来索引构建
- `covariates/`：协变量分类、覆盖检测、泄露检测
- `features/`：可重复特征工程与签名
- `router/`：确定性路由与回退策略
- `models/`：模型适配、TSFM/StatsForecast/Sktime 统一调用
- `backtest/`：滚动窗口 CV 与评估输出
- `eval/`：指标计算与汇总
- `calibration/`：区间/分位校准
- `anomaly/`：异常检测与评分
- `serving/`：流程编排与产物打包、结构化日志

## 5. 可维护性与复用策略（主流库对齐）

核心原则：**优先复用成熟库，保持薄适配层**，避免重复造轮子。以下为建议的职责对齐与集成方式。

建议复用对象与定位（默认实现优先级）：
- `statsforecast`：统计/基线模型与时间序列 CV 的默认实现，作为基础模型库优先接入。
- `utilsforecast`：评估函数 `evaluate`、预处理（如 `fill_gaps`）与轻量特征工程 `pipeline` 的默认实现。
- `hierarchicalforecast`：层级预测与一致性处理的**默认实现**，以 `S_df` 与 `tags` 作为标准输入。
- `tsfeatures`：时间序列统计特征提取的默认实现，输入为 `unique_id`/`ds`/`y` 面板数据。
- `sktime`：传统模型生态与转换器作为可选扩展入口（不是默认）。
- `tsfresh`：高维特征自动提取的可选扩展能力（更重，非默认）。

数据格式对齐（契约驱动）：
- `utilsforecast` 的 `evaluate` 与预处理函数默认使用 `unique_id`/`ds`/`y` 约定，与 `PanelContract` 对齐。
- `tsfeatures` 接收 `unique_id`/`ds`/`y` 面板，并可基于频率生成特征，与 `PanelContract` 对齐。
- `hierarchicalforecast` 以 `S_df` 与 `tags` 作为层级约束输入，便于与 `ForecastResult` 与层级结构对接。

默认库映射表（面向维护成本）：

| 模块 | 默认实现 | 说明 |
| --- | --- | --- |
| `models/` | `statsforecast` | 统计/基线模型与时间序列 CV |
| `eval/` | `utilsforecast` | 指标函数与评估流程 |
| `hierarchy/` | `hierarchicalforecast` | 层级预测与一致性处理 |
| `features/` | `tsfeatures` | 统计特征提取（默认） |
| `features/extra` | `tsfresh`（可选） | 高维特征提取 |
| `models/sktime` | `sktime`（可选） | 传统模型生态扩展 |

替换策略（面向可维护性）：
- 统计模型与基线预测逻辑尽量迁移至 `statsforecast`，仅保留薄适配层。
- 层级预测与一致性处理优先由 `hierarchicalforecast` 承担，保留轻量 wrapper 做数据转换与契约校验。
- 评估指标与汇总统一迁移至 `utilsforecast`，避免自维护指标实现。
- 特征提取优先通过 `tsfeatures`，高维特征再考虑 `tsfresh`。
- 传统模型或特殊需求通过 `sktime` 接入。

集成与演进约束：
- `models/` 与 `features/` 仅做 **thin adapter**，不在内部复制算法实现。
- 非核心能力统一做 **optional extras**，核心流程不依赖外部大库即可运行。
- 保持 `ForecastContract` 与 `RunArtifact` 输出稳定，即使替换底层实现也不破坏契约。
- 复用优先级：已有成熟实现 > 轻量封装 > 自研（仅在无可用库或需严格 PIT 时）。

## 6. 扩展模块与插件点

### 模型扩展
- TSFM 通过 `models/adapters` 统一注册
- 传统模型通过 `models/sktime` 或基线模块接入
- `get_adapter_capability()` / `list_adapter_capabilities()` 提供 capability + runtime availability 视图，便于 agent 做策略分支

### 特征扩展
- `features/` 提供 `FeatureConfig` 与 `FeatureFactory`，默认接入 `tsfeatures`。
- Feature hash 用于审计与复现

### 监控与层级
- `monitoring/`：漂移/稳定性检测
- `hierarchy/` 与 `reconciliation/`：层级预测与一致性处理（默认使用 `hierarchicalforecast`，以 `S_df`/`tags` 为核心输入）

## 7. 数据契约与工件

### 契约
- `PanelContract`: `unique_id`, `ds`, `y`
- `ForecastContract`: `model`, `yhat`, quantiles/intervals

### 层级约束输入
`hierarchicalforecast` 以 `S_df` 与 `tags` 描述层级结构，`tsagentkit` 以该形式作为默认层级输入契约；任何层级结构需转换为 `S_df`/`tags`。

### 层级输入契约细则（S_df / tags / Y_hat_df）
- `S_df`：层级求和矩阵 DataFrame，行索引为全体序列 `unique_id`，列为底层序列 `unique_id`，值为聚合权重（常见为 0/1）。
- `tags`：字典，键为层级名称，值为该层级包含的序列 `unique_id` 列表。
- `Y_hat_df`：基础预测 DataFrame，包含 `unique_id`、`ds` 与模型列（每个模型一列）。
- `Y_df`：历史数据 DataFrame（可选），包含 `unique_id`、`ds`、`y`，用于某些一致性方法的校准。

### 回测输出标准（CVFrame / BacktestReport）
- `CVFrame`（长表）：`unique_id`, `ds`, `cutoff`, `model`, `y`, `yhat`, `q_*`（可选）
- `BacktestReport`：至少包含 `cv_frame`, `metrics`, `summary`, `errors`, `n_windows`, `horizon`
- 回测指标计算使用 `utilsforecast.evaluate`，内部允许将 `CVFrame` pivot 为宽表再计算。

### 评估输出标准（MetricFrame / ScoreSummary）
- `MetricFrame`（长表）：`unique_id`（可选）, `cutoff`（可选）, `model`, `metric`, `value`
- `ScoreSummary`：`model`, `metric`, `value` 的聚合结果
- `utilsforecast.evaluate` 原始输出为“每指标一行、模型列展开”的宽表，`tsagentkit` 统一转换为长表保存。

### 特征矩阵契约（FeatureMatrix）
- `FeatureMatrix.data` 必须包含 `unique_id`, `ds`, `y` 与特征列
- `feature_cols` 为特征列名，默认来源于 `tsfeatures` 输出
- 为避免与协变量/目标列冲突，特征列允许统一前缀 `tsf_`（若发生命名冲突）。
- 默认适配器建议实现于 `features/tsfeatures_adapter.py`，由 `FeatureFactory` 作为入口调用

### 主要工件
- `ForecastResult`：预测结果 + provenance
- `RunArtifact`：完整执行产物（含报告与 metadata）

## 8. 稳定 API 与内部 API

建议将以下接口视为**稳定 API**（assembly-first 主路径）：

- `validate_contract()`
- `run_qa()`
- `align_covariates()`
- `TSDataset.from_dataframe()` / `build_dataset()`
- `make_plan()`
- `build_plan_graph()` / `attach_plan_graph()`
- `rolling_backtest()`
- `models.fit()` / `models.predict()`
- `get_adapter_capability()` / `list_adapter_capabilities()`
- `package_run()`
- `save_run_artifact()` / `load_run_artifact()`
- `validate_run_artifact_for_serving()` / `replay_forecast_from_artifact()`
- `TaskSpec`
- `fit_calibrator()` / `apply_calibrator()`
- `detect_anomalies()`

兼容 API：
- `run_forecast()`（便利包装器，语义保持稳定）

兼容边界（必须保持）：
- step-level 组合语义与 `package_run()` 的产物字段语义
- `run_forecast()` 的包装语义与返回 `RunArtifact` 的字段名
- `PanelContract` / `ForecastContract` 的必需列与列名
- `ForecastResult.df` 的最小列集合：`unique_id`, `ds`, `model`, `yhat`

允许调整（保持上述契约前提）：
- `BacktestReport` 内部字段结构与聚合明细
- `MetricFrame` / `ScoreSummary` 的行排序与附加字段
- `PlanSpec` 内部候选模型列表与 Router 规则（不改变外部契约）

## 9. 目录结构（实现）

- `src/tsagentkit/`：库源码
- `docs/PRD.md`：产品需求与约束
- `docs/ARCHITECTURE.md`：技术架构（本文档）

## 10. 版本与兼容策略（建议）

- 仅在 minor/major 版本更新中引入破坏性 API 改动
- `RunArtifact`、`TaskSpec` 视为最优先兼容对象

## 11. 后续演进建议

- 将 `RouteDecision` 纳入产物并持久化
- 增强 step-level 示例与测试，防止文档回退到 wrapper-first
- 将 `contracts/` 与 `results/` 进一步解耦

## 12. 迁移与弃用清单（面向可维护性）

- `hierarchy/` 内自研一致性与评估逻辑逐步弃用，统一替换为 `hierarchicalforecast` 适配层
- `backtest/metrics.py` 与 `eval` 内自研指标逐步弃用，统一迁移至 `utilsforecast.evaluate`
- `features/` 内自研特征块逐步收敛为 `tsfeatures` 默认实现
- `HierarchyStructure` 保留为兼容层，但内部以 `S_df`/`tags` 为唯一权威输入
