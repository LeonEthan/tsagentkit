
项目审视：问题与优化方向
我从"面向 coding agent 的工具箱"这个核心定位出发，对项目进行了全面审查。以下是我的分析：

一、代码层面的具体问题
1. 测试覆盖缺口
以下模块缺少专项测试，对于一个面向 agent 的工具库来说，这是可靠性风险：
- eval/ — 283 行的评估逻辑（evaluate_forecasts, WAPE, MASE, quantile loss），无测试文件
- calibration/ — 137 行的 conformal calibration 逻辑，无测试文件  
- time/ — 135 行的 infer_freq, make_regular_grid, make_future_index，无测试文件
- monitoring/alerts.py — 无专项测试
- utils/temporal.py, utils/signature.py — 无专项测试
- test_v02_integration.py 中的 test_full_pipeline_with_monitoring 实际上把核心 pipeline 调用注释掉了，形同虚设
2. 硬编码/占位符
- serving/orchestration.py 中层级协调方法被硬编码为 "bottom_up"，没有从 plan 配置读取
- router/bucketing.py 中 get_model_for_bucket() 对 HEAD bucket 返回 "SeasonalNaive"，注释标注 "Placeholder for TSFM"
- tsfm_cache.py 文档字符串提到"weak references"，但实现使用的是普通字典缓存
3. 文档重复
skill/ 和 src/tsagentkit/skill/ 下存在三组完全相同的文件（README.md, recipes.md, tool_map.md）。手动维护同步容易导致漂移，应考虑构建时复制或 symlink。
4. 函数签名不一致
CLAUDE.md 中的 step-level pipeline 示例与 skill/README.md 中 fit()/predict() 的参数顺序/名称不完全一致，可能误导 agent。

二、面向 Coding Agent 的架构优化方向
5. 缺少 Schema-Driven 的 API Discovery 机制
当前 agent 依赖阅读 markdown 文档来理解 API。建议：
- 为每个公开 API 提供机器可读的 schema（如 JSON Schema 或 Pydantic model export），让 agent 能通过代码内省而非文档阅读来发现 API
- 考虑提供一个 tsagentkit.describe() 或 tsagentkit.api_map() 函数，返回结构化的 API 清单
- 这比 tool_map.md 更可靠——agent 在代码中调用比解析 markdown 更不容易出错
6. 错误信息对 Agent 不够友好
当前的错误码系统（E_SPLIT_RANDOM_FORBIDDEN 等）是好的设计，但错误消息主要面向人类开发者。建议：
- 每个错误附带 structured remediation hint，比如 {"error": "E_COVARIATE_LEAKAGE", "fix": "call mask_observed_for_training() before fit", "example": "..."}
- Agent 拿到这种结构化错误后可以自动修复，而不需要去查文档理解错误含义
7. Pipeline 的可观测性和断点续跑
run_forecast() 是一个近 1000 行的编排函数，步骤间紧耦合。对 agent 来说：
- 如果某个中间步骤失败，agent 无法从断点恢复，只能重跑整个 pipeline
- 建议支持 checkpoint/resume 机制：每个步骤完成后持久化中间状态，允许 agent 从上一个成功步骤继续
- 或者提供一个 PipelineRunner 类，显式管理步骤状态、支持 skip/retry 单个步骤
8. 缺少 Dry-Run / Validation-Only 模式
Agent 在帮用户搭建 pipeline 时，经常需要先验证配置是否正确而不实际执行。建议：
- 提供 validate_pipeline(data, task_spec, ...) -> ValidationResult，只做 schema 校验、数据质量检查、模型可用性检查，不实际训练
- 这让 agent 可以快速迭代配置而不浪费计算资源
9. 缺少版本化的 Example/Template 系统
skill/recipes.md 的 recipes 是静态文本。建议：
- 将 recipes 做成可执行的 Python 模块（如 tsagentkit.examples.retail_daily），agent 可以 import 并修改参数
- 每个 example 附带最小可运行数据集（synthetic data generator）
- 这样 agent 可以 from tsagentkit.examples import retail_daily; retail_daily.run(my_data) 式地搭积木
10. MCP Server / Tool Use 集成缺失
项目目标是让 coding agent 使用，但当前 agent 只能通过写 Python 代码来调用。考虑：
- 提供一个 MCP (Model Context Protocol) Server 实现，让 Claude Code / OpenCode 等工具可以直接通过 tool use 调用 tsagentkit 的 API，而不需要 agent 先写代码再执行
- 每个 step API 对应一个 MCP tool，参数用 JSON Schema 描述
- 这是"低门槛搭积木"最直接的路径
11. 数据诊断与自动推荐能力薄弱
当前 router 的 make_plan() 做了基础的统计判断（间歇性、稀疏性、季节性），但：
- 缺少面向 agent 的数据诊断报告：agent 拿到一份数据后，应该能调用一个函数得到"这份数据适合什么模型、可能遇到什么问题、建议怎么处理"的结构化建议
- 建议提供 diagnose(data) -> DiagnosisReport，包含数据质量评分、推荐模型、推荐 horizon、潜在风险
12. 缺少 Guardrail 的可配置性
当前的 guardrail（如禁止随机切分、covariate leakage 检查）是硬编码的。对于高级用户/agent 来说：
- 应该支持按需启用/禁用特定 guardrail
- 通过一个 GuardrailConfig 统一管理，而不是散落在各个模块中


三、优先级建议
| 优先级 | 项目 | 理由 |
|--------|------|------|
| P0 | 补全 eval/calibration/time 的测试 | 核心功能无测试，agent 依赖的可靠性基础 |
| P0 | 修复硬编码（reconciliation method, bucketing placeholder） | 直接影响输出正确性 |
| P1 | 结构化错误 + remediation hint | Agent 自愈能力的基础 |
| P1 | diagnose() 数据诊断 API | Agent"搭积木"的第一步 |
| P1 | Dry-run / validation-only 模式 | 降低 agent 试错成本 |
| P2 | MCP Server 集成 | 真正做到"低门槛" |
| P2 | 可执行 examples 模块 | 替代静态 recipes |
| P2 | Schema-driven API discovery | 减少 agent 对文档的依赖 |
| P3 | Pipeline checkpoint/resume | 长 pipeline 的鲁棒性 |
| P3 | Guardrail 可配置化 | 灵活性 |

总体而言，项目的核心架构设计（分层、assembly-first、TSFM-first、immutable data contracts）是扎实的。主要差距在于：从"一个好的 Python 库"到"一个对 coding agent 真正友好的工具箱"之间还有不少路要走。当前更像是给懂时序的开发者用的库，而不是让 agent 能低门槛地帮不懂时序的用户搭系统。