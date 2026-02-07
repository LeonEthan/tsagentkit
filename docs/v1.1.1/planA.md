  ---
  整体评价

  tsagentkit 在 agent-ready 设计方面已经做得相当出色，但仍有几个关键维度可以优化。

  ---
  一、发现的问题

  1. 概念密度过高，新手agent门槛陡峭

  当前问题：
  - TaskSpec 有 20+ 个配置字段，新agent难以判断哪些是"必须理解"的
  - tsfm_policy 的默认 mode="required" 在 v1.1.0 变更，但文档中的示例没有突出这一点
  - Covariate 的三个角色（static/past/future_known）对新手不够直观

  # 当前：agent需要理解这么多概念才能开始
  spec = TaskSpec(
      h=7,
      freq="D",
      tsfm_policy={"mode": "required"},  # 默认值，但agent不知道
      covariates=CovariateSpec(roles={...}),  # 复杂的角色映射
      backtest=BacktestSpec(n_windows=5, ...),  # 嵌套配置
  )

  2. API 不一致性：wrapper vs assembly

  skill/README.md Pattern 1 vs Pattern 2 的参数不一致：

  # Pattern 2 (Wrapper) - 简单
  run_forecast(df, spec, mode="standard")

  # Pattern 1 (Assembly) - 发现的问题：
  # 1. validate_contract 接收的是 df，不是 spec
  # 2. fit 和 predict 的 covariates 参数位置不一致
  # 3. package_run 的参数列表冗长，需要手动传递太多东西

  Pattern 1 中的示例代码实际上无法运行（第72行 validate_contract(df) 缺少 spec 参数）。

  3. Error Recovery 缺乏"下一步行动"提示

  虽然错误代码很详细，但 agent 不知道：

  except EDSNotMonotonic:
      # 错误信息告诉我排序问题，但没有告诉我要：
      # df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

  4. Covariate 处理复杂度过高

  # 当前需要 agent 写的代码
  aligned = align_covariates(df, spec)
  dataset = build_dataset(aligned.panel, spec, validate=False).with_covariates(
      aligned,
      panel_with_covariates=df,  # 为什么要传两次 df？
  )

  5. 缺乏"渐进式学习"路径

  - 没有 "Hello World" 级别的最小示例
  - 6个recipes都是生产级复杂度，没有"从简单到复杂"的梯度
  - 缺少"常见错误及修复"的 troubleshooting guide

  ---
  二、优化方向建议

  建议 1：增加"渐进式 API 层"

  # Layer 0: 零配置快速开始（新增）
  from tsagentkit.quickstart import forecast  # 新增模块

  result = forecast(df, horizon=7)  # 只需要 df 和 horizon

  # Layer 1: 智能默认值（简化当前 API）
  from tsagentkit import TaskSpec, run_forecast

  spec = TaskSpec(h=7)  # freq 自动推断，TSFM 按需回退
  result = run_forecast(df, spec)

  # Layer 2: 当前 assembly-first（保留给高级用户）
  from tsagentkit import validate_contract, run_qa, ...

  # Layer 3: 完全自定义（当前 fit/predict 级别）

  建议 2：增加"自修复" wrapper

  from tsagentkit import auto_forecast  # 新增

  result = auto_forecast(df, horizon=7)
  # 自动处理：
  # - 列名标准化 (item_id/date/target -> unique_id/ds/y)
  # - 排序修复
  # - 频率推断
  # - 异常值处理
  # 返回详细日志说明做了什么修复

  建议 3：Covariate API 简化

  # 当前（复杂）
  cov_spec = CovariateSpec(roles={"promo": "future_known", "price": "past"})
  aligned = align_covariates(df, spec, covariates=cov_spec)
  dataset = build_dataset(...).with_covariates(aligned, ...)

  # 建议：inline 声明
  from tsagentkit import TaskSpec, Covariate

  spec = TaskSpec(
      h=7,
      covariates=[
          Covariate("promo", type="future_known"),  # 内联声明
          Covariate("price", type="past"),
      ]
  )
  # 内部自动处理 align_covariates

  建议 4：Agent-friendly Error Messages

  # 当前
  EDSNotMonotonic: "Time index not monotonic per series."

  # 建议（包含可执行修复代码）
  EDSNotMonotonic:
      "Time index not monotonic per series. Fix with:\n"
      "df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)"

  或者提供 repair() 方法：

  from tsagentkit import validate_contract, repair

  report = validate_contract(df, spec)
  if not report.valid:
      df = repair(df, report)  # 自动应用所有修复

  建议 5：增加"决策助手"（Agent Decision Helper）

  新增 skill/decision_tree.md 或代码内的交互式帮助：

  from tsagentkit.guide import suggest_spec  # 新增

  suggest_spec(df, goal="daily_sales_forecast")
  # 返回：
  # {
  #   "recommended_spec": {...},
  #   "reasoning": "检测到日频数据，建议使用 Chronos (TSFM)...",
  #   "warnings": ["部分序列历史较短 (< 56天)，可能影响精度"],
  #   "alternatives": [...]
  # }

  建议 6：改进 Skill 文档结构

  当前 skill/ 目录的文档是"参考型"，建议增加"教程型"：

  skill/
  ├── README.md              # 快速开始（简化版）
  ├── TUTORIAL.md            # 新增：从0到1的渐进教程
  │   ├── 01-hello-world.md
  │   ├── 02-handling-errors.md
  │   ├── 03-adding-covariates.md
  │   ├── 04-production-deployment.md
  │   └── 05-troubleshooting.md
  ├── recipes.md             # 现有：生产模板
  ├── tool_map.md            # 现有：API参考
  └── DECISION_GUIDE.md      # 新增：如何选择API/模式

  建议 7：Colab/Jupyter 友好的输出

  result = run_forecast(df, spec)

  # 新增：丰富的可视化摘要
  result.summary()  # 文本摘要（已有）
  result.plot()     # 新增：可视化预测结果
  result.diagnose() # 新增：数据质量和模型选择诊断图表

  ---
  三、优先级建议
  ┌────────┬───────────────────────────────────────────┬───────────────────┐
  │ 优先级 │                  优化项                   │       影响        │
  ├────────┼───────────────────────────────────────────┼───────────────────┤
  │ 🔴 P0  │ 修复 skill/README.md Pattern 1 的错误代码 │ 避免agent产生幻觉 │
  ├────────┼───────────────────────────────────────────┼───────────────────┤
  │ 🔴 P0  │ 增加 repair() 或错误消息包含修复代码      │ 大幅降低调试成本  │
  ├────────┼───────────────────────────────────────────┼───────────────────┤
  │ 🟡 P1  │ 增加 tsagentkit.quickstart 简化API层      │ 降低新手门槛      │
  ├────────┼───────────────────────────────────────────┼───────────────────┤
  │ 🟡 P1  │ 增加渐进式教程文档                        │ 改善学习曲线      │
  ├────────┼───────────────────────────────────────────┼───────────────────┤
  │ 🟢 P2  │ Covariate API 简化                        │ 减少概念数量      │
  ├────────┼───────────────────────────────────────────┼───────────────────┤
  │ 🟢 P2  │ suggest_spec() 决策助手                   │ 提升智能体验      │
  ├────────┼───────────────────────────────────────────┼───────────────────┤
  │ 🔵 P3  │ 可视化输出                                │ 增强交互体验      │
  └────────┴───────────────────────────────────────────┴───────────────────┘
  ---
  四、具体代码示例：理想中的新手体验

  import pandas as pd
  from tsagentkit.quickstart import forecast, diagnose

  # 1. 加载数据
  df = pd.read_csv("sales.csv")  # 列名任意：item_id, date, sales

  # 2. 诊断（可选但推荐）
  diagnose(df)
  # 输出：
  # ✓ 检测到 3 个时间序列
  # ✓ 频率: 每日 (D)
  # ✓ 平均历史长度: 365 天
  # ⚠ 建议: 使用 horizon=14 进行两周预测

  # 3. 预测（自动处理列名映射、排序、修复）
  result = forecast(df, horizon=14)

  # 4. 查看结果
  print(result.df)  # 预测值
  print(result.metrics)  # 回测指标
  result.plot()  # 可视化

  对比当前需要的代码：

  import pandas as pd
  from tsagentkit import TaskSpec, validate_contract, run_qa, build_dataset, make_plan
  from tsagentkit import fit, predict, package_run, align_covariates

  # 1. 加载并手动标准化
  df = pd.read_csv("sales.csv")
  df = df.rename(columns={"item_id": "unique_id", "date": "ds", "sales": "y"})
  df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

  # 2. 创建配置
  spec = TaskSpec(h=14, freq="D")

  # 3. 验证
  report = validate_contract(df, spec)
  report.raise_if_errors()

  # 4. QA
  qa = run_qa(df, spec)

  # 5. 构建数据集
  dataset = build_dataset(df, spec)

  # ... 等等

  ---
  总结

  tsagentkit 在架构设计和 production-readiness 方面已经很出色，但从"让 coding agent 低门槛、搭积木式使用"的角度，需要在概念简化、错误自修复、渐进式学习路径三个方向上投入更多。特别是

  1. 一个能跑的 "Hello World"（当前 Pattern 1 代码有bug）
  2. 错误即修复指南（不只是告诉你错了，还告诉你怎么修）
  3. 智能默认值（让 agent 用最少配置获得合理结果）

  这样能让 coding agent 更快上手，也更能体现"搭积木"的灵活性优势。