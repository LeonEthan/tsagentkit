你的方向是对的：**“assembly-first + guardrail + artifact”**非常适合 coding agent。
但从“低门槛搭积木”视角看，当前项目更像“专家库”，还没完全产品化为“agent builder kit”。

主要问题（按影响排序）

- 默认安装太重：核心依赖直接包含 torch、chronos、gluonts 等，首次接入成本高、失败率高（pyproject.toml:7）。
- 默认策略偏硬：TSFMPolicy.mode 默认 required，且无可用适配器会直接失败，对新手/轻量场景不友好（src/tsagentkit/contracts/task_spec.py:74, src/tsagentkit/router/router.py:81）。
- “搭积木”扩展点还不够插件化：路由候选和模型分发有较多硬编码（src/tsagentkit/router/router.py:92, src/tsagentkit/models/__init__.py:35）。
- wrapper 存在隐式降级：非 strict 下 covariate 错误会“成功继续并丢弃协变量”，对 agent 来说可解释性不足（src/tsagentkit/serving/orchestration.py:212, src/tsagentkit/serving/
orchestration.py:224）。
- agent 入口不够友好：缺少真正 CLI/脚手架，main.py 目前只是 hello world（main.py:1）。
- 文档链路有重复维护成本：skill/ 与包内镜像强一致，易产生维护负担（tests/docs/test_phase6_doc_consistency.py:12）。
- 类型质量护栏偏弱：mypy 处于 ignore_errors=true，对 agent 自动改代码时不够“防呆”（pyproject.toml:78）。

建议路线（你这个定位下最值钱）

- P0（先做）：拆分安装档位（core/tsfm/full）；新增 TaskSpec 场景预设（starter/pro）；把“降级动作”变成显式策略和结构化状态。
- P1：把 router/model/fallback 变成可注册模块（YAML/JSON plan 模板）；提供机器可读 tool_map.json 给 agent 直接消费。
- P2：补 tsagentkit init / tsagentkit doctor / tsagentkit run 三件套 CLI，让 Codex/Claude Code 能一条命令落地样板系统。