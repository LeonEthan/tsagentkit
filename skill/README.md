# tsagentkit Skill

This folder contains the **tsagentkit Claude Skill** - a modular guide for building production time-series forecasting systems.

## Structure

```
skill/
├── SKILL.md                  # Main skill file (entry point)
├── README.md                 # This file
├── scripts/                  # Executable utilities
│   ├── forecast.py          # CLI forecast runner
│   ├── health_check.py      # System health check
│   └── validate_data.py     # Data validation
├── references/              # Detailed documentation
│   ├── quickstart.md        # 3-minute getting started
│   ├── recipes.md           # End-to-end templates
│   ├── tool_map.md          # Task-to-API lookup
│   └── troubleshooting.md   # Error codes and fixes
└── assets/                  # Templates and assets (empty)
```

## Quick Usage

When this skill is loaded, Claude will have access to:

1. **SKILL.md** - Core guidance for forecasting tasks
2. **scripts/** - Executable Python scripts for common operations
3. **references/** - Detailed docs loaded on-demand

## Available Scripts

| Script | Purpose |
|--------|---------|
| `scripts/forecast.py` | Run forecasts from command line |
| `scripts/health_check.py` | Check system health and model availability |
| `scripts/validate_data.py` | Validate data format before forecasting |

## Reference Docs

| Doc | Purpose |
|-----|---------|
| `references/quickstart.md` | 3-minute getting started guide |
| `references/recipes.md` | Copy-paste ready code templates |
| `references/tool_map.md` | Find the right API for your task |
| `references/troubleshooting.md` | Error codes and solutions |

## For Skill Developers

This skill follows the official Anthropic skill format:
- YAML frontmatter in SKILL.md with `name` and `description`
- Imperative/infinitive writing style
- Progressive disclosure (metadata → SKILL.md → references)
- Bundled resources for reusable scripts
