# Mathematical Verifier: Safety Proof System

## 🔬 Research Vision

This folder contains the **Mathematical Verifier** component of the research vision:

```
┌─────────────────┐
│  Math Verifier  │
│                 │
│ • Safety Proofs │
│ • Constraints   │
│ • Guarantees    │
│ • Formal Logic  │
└─────────────────┘
```

## 🎯 Core Purpose

The Mathematical Verifier focuses on:
- **Formal safety verification** using mathematical proofs
- **Constraint enforcement** through formal logic
- **Safety guarantees** with mathematical rigor
- **Rule-based validation** of actions and plans

## 🔧 Key Components

### `rules/verify.py`
- Core rule verification system
- Precondition and invariant checking
- Safety constraint enforcement
- Mathematical proof generation

### `planner/search.py`
- BFS search with verification
- Only expands verified-safe actions
- Mathematical safety guarantees
- Formal correctness proofs

### `eval/metrics.py`
- Safety metrics and statistics
- Verification performance analysis
- Mathematical proof validation
- Formal correctness measures

## ✅ What This System Does

- **Mathematical verification** of all actions
- **Safety guarantees** through formal proofs
- **Constraint enforcement** using formal logic
- **Rule-based validation** with mathematical rigor

## 🚫 What This System Does NOT Do

- **No creative planning** (that's the LLM's job)
- **No natural language understanding** (LLM handles that)
- **No high-level reasoning** (LLM provides that)
- **No plan generation** (LLM generates, we verify)

## 🔄 Integration

This system works with the **LLM as Brain** (`llm_as_the_brain/`):
1. LLM generates creative high-level plan
2. **This system** verifies mathematical safety
3. If safe: plan gets executed
4. If unsafe: feedback sent back to LLM

## 📊 Research Contribution

This demonstrates how **mathematical verification** can provide **rigorous safety guarantees** for LLM-generated plans, ensuring safety without limiting creativity.