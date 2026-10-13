# Coaching playbooks

Short, opinionated coaching guidance, one file per topic. Each chat turn, a
router picks the 0-2 playbooks relevant to the user's message and injects them
into that turn's instructions. `coaching-core.md` is not routed: it is always
part of the system prompt.

These replace the old AI-generated `knowledge_base/` + Qdrant retrieval. They are
not reference articles: the model already knows textbook material. A playbook
holds what it would not reliably do on its own: what to ask first, the positions
we take, the mistakes to steer people away from, and when to refer out.

## Format

```
---
id: fueling-and-hydration          # stable key the router returns; matches the filename
title: Fueling and hydration
description: One line the router reads to decide relevance. Name the concrete
  question types it covers.
---

Body: markdown, about 300-500 words. Positions, not prose.

## Sources
For human review only; stripped before injection.
```

## Writing rules

- Lead with "Ask first": the 1-3 facts that change the answer.
- State positions plainly ("we recommend", "don't"), with the number when there
  is one. Give ranges, not false precision.
- Keep numbers consistent with the calculator tools in `app/tools/` (zone models
  and protein ranges). If you change one, change the other.
- Medical red flags always win: point to `pain-and-injury` and never coach
  through them.
- Triathlon first, with single-sport notes where the advice differs.
- No content about subjects outside endurance coaching.
