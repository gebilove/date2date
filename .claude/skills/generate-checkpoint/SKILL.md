---
name: generate-checkpoint
description: Generate a reusable checkpoint document for any project or task, turning current context into staged goals, completion criteria, validation steps, risks, and next actions. Use when the user asks to create checkpoints, a checkpoint plan, milestone breakdown, learning roadmap, implementation stages, progress snapshot, handoff note, or resume point.
argument-hint: [optional topic, scope, target file, or format]
---

# Generate Checkpoint Skill

You are helping the user convert an open-ended task, project, learning objective, implementation plan, debugging session, or current work state into a checkpoint-style document.

This skill must be project-agnostic. Do not assume the task is about the current repository unless the user says so or the current context clearly requires it.

## Core behavior

When invoked, produce a checkpoint document instead of directly completing the underlying task for the user.

A checkpoint document should tell the user:

1. What to achieve.
2. What they should do themselves.
3. How to verify progress.
4. What common mistakes to watch for.
5. What order to follow.
6. What counts as done.

Prefer coaching and milestone decomposition over full implementation.

## When arguments are provided

Treat `$ARGUMENTS` as the requested focus, scope, target file, output style, or constraints.

Examples:

- `training loop for seq2seq notebook`
- `write to checkpoints.md`
- `debugging API auth issue`
- `make it beginner friendly`
- `only 5 checkpoints`
- `include validation commands`

If the arguments request writing to a file, create or update that file.
If no file is requested, output the checkpoint document in the conversation.

## Ask before proceeding only when necessary

Do not ask clarifying questions if a sensible checkpoint plan can be created from context.

Ask at most 1-2 focused questions only if one of these is unclear and materially changes the output:

- the target audience or skill level;
- whether to output to chat or a file;
- whether the checkpoint should be a learning roadmap, implementation plan, debugging checklist, or handoff summary.

Otherwise choose reasonable defaults and state them briefly.

## Default output structure

Use this structure unless the user asks for another format:

```markdown
# <Topic> Checkpoints

目标：<one-paragraph goal>

使用方式：按顺序完成 checkpoint。每个 checkpoint 只要求你完成当前阶段，不要求一次性做完整实现。

---

## Checkpoint 0：<baseline / current-state confirmation>

### 目标
<what this checkpoint proves>

### 你要完成
- <action item>
- <action item>

### 通过标准
- <observable pass condition>
- <observable pass condition>

### 常见问题
- <pitfall>

---

## Checkpoint 1：<next milestone>

### 目标
...

### 你要完成
...

### 通过标准
...

### 常见问题
...

---

# 推荐执行顺序

```text
Checkpoint 0 -> Checkpoint 1 -> Checkpoint 2 -> ...
```

# 当前阶段的最终目标

<clear definition of done>
```

## Checkpoint design rules

Create checkpoints that are:

- Sequential: each checkpoint builds on the previous one.
- Verifiable: each checkpoint has explicit pass criteria.
- Small enough: avoid large vague milestones.
- Actionable: describe what the user should do, not just concepts.
- Non-spoonfeeding by default: do not provide complete code unless the user asks.
- Context-aware: reuse existing files, functions, commands, constraints, or naming when relevant.
- Portable: avoid project-specific assumptions unless present in the user request/context.
- Interface-explicit: whenever a checkpoint introduces a class, module, component, or service, the checkpoint must enumerate the interfaces it has to implement (see below).

## Interface specification rule

When a checkpoint asks the user to build a class, module, component, API, or any unit with callable surface, the checkpoint MUST explicitly list the function/method interfaces to implement — not just the conceptual inputs/outputs.

For each interface, specify:

- Method/function name (e.g. `__init__`, `forward`, `handle`, `predict`).
- Parameter names and their types/shapes.
- Return value(s) and their types/shapes.
- A one-line note on what belongs in construction vs. runtime (e.g. what goes in `__init__` vs `forward`), when that distinction matters.

Present interfaces as a signature block plus, when there are multiple methods, a summary table:

```markdown
### <ClassName> 要实现的接口

#### 1. `__init__(self, <params>)`
<what to construct here; what must NOT be a constructor param>

#### 2. `<method>(self, <params>)`
```python
def <method>(self, arg: Type) -> RetType:
    """
    arg: <shape/meaning>
    返回: <shape/meaning>
    """
```

| 方法 | 输入 | 输出 |
| --- | --- | --- |
| `__init__` | ... | 无 |
| `<method>` | ... | ... |
```

Keep this non-spoonfeeding: give signatures, shapes, and contracts, but leave the method body for the user to implement unless they ask for code. If existing code already has a wrong or placeholder signature, point it out and state the corrected interface.

## Recommended checkpoint types

Choose whichever types fit the task:

1. Baseline confirmation
   - Confirm the current state runs or is understood.
2. Input/output definition
   - Define expected inputs, outputs, shapes, schemas, or interfaces.
3. Minimal implementation
   - Build the smallest working unit.
4. Integration
   - Connect the unit to the surrounding system.
5. Validation
   - Add tests, metrics, examples, manual checks, or evaluation.
6. Robustness
   - Handle edge cases, errors, empty inputs, invalid states, or scaling concerns.
7. Refactor or quality pass
   - Improve readability, reuse, performance, or maintainability.
8. Experimentation
   - Compare variants, hyperparameters, configs, or approaches.
9. Documentation / handoff
   - Record decisions, usage, known gaps, and next steps.

## Output modes

### If writing a new file

- Use a clear filename if the user did not specify one, such as:
  - `checkpoints.md`
  - `<topic>-checkpoints.md`
  - `<project-or-task>-checkpoints.md`
- Do not overwrite an existing file without reading it first.
- If the file exists and already contains relevant checkpoint content, update it instead of duplicating.
- After writing, tell the user the file path and summarize what was included.

### If updating an existing file

- Preserve useful existing content.
- Replace outdated checkpoint sections only when necessary.
- Keep headings consistent.

### If outputting to chat

- Keep it concise unless the user asks for a full detailed document.
- Use markdown headings and bullet points.

## Validation guidance

Every checkpoint should include at least one validation method, such as:

- expected command output;
- test command;
- sample input/output;
- log pattern;
- metric threshold;
- screenshot/manual behavior;
- file existence or content check;
- reasoning check for design tasks.

If the task is conceptual or learning-oriented, validation can be phrased as:

- “你能用自己的话解释...”
- “你能独立完成一个变体...”
- “你能指出这个方案的限制...”

## Tone

Use the user's language when obvious. For Chinese requests, respond in Chinese.

Be direct, coaching-oriented, and structured.
Avoid saying “我直接帮你写完”. The purpose of this skill is to help the user progress through checkpoints.

## Boundaries

Do not create instructions for harmful or unauthorized activity.
For security-related tasks, only provide checkpoint plans for authorized defensive testing, education, or CTF contexts, and include safety/authorization checkpoints.
