---
name: long-answer-markdown
description: When an answer is likely to exceed 50 lines, write the response as a Markdown document file instead of printing the full content in chat. Use when the user asks for long explanations, long plans, long reviews, long tutorials, long experiment notes, or any answer that may exceed 50 lines.
argument-hint: [optional topic or target markdown file]
---

# Long Answer Markdown Skill

Use this skill when the requested response is likely to be longer than 50 lines.

## Goal

Keep chat concise by writing long content into a Markdown file and returning only a short summary plus the file path.

## Behavior

Before drafting the full answer, estimate whether the final response would exceed 50 lines.

If the answer is likely to be 50 lines or fewer:

- Answer normally in chat.

If the answer is likely to exceed 50 lines:

1. Create a Markdown file for the full response.
2. Put the complete answer in that file.
3. In chat, respond with only:
   - the file path;
   - a 3-8 bullet summary;
   - any important caveat or next action.

## File location

Default to writing under:

```text
./docs/answers/
```

Use a short kebab-case filename based on the topic, for example:

```text
./docs/answers/seq2seq-experiment-tracking.md
```

If `$ARGUMENTS` contains a target path ending in `.md`, use that path instead.

## Markdown requirements

The Markdown file should be self-contained and readable later without needing the chat history.

Include when relevant:

- title;
- date;
- context;
- main explanation;
- code snippets;
- recommended next steps;
- pitfalls;
- summary.

## Chat response format

After writing the Markdown file, keep the chat response short:

```markdown
已将完整内容写入：`<path>`

摘要：
- ...
- ...
- ...
```

Do not paste the full Markdown content back into chat unless the user explicitly asks.

## Important notes

- Do not use this skill for short answers.
- Do not create a file if the user explicitly asks to keep the full answer in chat.
- If writing a file would overwrite existing content, read the existing file first and avoid destructive overwrite unless clearly intended.
