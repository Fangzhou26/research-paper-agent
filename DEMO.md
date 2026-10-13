# Research Agent Demo

## Run

From the repository root:

```bash
.venv/bin/research-agent demo
```

The demo is offline and does not read an API key. It starts from the repository's
small, fully synthetic sample project, creates an isolated workspace under
`/tmp`, and leaves the generated artifact there for inspection.

For the browser workspace, run:

```bash
.venv/bin/research-agent serve
```

Then open `http://127.0.0.1:8000`. The interface streams the same execution as
structured events and exposes the validated plan, failure recovery, tool trace,
citation gate, final artifact, and provenance manifest.

The workspace also has a `Live model` mode. It accepts a research objective and
runs the configured LLM planner and executor in an isolated copy of the selected
corpus. The UI disables this mode when its API-key environment variable is not
available. Live runs consume provider quota; the default Demo mode does not.

Enable `Seed discovery` inside Live mode to run the full MVP from a topic,
innovation points, and 1--3 uploaded seed PDFs. The PDFs are validated, copied
into the isolated workspace, ingested, and extracted before discovery starts.
The discovery stage records query decomposition, provider results, screening
decisions, and warnings before the accepted corpus enters the Planner. OpenAlex and
Semantic Scholar are attempted first; arXiv provides a no-key abstract fallback.

To choose a workspace explicitly:

```bash
.venv/bin/research-agent demo --workspace /tmp/my-research-agent-demo
```

The target directory must be empty. To show only the successful path:

```bash
.venv/bin/research-agent demo --no-failure
```

## What It Demonstrates

1. A fictional paper record, note, evidence passage, and page cache are copied
   into an isolated demo workspace. No real paper or private project files are
   required.
2. A three-task plan is validated before execution.
3. The scripted provider fails on its first decision.
4. The executor persists a blocked checkpoint.
5. An explicit retry resumes the task from its saved conversation.
6. The agent uses the production dispatcher to search pages, read evidence,
   load a note, resolve a citation key, and write an artifact.
7. The citation audit detects the deliberately invented key `invented2026`.
8. The final artifact passes citation-key validation and includes a provenance
   manifest with paper and evidence IDs.

The provider is scripted only to make the interview demo deterministic. The
plan validation, task executor, tool allowlists, dispatcher, toolkit,
checkpoints, citation audit, and output writer are production components.

## Suggested Interview Walkthrough

Point out these fields in the JSON result:

- `failure`: the first run stopped at the analysis task;
- `recovery`: the same plan completed after `retry_blocked`;
- `citation_guard`: the invented citation was detected;
- `tool_trace`: every action crossed the closed dispatcher;
- `output`: the final draft and provenance manifest can be inspected directly.

Then contrast the stable offline demo with the live command:

```bash
.venv/bin/research-agent run
```

The live path uses the configured model and API key while preserving the same
execution and safety boundaries.
