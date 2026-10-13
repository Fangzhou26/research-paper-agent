const state = {
  project: null,
  runId: null,
  source: null,
  events: [],
  tasks: new Map(),
  artifact: null,
  liveReady: false,
};

const elements = Object.fromEntries([
  "projectSelect", "projectTitle", "researchQuestion", "paperCount", "noteCount",
  "evidenceCount", "corpusCoverage", "paperList", "runButton", "faultToggle",
  "runStatus", "runId", "eventCount", "citationState", "planState", "planTasks",
  "timeline", "citationGuard", "artifactContent", "provenance", "artifactId",
  "artifactPaper", "artifactEvidence", "copyArtifact", "clearTrace", "runSubtitle",
  "objectiveInput", "faultControl", "connectionState",
  "seedModeControl", "seedModeToggle", "seedSetup", "topicInput",
  "innovationInput", "seedFiles", "seedFileState", "paperLimit",
].map((id) => [id, document.getElementById(id)]));

const eventNames = {
  run_started: "Run accepted",
  workspace_prepared: "Workspace isolated",
  planner_started: "Planner started",
  discovery_started: "Discovery started",
  seed_loaded: "Seed PDF loaded",
  queries_planned: "Queries planned",
  search_completed: "Search completed",
  screening_completed: "Screening completed",
  discovery_source_failed: "Discovery source failed",
  plan_validated: "Plan validated",
  task_started: "Task started",
  tool_requested: "Tool requested",
  tool_succeeded: "Tool succeeded",
  tool_failed: "Tool failed",
  provider_failed: "Provider unavailable",
  run_blocked: "Run checkpointed",
  recovery_started: "Recovery started",
  task_finishing: "Task completed",
  citation_rejected: "Citation rejected",
  run_completed: "Run completed",
  run_stopped: "Run stopped",
  run_failed: "Run failed",
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

async function loadProjects() {
  const [payload, health] = await Promise.all([api("/api/projects"), api("/api/health")]);
  state.liveReady = Boolean(health.live_model?.ready);
  elements.connectionState.innerHTML = `<i></i> ${state.liveReady ? escapeHtml(health.live_model.model) : "Demo ready"}`;
  elements.projectSelect.innerHTML = payload.projects.map((project) =>
    `<option value="${escapeHtml(project.project_id)}">${escapeHtml(project.title)}</option>`
  ).join("");
  if (!payload.projects.length) throw new Error("No research project is available");
  showProject(payload.projects[0]);
}

function showProject(project) {
  state.project = project;
  elements.projectTitle.textContent = project.title;
  elements.researchQuestion.textContent = project.research_question;
  elements.objectiveInput.value = project.research_question;
  elements.topicInput.value = project.research_question;
  elements.seedFiles.value = "";
  updateSeedFileState();
  elements.paperCount.textContent = project.stats.papers;
  elements.noteCount.textContent = project.stats.notes;
  elements.evidenceCount.textContent = project.stats.evidence;
  elements.corpusCoverage.textContent = `${project.stats.notes} / ${project.stats.papers}`;
  elements.paperList.innerHTML = project.papers.map((paper) => `
    <div class="paper-item ${paper.has_note ? "has-note" : ""}">
      <strong>${escapeHtml(paper.title)}</strong>
      <div class="paper-meta">
        <span>${paper.year || "Year unknown"}</span>
        <span>${paper.has_note ? "Note ready" : "Not analyzed"}</span>
        <span>${paper.evidence_count} evidence</span>
      </div>
    </div>
  `).join("");
}

async function startRun() {
  if (!state.project) return;
  const seedDiscovery = currentMode() === "live" && elements.seedModeToggle.checked;
  const innovationPoints = inputLines(elements.innovationInput.value);
  const seedFiles = [...elements.seedFiles.files];
  if (seedDiscovery && (!elements.topicInput.value.trim() || !innovationPoints.length || !seedFiles.length)) {
    failRun("Topic, innovation points, and at least one seed PDF are required.");
    return;
  }
  if (seedFiles.length > 3) {
    failRun("Choose no more than three seed PDFs.");
    return;
  }
  resetRun();
  elements.runButton.disabled = true;
  elements.runButton.textContent = "Running";
  setStatus("running");
  elements.runSubtitle.textContent = "Executing validated tasks in an isolated workspace.";
  try {
    const seedUploadIds = seedDiscovery ? await uploadSeedPapers(seedFiles) : [];
    const run = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({
        project_id: state.project.project_id,
        mode: currentMode(),
        objective: currentMode() === "live" && !seedDiscovery ? elements.objectiveInput.value.trim() : null,
        simulate_failure: elements.faultToggle.checked,
        topic: seedDiscovery ? elements.topicInput.value.trim() : null,
        innovation_points: seedDiscovery ? innovationPoints : [],
        seed_upload_ids: seedUploadIds,
        max_discovered_papers: Number(elements.paperLimit.value),
      }),
    });
    state.runId = run.run_id;
    elements.runId.textContent = run.run_id;
    subscribe(run.run_id);
  } catch (error) {
    failRun(error.message);
  }
}

async function uploadSeedPapers(files) {
  const uploads = [];
  for (const file of files) {
    elements.runSubtitle.textContent = `Uploading ${file.name}`;
    const upload = await api(`/api/projects/${encodeURIComponent(state.project.project_id)}/seed-papers`, {
      method: "POST",
      headers: {
        "Content-Type": "application/pdf",
        "X-Filename": encodeURIComponent(file.name),
      },
      body: file,
    });
    uploads.push(upload.upload_id);
  }
  return uploads;
}

function subscribe(runId) {
  if (state.source) state.source.close();
  state.source = new EventSource(`/api/runs/${runId}/events`);
  state.source.addEventListener("run_event", (message) => {
    const event = JSON.parse(message.data);
    state.events.push(event);
    elements.eventCount.textContent = state.events.length;
    handleEvent(event);
  });
  state.source.onerror = async () => {
    if (!state.runId) return;
    const run = await api(`/api/runs/${state.runId}`).catch(() => null);
    if (run?.finished) {
      state.source.close();
      if (run.status === "failed") failRun(run.error || "Run failed");
    }
  };
}

function handleEvent(event) {
  const type = event.event_type;
  const data = event.data || {};
  addTimeline(event);
  if (type === "plan_validated") {
    elements.planState.textContent = "Passed";
    state.tasks = new Map(data.tasks.map((task) => [task.task_id, { ...task, status: "pending" }]));
    renderPlan();
  } else if (type === "task_started") {
    updateTask(data.task_id, "running");
  } else if (type === "task_finishing") {
    const taskId = data.task_id || [...state.tasks.values()].find((task) => task.status === "running")?.task_id;
    if (taskId) updateTask(taskId, data.status || "completed");
  } else if (type === "provider_failed") {
    setStatus("blocked");
    const active = [...state.tasks.values()].find((task) => task.status === "running");
    if (active) updateTask(active.task_id, "blocked");
  } else if (type === "recovery_started") {
    setStatus("recovering");
    updateTask(data.task_id, "running");
  } else if (type === "citation_rejected") {
    elements.citationState.textContent = "Rejected unsafe key";
    elements.citationGuard.className = "guard-content rejected";
    elements.citationGuard.innerHTML = `
      <strong>Unsafe candidate blocked</strong>
      <p>Unknown citation key: <code>${escapeHtml(data.unknown_keys.join(", "))}</code></p>
    `;
  } else if (type === "discovery_started") {
    elements.planState.textContent = "Discovering";
  } else if (type === "screening_completed") {
    elements.planState.textContent = `${data.accepted_count} papers`;
    elements.runSubtitle.textContent = `${data.accepted_count} papers admitted from ${data.candidate_count} candidates.`;
  } else if (type === "run_completed") {
    if (state.source) state.source.close();
    for (const task of state.tasks.values()) task.status = "completed";
    renderPlan();
    setStatus("completed");
    elements.citationState.textContent = "Passed";
    elements.runSubtitle.textContent = "All tasks completed with citation and provenance checks.";
    loadArtifact();
    finishControls();
  } else if (type === "run_failed") {
    if (state.source) state.source.close();
    failRun(data.error || "Run failed");
  } else if (type === "run_stopped") {
    if (state.source) state.source.close();
    setStatus(data.status || "blocked");
    elements.runSubtitle.textContent = data.message || "The run needs attention before it can continue.";
    finishControls();
  }
}

function updateTask(taskId, status) {
  const task = state.tasks.get(taskId);
  if (!task) return;
  task.status = status;
  renderPlan();
}

function renderPlan() {
  if (!state.tasks.size) return;
  elements.planTasks.classList.remove("empty-state");
  elements.planTasks.innerHTML = [...state.tasks.values()].map((task, index) => `
    <div class="plan-task ${task.status}">
      <span class="task-index">${index + 1}</span>
      <div>
        <strong>${escapeHtml(task.question)}</strong>
        <span>${escapeHtml(task.task_type)}</span>
      </div>
      <span class="task-status">${escapeHtml(task.status)}</span>
    </div>
  `).join("");
}

function addTimeline(event) {
  if (elements.timeline.classList.contains("empty-state")) {
    elements.timeline.classList.remove("empty-state");
    elements.timeline.innerHTML = "";
  }
  const tone = event.event_type.includes("failed") || event.event_type === "run_blocked"
    ? "failure"
    : event.event_type === "recovery_started"
      ? "recovery"
      : event.event_type.includes("completed") || event.event_type === "tool_succeeded"
        ? "success" : "";
  const detail = eventDetail(event);
  const row = document.createElement("div");
  row.className = `timeline-event ${tone}`;
  row.innerHTML = `
    <span class="event-time">${formatTime(event.timestamp)}</span>
    <div class="event-body">
      <strong>${escapeHtml(eventNames[event.event_type] || event.event_type)}</strong>
      <p>${escapeHtml(detail)}</p>
    </div>
  `;
  elements.timeline.appendChild(row);
  elements.timeline.scrollTop = elements.timeline.scrollHeight;
}

function eventDetail(event) {
  const data = event.data || {};
  if (event.event_type === "tool_requested") return data.tool_name || "Tool call";
  if (["tool_succeeded", "tool_failed"].includes(event.event_type)) return data.tool_name || "Tool result";
  if (event.event_type === "task_started") return data.question || data.task_id;
  if (event.event_type === "task_finishing") return data.summary || "Task finished";
  if (event.event_type === "provider_failed") return data.message || "Provider unavailable";
  if (event.event_type === "citation_rejected") return `Unknown: ${(data.unknown_keys || []).join(", ")}`;
  if (event.event_type === "plan_validated") return `${data.tasks.length} tasks admitted`;
  if (event.event_type === "planner_started") return data.objective || "Planning";
  if (event.event_type === "discovery_started") return data.topic || "Expanding seeds";
  if (event.event_type === "seed_loaded") return data.title || data.paper_id;
  if (event.event_type === "queries_planned") return `${(data.queries || []).length} queries via ${data.planner_mode}`;
  if (event.event_type === "search_completed") return `${data.result_count} results for ${data.query}`;
  if (event.event_type === "screening_completed") return `${data.accepted_count} of ${data.candidate_count} candidates accepted`;
  if (event.event_type === "discovery_source_failed") return data.message || data.stage;
  if (event.event_type === "workspace_prepared") return data.paper_id || "Workspace ready";
  if (event.event_type === "recovery_started") return `Retrying ${data.task_id}`;
  if (event.event_type === "run_stopped") return data.message || data.status;
  return data.message || data.status || data.run_id || "Recorded";
}

async function loadArtifact() {
  try {
    const artifact = await api(`/api/runs/${state.runId}/artifact`);
    state.artifact = artifact;
    elements.artifactContent.classList.remove("empty-state");
    elements.artifactContent.textContent = artifact.content;
    elements.provenance.hidden = false;
    elements.artifactId.textContent = artifact.manifest.artifact_id;
    elements.artifactPaper.textContent = (artifact.manifest.paper_ids || []).join(", ");
    elements.artifactEvidence.textContent = (artifact.manifest.evidence_ids || []).join(", ");
    elements.copyArtifact.disabled = false;
    elements.citationGuard.className = "guard-content passed";
    elements.citationGuard.innerHTML = `
      <strong>Final artifact accepted</strong>
      <p>All citation keys resolved. Paper and evidence provenance are attached.</p>
    `;
  } catch (error) {
    elements.artifactContent.textContent = error.message;
  }
}

function resetRun() {
  if (state.source) state.source.close();
  state.runId = null;
  state.events = [];
  state.tasks = new Map();
  state.artifact = null;
  elements.runId.textContent = "Starting";
  elements.eventCount.textContent = "0";
  elements.citationState.textContent = "Pending";
  elements.planState.textContent = "Waiting";
  elements.planTasks.className = "plan-tasks empty-state";
  elements.planTasks.textContent = "Waiting for validated plan";
  elements.timeline.className = "timeline empty-state";
  elements.timeline.textContent = "Connecting to event stream";
  elements.citationGuard.className = "guard-content pending";
  elements.citationGuard.innerHTML = "<strong>Awaiting artifact</strong><p>No citation checks have run.</p>";
  elements.artifactContent.className = "artifact-content empty-state";
  elements.artifactContent.textContent = "The synthesis will appear here";
  elements.provenance.hidden = true;
  elements.copyArtifact.disabled = true;
}

function setStatus(status) {
  elements.runStatus.className = `status-badge ${status}`;
  elements.runStatus.textContent = status.replace("_", " ");
}

function failRun(message) {
  setStatus("failed");
  elements.runSubtitle.textContent = message;
  finishControls();
}

function finishControls() {
  elements.runButton.disabled = currentMode() === "live" && !state.liveReady;
  elements.runButton.textContent = currentMode() === "live" ? "Run live" : "Run again";
}

function currentMode() {
  return document.querySelector('input[name="runMode"]:checked').value;
}

function syncMode() {
  const live = currentMode() === "live";
  const seedDiscovery = live && elements.seedModeToggle.checked;
  elements.objectiveInput.disabled = !live || seedDiscovery;
  elements.faultControl.hidden = live;
  elements.seedModeControl.hidden = !live;
  elements.seedSetup.hidden = !seedDiscovery;
  elements.runButton.textContent = live ? "Run live" : "Run demo";
  elements.runButton.disabled = live && !state.liveReady;
  if (live && !state.liveReady) {
    elements.runSubtitle.textContent = "Live model requires the configured API key.";
  } else {
    elements.runSubtitle.textContent = live
      ? "The planner and agent will use the configured model."
      : "Ready to execute an isolated research workflow.";
  }
}

function inputLines(value) {
  return value.split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
}

function updateSeedFileState() {
  const files = [...elements.seedFiles.files];
  elements.seedFileState.textContent = files.length
    ? `${files.length} PDF${files.length === 1 ? "" : "s"} selected`
    : "No PDFs selected";
}

function formatTime(timestamp) {
  return new Date(timestamp).toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

elements.runButton.addEventListener("click", startRun);
elements.clearTrace.addEventListener("click", () => {
  elements.timeline.className = "timeline empty-state";
  elements.timeline.textContent = "Trace cleared";
});
elements.copyArtifact.addEventListener("click", async () => {
  if (!state.artifact) return;
  await navigator.clipboard.writeText(state.artifact.content);
  elements.copyArtifact.textContent = "Copied";
  setTimeout(() => { elements.copyArtifact.textContent = "Copy"; }, 1200);
});
elements.projectSelect.addEventListener("change", async (event) => {
  const project = await api(`/api/projects/${encodeURIComponent(event.target.value)}`);
  showProject(project);
});
document.querySelectorAll('input[name="runMode"]').forEach((input) => {
  input.addEventListener("change", syncMode);
});
elements.seedModeToggle.addEventListener("change", syncMode);
elements.seedFiles.addEventListener("change", updateSeedFileState);

loadProjects().then(syncMode).catch((error) => failRun(error.message));
