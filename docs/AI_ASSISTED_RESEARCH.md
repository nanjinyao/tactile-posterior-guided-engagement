# AI-Assisted Research Workbook

This companion to the [illustrated README](../README.md) makes the proposed learning process operational. It contains evidence boundaries, reusable prompts, interface examples, and exercises for repeating the project with stronger personal understanding. The prompts and templates below are new suggestions. They are not verbatim historical conversations or evidence that the suggested exercises have already been completed.

## What the records establish

Three different roles need to remain clear:

| Role | Function | Appropriate evidence |
|---|---|---|
| Development assistant: ChatGPT, Claude, or Codex | Help read, propose, implement, debug, analyze, and write | Source passages, accepted/rejected suggestions, code changes, and validation records |
| Runtime learned model | Convert a tactile fingerprint into an inference output | Frozen data splits, calibration, model identity, input/output contracts, and task-level results |
| Researcher | Select questions, justify assumptions, design experiments, interpret evidence, and take responsibility for claims | Independent explanations, derivations, predictions, measurements, and decisions |

The [design decisions](../mujoco_unscrewing_probe_new/DESIGN_DECISIONS.md), entry `2026-07-09 23:23`, explicitly record a ChatGPT roadmap and the subsequent narrowing to an M8-first implementation. The local source-workspace `project_trace.md`, entry `2026-06-24 03:26:28 +08:00`, records selective use of Claude formalization suggestions: force-cone notation, separation of frame/clocking variables, a joint Bayesian transition, planning terminology, and predictive observation distributions for information gain. That local trace was not included in the uploaded snapshot; this paragraph provides a provenance summary rather than a link to an unavailable file. It concerns early reading-material revisions and does not establish that all those ideas were implemented.

[Project rules and lessons](../mujoco_unscrewing_probe_new/PROJECT_RULES_AND_LESSONS.md) and the design records document corrections involving true-centre leakage, diagnostic versus verification signals, fair baselines, and trajectory interpretation. They do not identify who first discovered every issue.

The 2026-10-27 packaging, README, reflection documents, restore tooling, and validation were prepared in a Codex-assisted session. The English narrative is an AI-assisted synthesis of existing evidence. No robot was connected, and no new training or full physical benchmark was performed during this documentation work. The [validation record](VALIDATION_20261027.md) lists checks actually completed.

There is no complete prompt archive, reliable per-file AI attribution, or measured time-saving estimate. Claims such as "the assistant invented the method," "all code was independently written," or "AI saved a particular percentage of time" would exceed the evidence.

## A small context packet for every task

An assistant is easier to assess when the question, evidence, and permitted change are explicit. Before a substantial request, write the following packet yourself:

```text
Question:
What precise uncertainty am I trying to resolve?

Current explanation:
What do I currently think, and what observation supports it?

Evidence:
Relevant paper sections, source files, configuration, run IDs, and plots.

Contract:
Inputs, outputs, units, frames, timing, and available information.

Constraints:
Which comparison conditions or interfaces must remain fixed?

Requested artifact:
A derivation, comparison table, minimal patch, diagnostic, or experiment plan.

Falsification:
What result would make me reject the current explanation?

My prediction:
What do I expect to happen, before inspecting the new result?
```

For example, "improve the shortcut" is underspecified. A useful packet instead states that direction is informative but radial distance is ambiguous, provides posterior samples and controller limits, and asks whether a particular action rule is excessively sensitive to reverse probability mass.

## Suggested prompts across the research cycle

### 1. Find the problem before selecting the algorithm

**First do independently:** draw the task, identify one failure, and list two possible physical causes.

```text
Here is my task sketch, one failure trace, and my current explanation.
Separate observed facts from inferred causes. Propose three alternative
explanations and the cheapest measurement that distinguishes each one.
Do not propose a new model until you identify what information is missing.
Help me state one falsifiable question with a measurable endpoint.
```

**Human check:** explain why answering this question would change a design decision. A problem statement that could fit almost any robotics task is still too broad.

### 2. Review literature with source-level accountability

**First do independently:** read the primary source and write a short summary of its observation, action, assumption, and evidence.

```text
Using only the supplied paper text and my notes, build a matrix of sensors,
observations, hidden variables, actions, assumptions, success criteria,
and evaluation conditions. Give a page or section for each important claim.
Mark missing information as unknown. Identify where my summary overstates
the paper and where its assumptions differ from this apparatus.
```

**Human check:** open every critical source location, explain the central equation without the summary, and identify one result worth reproducing. If further literature is needed, verify the actual paper before adding it to the argument. Assistant-generated references are candidates to check.

### 3. Formalize the problem without hiding assumptions

**First do independently:** define variables, dimensions, frames, observations, actions, and the endpoint.

```text
Review my formulation for dimensional consistency, frame conventions,
conditioning variables, and hidden access to ground truth. Expand one
Bayesian update step. Distinguish a model assumption, an empirical fact,
a heuristic, and a proved statement. Construct a counterexample to my
strongest claim, especially if direction is identifiable but distance is not.
```

**Human check:** reproduce the update on a small numerical example. If the method claims information gain, specify the predictive distribution over future observations. If it claims coverage or stability, state and test the assumptions required by that claim.

### 4. Design an informative simulation experiment

**First do independently:** declare the primary metric, information available to each method, fixed starts, budget, and split.

```text
Audit this comparison before suggesting performance changes. Look for true
centre leakage, unequal information, different recovery budgets, tuning on
test cases, and success-conditioned reporting. Propose the smallest ablation
that would show whether the bounded shortcut, rather than another component,
causes the observed path reduction. State what a negative result would mean.
```

**Human check:** inspect one success and one failure for every method. Reconstruct the summary from raw records, and report failures and censoring alongside conditional statistics. Keep exploratory changes away from the frozen final evaluation.

### 5. Generate CAD that can be measured and tested

**First do independently:** draw the mating interface, choose datums, state dimensions/tolerances, and identify the load path and print orientation.

```text
Use this dimension table and source interface to propose a parametric CAD
construction. Keep nominal geometry and printer-clearance variants explicit.
Explain the datum and each transform. Identify unsupported features, thin
sections, and operations likely to create invalid meshes. Produce a validation
plan that separates geometry checks, mesh checks, print fit, and installed
alignment. Do not infer a successful physical fit from a rendered image.
```

**Human check:** inspect sections, print a fit coupon, measure the part, and update the dimension table. Record any hardware geometry change as an experimental variable. See the [tooling README](../real_device/model/parametric_tooling/README.md) for the actual generator and artifacts.

### 6. Inspect a hardware interface and deployment path

**First do independently:** identify the command owner, sensing path, stop conditions, and distinction between commands and measurements.

```text
Trace the supplied saved cone from the sensor interface through coordinate
transforms, CSV fields, feature construction, model loading, posterior JSON,
and bounded action selection. List units, frames, timestamps, validity rules,
and executable paths. Use only supplied source and logs. Flag unknowns.
Propose an offline replay or preflight check before suggesting a motion change.
```

**Human check:** run the replay in the deployment environment, confirm the loaded model and configuration, and compare a known static measurement with the expected sign and magnitude. Do not label interpolated command samples as measured continuous pose feedback.

### 7. Diagnose a failure without patching every symptom

**First do independently:** write a timestamped symptom and at least two competing explanations.

```text
Build a causal timeline from this run. For each explanation, give supporting
evidence, contradicting evidence, and an unresolved measurement. Recommend
one discriminating test and predict its result under each explanation.
Keep the controller unchanged unless the evidence identifies the relevant
component. After the test, help me update the explanation rather than only
change a threshold.
```

**Human check:** make the prediction before the test. For the recorded M12 interpreter failure, for example, the first useful question is whether the intended model ran at all; changing its architecture would not address the immediate cause.

### 8. Write and defend the result

**First do independently:** write a short claim from a figure, one limitation, and one alternative interpretation.

```text
Review this claim against the linked artifacts. Identify unsupported causal,
novelty, generalization, or success-rate language. Check the denominator,
budget, endpoint, and whether medians condition on different successful sets.
Suggest a clearer argument that retains the negative findings. Do not add
results, citations, completed learning activities, or authorship claims.
```

**Human check:** explain the claim orally from the raw evidence, including what would change the conclusion. AI editing should make an argument easier to inspect, not stronger than its evidence.

## Proposed read-only hardware tools

This section is an integration design for future work. The repository does not contain a verified live ChatGPT/Claude bridge implementing these tools. The deployed Python model runtime is a separate, non-LLM inference process and does not own the robot connection.

| Proposed tool | Input | Output | Useful question |
|---|---|---|---|
| `read_interface_contract` | Component and revision | Signals, types, units, frames, timing, ownership, and error states | What does this field actually mean? |
| `get_cached_status` | Approved status snapshot ID | Sample timestamp, sample age, source, and status values | Is this status fresh enough to interpret? |
| `read_run_trace` | Run ID and time interval | Selected signal rows and state events with provenance | Which event preceded the failure? |
| `replay_saved_cone` | Recorded cone and frozen model/configuration IDs | Input validity, posterior, model identity, and errors | Can the logged inference be reproduced offline? |

A response envelope could look like this; the values are illustrative, not a real run:

```json
{
  "source": "saved_run/status_snapshot.json",
  "source_revision": "<commit-or-sha256>",
  "sample_time_utc": "<timestamp>",
  "sample_age_ms": 25,
  "signal": "wrench",
  "frame": "world",
  "moment_reference": "<explicit-reference-point>",
  "units": ["N", "N", "N", "N*m", "N*m", "N*m"],
  "valid": true,
  "values": [0.0, 0.0, 3.0, 0.0, 0.0, 0.0]
}
```

The adapter would expose existing evidence through a narrow contract. Motion ownership would remain with the reviewed controller and operator workflow. An explanation based on stale cached status should say that it is stale. A replay failure should return a failure, rather than silently substituting a classical cue and calling it the requested model.

Product-specific setup would need to be checked against the actual client, tool mechanism, and laboratory environment at implementation time. This design describes the information boundary, not a tested installation recipe.

## Derivations and exercises that build transferable understanding

### Exercise A: coordinate frames and wrench reference points

Write down both frame axes and the point about which the moment is measured. In one common coordinate system, if $r_{B\rightarrow A}$ is the vector from new reference point $B$ to old reference point $A$, then

$$
\tau_B = \tau_A + r_{B\rightarrow A}\times F.
$$

Rotating the coordinate system also rotates force, torque, and the reference-point vector consistently. Specify the direction of the rotation matrix before using it.

1. Choose a force and a known lever arm; compute the moment by hand.
2. Implement the same transformation and verify the inverse.
3. Change the reference point or sign deliberately and predict the error.
4. Relate that error to the harmonics in a conical probe fingerprint.

**Understanding check:** explain why a moment-origin error can resemble a task-relevant directional signal. This links rigid-body mechanics to inference distribution shift.

### Exercise B: direction, distance, and posterior calibration

Create two possible centres on the same ray but at different distances. Supply an observation that identifies only the ray. Construct a small weighted particle set and show that the radial ambiguity remains. Then add an explicitly stated radial observation model and inspect what changes.

Repeat with a bimodal angular estimate and with a prior that excludes the true centre. Compare nominal probability mass with empirical coverage over repeated synthetic trials. A high direction score is not sufficient evidence of a correct centre distribution.

**Understanding check:** explain which information comes from the observation and which comes from the prior. State how a wrong prior can defeat a confident model.

### Exercise C: action selection and coverage limits

Implement the shortcut rule on synthetic particles. Sweep quality, reverse mass, and controller limits, recording which term sets the action. Draw the residual region and a finite-pitch search path near its boundary.

**Understanding check:** produce one case where a zero shortcut is sensible and one case where a finite search path leaves part of a boundary uncovered. Distinguish an effective heuristic from a geometric guarantee.

### Exercise D: compliant control and state verification

Start with a one-dimensional mass-damping-stiffness or admittance example. Define sign conventions, units, sampling period, saturation, and contact stiffness. Predict the effect of changing damping, delay, or a force bias, then compare with a small simulation.

Next, replay a recorded engagement state sequence. Separate contact detection, capture, insertion, and dwell verification. Identify where each transition gets its evidence.

**Understanding check:** explain why a force-stable interval alone may not establish insertion, and why a learning confidence score should not terminate the physical task. A toy-model result does not certify the full robot's stability.

### Exercise E: fair evaluation and causal claims

Construct a small example in which a method has a low median time on its successes because it fails the hard cases. Compare success-conditioned medians with paired differences and budgeted completion profiles. Define how timeouts and retries enter the outcome.

**Understanding check:** explain why the current simulation table cannot establish unconditional time superiority from the medians alone. Propose an ablation that isolates the shortcut while keeping sensing, controller, starts, and budget fixed.

## A contribution and decision ledger

Use a short entry when a decision changes the research direction or interpretation. Record enough to make responsibility inspectable without turning every edit into bureaucracy.

```text
Date and question:
My initial reasoning and prediction:
Evidence already available:
AI suggestion or critique:
What I accepted, changed, or rejected, and why:
My independent derivation, measurement, or implementation:
Discriminating experiment and frozen conditions:
Observed result, including failures:
Resulting claim and remaining uncertainty:
Artifact paths, configuration, geometry version, and source revision:
What I can now explain independently:
```

For an older decision whose authorship cannot be reconstructed, say so. The purpose is to document reasoning and accountability, not to invent a retrospective division of labor or assign a numerical originality score.

## Make learning visible over time

Use a separate page for each core concept. Include an intuitive explanation, a derivation, a small program, a counterexample, a link to a physical phenomenon, and the project decision it informs. Revisit the page after a delay and answer a changed problem without the earlier solution.

| Level | What I can do | Evidence to retain |
|---|---|---|
| Explain | State the concept, assumptions, and units in my own words | A brief closed-book explanation |
| Derive | Reconstruct the key relation and its conditions | A worked derivation |
| Implement | Build a minimal version and inspect its behavior | A small script and predicted outputs |
| Diagnose | Find why it fails under a changed condition | A counterexample and discriminating test |
| Transfer | Apply it to a new geometry or sensing condition | A new prediction and measured result |

These are proposed checks, not claims that the author has completed them. The aim is to make AI assistance produce deeper judgment: I form an explanation, ask for criticism, test it, and then reconstruct the explanation independently.
