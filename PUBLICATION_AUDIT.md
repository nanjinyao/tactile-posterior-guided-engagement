# Publication audit — 2026-09-17

**Overall verdict: repository preparation completed; public release and full
reproducibility remain blocked by the explicit items in section C.**
No experiment, model weight, calibration value, raw record, manuscript claim,
or shipped figure/video was changed. No files were deleted, staged, committed,
uploaded, or pushed. No robot connection or expensive experiment was launched.

## A. READY FOR GITHUB

The following are organized and documented, subject to the release decisions
in C; this is not blanket clearance to upload the entire working directory.

- Root README with actual manuscript title, installation/dependency guidance,
  directory roles, simulation and hardware workflows, limitations, citation
  placeholders and explicit unselected license status.
- Reproducibility map for all 10 figures and 4 tables, including exact named
  hardware campaigns, trial directories, model bundles and plotting functions.
- Simulation source, JSON geometry/configs, MuJoCo assets and stored evidence
  are present. All ten manuscript figure inputs exist.
- The five-method report, summary, task contract, source manifest and two model
  contract hashes checked by `tools/verify_paper_evidence.py` match. Individual
  trials reproduce the stored 20/19/19/17/16 terminal counts. The 19 common
  PGS-O/Outward completions give the reported -41.14 mm median paired path gap.
- Existing cross-size and M8 artifact tests passed. Selected controller tests
  passed. This validates retained artifacts and limited logic, not fresh physics.
- A local top-level Git repository is initialized, with no commits/remotes.
  Local caches/environments/editor/build metadata are ignored. Scientific
  outputs, logs, calibration, CAD, figures and videos remain available.

### Audit coverage and evidence

The initial workspace contained **84,186 files, 10,391,096,073 bytes** (about
10.4 GB). The saved inventory includes the newly added audit script (84,187
files at scan time). It traversed every directory, including copied
environments and the nested Git repository. This inventory is separate from
the deeper semantic review of active implementation and paper dependencies.

Outside environments/Git, the UTF-8 scan covered **8,308 text files** up to
20 MB each: 293 Python, 4,238 JSON, 3,200 CSV, 80 MATLAB, 253 Markdown, plus
shell/config/TeX/C/C++/other source. Python AST and JSON parsing found no errors.
One existing Python invalid-escape SyntaxWarning appeared in a legacy plotter.
The duplicate scan found **203 SHA-256-equal groups** (files >=1 KB in matching
size groups). Duplicates were retained because some are deliberate bilingual
replays, source snapshots or evidence copies.

Private local reports are in ignored `.publication-audit/`: full inventory,
findings (path/line/category only), source hashes, duplicate groups, parse
summary, test logs, extracted PDF text and a first-page render. The audit does
not claim line-by-line scientific review of all archived reference code,
playback of every video, or privacy clearance of binary content, embedded slide
notes, Git history, or text files larger than the scan limit.

### Validation actually performed

| Check | Outcome |
|---|---|
| Standard-library stored-evidence verifier | PASS: 10 figure files; report/summary/task/model/source-manifest hashes; M8 counts; common-success path statistic; recovery counts; three campaign records located |
| M8/cross-size artifact tests plus hardware static suite (three runtime-heavy test modules excluded) | **175 passed, 4 failed, 29 subtests passed** |
| Posterior search, Cartesian admittance, terminal verifier | **18 passed** |
| Added checkpoint relocation tests | **2 passed**; recorded/native and copied POSIX/Windows paths, missing-seed/bundle/traversal rejection |
| Three frozen M8 checkpoints through relocation helper | All resolve to existing local files; inference not validated |
| Python/JSON scan | 293 Python files and 4,238 JSON files parsed, no errors in scan scope |
| Figure CLI | `--help` passed; output-folder generation attempted and failed in fontTools PDF native extension after first PNG; original figures unchanged |
| MuJoCo/PyTorch-dependent tests | Four collection errors in attempted group; Windows DLL initialization/access-violation errors. No simulation or inference success claimed |
| Existing manuscript PDF | 14 pages; text extracted; page 1 rendered and visually inspected; author metadata field empty; all 10 TeX figure paths exist |
| Git ignore/status checks | Local metadata ignored; representative evidence stays visible; no staging/commit |

Audit environment: Windows, CPython 3.11.16, with the inventory/AST scan on
CPython 3.12.14. Exact diagnostic package versions are in
`docs/AUDIT_ENVIRONMENT.txt`. These are observed audit versions, not historical
paper dependencies or a recommended experiment lock.

Not performed: full test suite, fresh training/calibration, full simulation,
hardware inference replay, CAD regeneration, MATLAB execution, fresh TeX build,
full PDF visual QA, numerical equivalence of regenerated figures, networked
hardware checks, or security certification. The existing Poppler commands did
not produce usable output here; PyMuPDF supplied the PDF inspection fallback.

## B. CHANGES YOU MADE

1. Added `README.md`, `REPRODUCIBILITY.md`, this report and
   `docs/AUDIT_ENVIRONMENT.txt`.
2. Added root `.gitignore` for Python environments/caches, IDE/agent/sync files,
   Office lock files, known catkin build/devel directories and LaTeX auxiliaries.
   No blanket exclusion of data, results, logs, archives, figures or video.
3. Removed the manuscript repository's blanket `*.pdf` ignore rule so paper
   and vector figure PDFs are eligible for inclusion.
4. Added `requirements.txt` from active direct imports without inventing the
   original simulation's version pins. Existing hardware pin files unchanged.
5. Added `tools/audit_publication.py`, a read-only inventory/text/duplicate
   checker whose reports stay in the ignored local audit directory.
6. Added `tools/verify_paper_evidence.py`, a dependency-free, read-only check of
   key frozen evidence and selected manuscript numbers.
7. Added `src/checkpoint_paths.py` and changed only checkpoint path resolution
   in `mujoco_unscrewing_probe_new/src/m8_direction_runtime.py`: when an archived
   path is unavailable, resolve the same model bundle/seed/filename locally.
   Added two regression tests in `tests/test_checkpoint_relocation.py`.
   Existing checkpoint contracts and weights remain unchanged.
8. Made `overleaf_unscrewing/tools/plot_publication_figures.py` use `FFMPEG` or
   PATH instead of one local executable path, and added `--output-dir` while
   preserving source asset paths, plotting calculations and default destination.
9. Added navigation/status notes to the existing project index and manuscript
   README, preserving historical evidence statements.
10. Initialized the previously empty top-level `.git`. Preserved the independent
    `overleaf_unscrewing/.git` history. Created an ignored audit-only environment
    and local validation outputs. No global Git configuration changed.

The nested manuscript repository also reports eight pre-existing modified PNGs
relative to its HEAD. Their byte sizes already matched the current copies in
the initial inventory; they were not restored or overwritten. The final
inventory comparison found no size changes in original simulation outputs,
hardware data or manuscript figure assets. The source/config hash comparison
found only the two intended Python-source modifications listed above.

## C. ISSUES / ITEMS REQUIRING MY CONFIRMATION

### Scientific provenance and manuscript consistency

1. **Exact hardware source is not frozen with its named result.** The current
   `real_device/execute_m12_contact_engagement_policy.m` hash differs from
   `source_sha256` in the Fig. 8 run's `summary.json`. No matching source snapshot
   was found among the scanned source hashes. Supply or identify that exact
   revision before claiming the current file produced those measurements.
   Retained runs and backups were not replaced or relabeled.
2. **Simulation/hardware model description differs.** The simulation interface
   is four-channel/360-phase CNN inference; the hardware bundle README,
   `offline_learning/common.py` and runtime contract describe seven channels,
   a selected `REAL_FOURIER_LINEAR_V1` model and TOP1 routing. The manuscript's
   general model section emphasizes the CNN/four-channel interface. Clarify
   which method details apply to hardware; do not change the implementation to
   make it match the prose.
3. **Hardware endpoint semantics need explicit explanation.** The named M12
   run records `success=false`, `task_completed=true` and a provisional depth
   event. Existing plots use timestamped events. Explain this distinction in
   release documentation/manuscript rather than changing raw flags. Named runs,
   extended budgets, 78 acquisition passes and population success are different
   claims. The conclusion's broad transfer/recovery language deserves author
   review against the separate simulation recovery evidence.
4. **Original software and source-version provenance incomplete.** Simulation
   manifests preserve task/model hashes but do not supply a complete historical
   package/driver lock. New Windows audit packages cannot establish equivalent
   physics. Historical hardware pinned environments are Linux/Python-3.8-era
    environments, not the new diagnostic environment.
    The portability patch also intentionally changes the current source hash;
    it does not rewrite frozen execution bundles or certify a fresh run against
    their old source hashes. New development runs must freeze their own sources.
5. **Full regeneration of composite figures unproven.** Figs. 2–3 use supplied
   or hybrid raster elements. Fig. 3's real keyframe extraction is not fully
   specified by its builder. Preserve the original inputs and supply editable
   provenance/selection metadata if a from-source reproduction is required.
6. **Tables and statistical paragraphs are manually maintained in LaTeX.**
   Relevant trial/report data exist, but there is no located single exporter
   that recreates all four tables and every stated significance test. The new
   verifier checks selected claims, not every numerical value or proof.
7. **Historical documentation drift.** The manuscript README previously used
   an older title and seven-page claim; the included current PDF has 14 pages.
   The root index and older meeting materials describe earlier evidence states.
   Canonical release scope should follow `main.tex` and the new map.

### Execution blockers and failing checks

8. **Missing maintained KST dependency.** Controller/tests compute a path
   outside this repository to `reference/KST-Kuka-Sunrise-Toolbox-master/Matlab_client`.
   `net_establishConnection.m` is absent at the expected location. Confirm the
   exact maintained revision, local modifications, redistribution permission
   and installation procedure; an arbitrary upstream copy may not match the
   protocol/hash checks. ROS/Bota/Sunrise setup is also installation-specific.
9. **Three sampling generator checks fail.** M8, M10 and M12 `--check` tests
   report stale JSON/CSV/PNG/SVG/operator images. Path, newline or renderer
   differences may contribute; geometric inconsistency has not been proven.
   No generator was allowed to overwrite scientific/configuration assets.
   Reconcile on the original environment before accepting regeneration.
10. **Native library execution blocked here.** PyTorch/MuJoCo fail Windows DLL
    initialization; fontTools' PDF extension reports a native-load/application
    control error. Full simulation, inference and complete figure generation
    therefore remain unverified. S4 additionally imports Linux-only `fcntl`.
11. **Hardware commissioning remains incomplete by design.** Several
    fixture/TCP/payload/wrench-transform fields remain pending; operator-taught
    alignment is not independent metrology. Preserve configs and gates. The
    M10 fixed-demo authorization does not grant general M10 deployment.

### Public release, privacy, rights and repository packaging

12. **License decision required.** No project-wide license found. The folder
    also contains third-party papers, a thesis/defense PDF, style/reference
    projects, CAD/photos/media, class files and notices. Confirm what may be
    redistributed, attribution requirements, and ownership. No license invented.
13. **Personal and internal information remains.** Scan candidates include
    10,667 absolute-path occurrences, 262 private-network occurrences and
    39 email occurrences. Locations include hardware controllers/configs/run
    manifests, traces, reference code/package metadata and class notices.
    The original contracts include local usernames; hardware records include
    installation/sensor identifiers. Some email occurrences are legitimate
    license attribution, not material to strip automatically. Review ignored
    `.publication-audit/findings.json` locally; matched values are not copied
    into this report. No credential-pattern matches were found in the scoped
    scan, which is not a guarantee of secret absence. Binary media, large text
    and nested Git history still require release/privacy review.
14. **Double-anonymous release scope needs confirmation.** The paper source
    explicitly requests double-anonymous handling, but a full workspace release
    includes local paths, meeting notes, photos, provenance and Git author/remote
    metadata that may identify contributors. Decide release timing and scope.
15. **Large-file distribution unresolved.** Two byte-identical presentation
    copies (`group_meeting_trace/实验结果.pptx` and
    `real_device/docs/实验结果.pptx`) are 1,317,351,694 bytes each. Two retained
    diagnostic CSVs under
    `mujoco_unscrewing_probe_new/outputs/current/diagnostics/cross_size_bumpless_transition_targeted_development/traces/m12/`
    are 167,230,295 and 182,426,015 bytes. Decide Git LFS or a stable external
    artifact deposit, with hashes and retrieval instructions, before upload.
    No copies or diagnostic records deleted and no storage policy imposed.
16. **Nested Git repository preserved.** `overleaf_unscrewing/.git` has existing
    history and a remote. A naive top-level `git add .` can create an embedded
    repository/gitlink instead of including manuscript files. Decide whether
    to preserve it as a properly configured submodule or import a reviewed
    snapshot with its history separately archived. Neither action was assumed.
17. **Local Git ownership mismatch.** Top-level status required a per-command
    `-c safe.directory=<this checkout>` override because the folder owner SID
    differs from the running user. No global safe-directory exemption was added.
18. **Preserved historical material is not automatically publication content.**
    `tmp/`, archives, sync-conflict traces, meeting drafts, reference projects
    and duplicated media are retained pending selection. `PROJECT_INDEX.md`
    names an older predecessor archive outside this tree; it is not present
    here and was not needed by the mapped current paper artifacts.

Resolve these items against a concrete release snapshot before committing or
publishing. Scientific fixes, regenerated calibration/data, removal of evidence,
rights decisions and history restructuring require the author's decision.

## D. REPRODUCIBILITY MAP

The complete, exact map is in
[REPRODUCIBILITY.md — Paper-to-artifact map](REPRODUCIBILITY.md#paper-to-artifact-map).
It covers Figures 1–10, Tables I–IV, the paired efficiency and recovery results,
CNN/proxy training, S4b adaptation, hardware acquisitions, real-model training,
named M12 examples, M10 pilots, config provenance and retained videos. Each row
distinguishes existing evidence from executed validation and missing provenance.
