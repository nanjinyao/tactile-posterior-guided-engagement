from __future__ import annotations

import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "run_m12_paired_full_loop_trial.m"
CONTROLLER = ROOT / "execute_m12_contact_engagement_policy.m"

EXISTING_M12_PUBLIC_ENTRIES = {
    "run_m12_full_loop_baseline_improved_direct_x3.m",
    "run_m12_full_loop_baseline_pure_outward_spiral.m",
    "run_m12_full_loop_proposed_local_spiral.m",
    "run_m12_full_loop_proposed_plus_entropy_grid.m",
    "run_m12_sampling_conical_fingerprints_26pt.m",
    "run_m12_sampling_conical_fingerprints_26pt_resume.m",
}
PAIRED_ENTRY = "run_m12_paired_full_loop_trial.m"
# A second launcher is sanctioned only because it answers a different
# question: the paired entry runs the unbiased four-method campaign, while
# this one runs a two-method case study at a deliberately discriminating
# offset.  Register any further launcher here on purpose -- the assertion
# below exists to stop public entries accumulating by accident.
PROPOSED_VS_DIRECT_ENTRY = "run_m12_proposed_vs_direct_trial.m"
EXPECTED_METHOD_MODES = (
    "FULL_LOOP_PROPOSED_PLUS_LEARNED_ENTROPY",
    "FULL_LOOP_PROPOSED_LEARNED_LOCAL_SPIRAL",
    "FULL_LOOP_BASELINE_IMPROVED_DIRECT_X3_LEARNED",
    "FULL_LOOP_BASELINE_PURE_OUTWARD_SPIRAL",
)


class TestM12PairedFullLoopTrialStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.entry = (
            ENTRY.read_text(encoding="utf-8")
            if ENTRY.is_file()
            else ""
        )

    def setUp(self) -> None:
        if (
            self._testMethodName
            != "test_one_canonical_public_entry_and_no_extra_public_scripts"
            and not ENTRY.is_file()
        ):
            self.skipTest(
                "paired full-loop entry has not been implemented yet"
            )

    def _constant_matrix(
        self,
        source: str,
        variable_names: tuple[str, ...],
    ) -> list[list[float]]:
        variables = "|".join(re.escape(name) for name in variable_names)
        match = re.search(
            rf"\b(?:{variables})\s*=\s*\[(?P<body>.*?)\]\s*;",
            source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(
            match,
            "missing explicit constant matrix "
            + "/".join(variable_names),
        )
        assert match is not None
        body = re.sub(r"%[^\n]*", "", match.group("body"))
        body = body.replace("...", " ")
        rows: list[list[float]] = []
        number = re.compile(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        )
        for raw_row in body.split(";"):
            raw_row = raw_row.strip()
            if not raw_row:
                continue
            values = [float(value) for value in number.findall(raw_row)]
            remainder = number.sub("", raw_row)
            remainder = re.sub(r"[\s,]+", "", remainder)
            self.assertEqual(
                remainder,
                "",
                "campaign matrices must be literal and deterministic",
            )
            rows.append(values)
        return rows

    def _matlab_function(self, source: str, name: str) -> str:
        declaration = re.search(
            rf"^function(?:(?:[^\n]*\.\.\.\n\s*)*)"
            rf"[^\n]*\b{re.escape(name)}\b[^\n]*$",
            source,
            flags=re.MULTILINE,
        )
        self.assertIsNotNone(
            declaration,
            f"missing MATLAB function {name}",
        )
        assert declaration is not None
        following = re.search(
            r"^function\b",
            source[declaration.end() :],
            flags=re.MULTILINE,
        )
        end = (
            len(source)
            if following is None
            else declaration.end() + following.start()
        )
        return source[declaration.start() : end]

    def _top_level_function(self) -> str:
        following = re.search(
            r"^function\b",
            self.entry[self.entry.find("\n") + 1 :],
            flags=re.MULTILINE,
        )
        if following is None:
            return self.entry
        first_newline = self.entry.find("\n") + 1
        return self.entry[: first_newline + following.start()]

    def test_one_canonical_public_entry_and_no_extra_public_scripts(
        self,
    ) -> None:
        self.assertTrue(
            ENTRY.is_file(),
            f"missing canonical public entry {ENTRY.name}",
        )
        paired = sorted(
            path.name
            for path in ROOT.glob("run_m12*paired*full*loop*.m")
        )
        self.assertEqual(paired, [PAIRED_ENTRY])
        active = {
            path.name
            for path in ROOT.glob("run_m12*.m")
            if path.is_file()
        }
        self.assertEqual(
            active,
            EXISTING_M12_PUBLIC_ENTRIES
            | {PAIRED_ENTRY, PROPOSED_VS_DIRECT_ENTRY},
            "add public entries deliberately, one registered name at a time",
        )

    def test_discriminating_launcher_declares_it_is_not_a_random_draw(
        self,
    ) -> None:
        # The two-method launcher exists to show where the methods differ, at
        # an offset picked for that purpose.  If that ever stops being stated
        # in the file and recorded in its checkpoint, the run becomes easy to
        # present as an unbiased sample, which it is not.
        entry = ROOT / PROPOSED_VS_DIRECT_ENTRY
        self.assertTrue(entry.is_file())
        source = entry.read_text(encoding="utf-8", errors="replace")
        for token in (
            "chosen deliberately to discriminate",
            "It is NOT a random draw and must never be presented as one",
            "ONE_FIXED_DELIBERATELY_DISCRIMINATING_POINT_NOT_A_RANDOM_DRAW",
            "DELIBERATELY_DISCRIMINATING_NOT_RANDOM_",
            "campaign.offset_selection_rationale",
            "privateTrialSpec.offset_selection_rationale",
        ):
            self.assertIn(token, source)
        # It must also stay honest about what the geometry cannot promise, and
        # keep the measured evidence the design rests on visible next to it.
        self.assertIn("NOT GUARANTEED", source)
        self.assertIn("Measured direction error", source)
        for measured in ("1.50 deg", "16.25 deg", "166.25 deg"):
            self.assertIn(measured, source)

    def test_public_entry_is_strictly_no_argument_and_has_no_prompt(
        self,
    ) -> None:
        declaration = next(
            (
                line.strip()
                for line in self.entry.splitlines()
                if line.strip() and not line.lstrip().startswith("%")
            ),
            "",
        )
        self.assertRegex(
            declaration,
            r"^function\s+(?:\w+|\[[^\]]+\])\s*=\s*"
            r"run_m12_paired_full_loop_trial\(\)\s*$",
        )
        self.assertNotIn("varargin", declaration)
        for prompt in (
            "input",
            "inputdlg",
            "questdlg",
            "menu",
            "uigetfile",
            "uigetdir",
            "listdlg",
            "keyboard",
            "pause",
            "waitforbuttonpress",
        ):
            self.assertIsNone(
                re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(prompt)}\s*\(",
                    self.entry,
                    flags=re.IGNORECASE,
                ),
                f"paired public entry must not call {prompt}",
            )

    def test_demo_has_one_fixed_reachable_offset_and_four_methods(
        self,
    ) -> None:
        self.assertIn(
            "'M12-CLEAN_CANDIDATE-SUPPORTED-A003p75'",
            self.entry,
        )
        offsets = self._constant_matrix(
            self.entry,
            ("offsetsFixtureMm", "offsets_fixture_mm"),
        )
        self.assertEqual(len(offsets), 1)
        self.assertEqual(len(offsets[0]), 2)
        self.assertTrue(
            all(math.isfinite(value) for row in offsets for value in row)
        )
        self.assertAlmostEqual(offsets[0][0], 4.580702667, places=9)
        self.assertAlmostEqual(offsets[0][1], 0.420907446, places=9)
        radius = math.hypot(*offsets[0])
        angle = math.degrees(math.atan2(offsets[0][1], offsets[0][0]))
        self.assertAlmostEqual(radius, 4.600, places=8)
        self.assertAlmostEqual(angle, 5.250, places=8)

        source_offsets = self._constant_matrix(
            self.entry,
            (
                "sourceSampleOffsetFixtureMm",
                "source_sample_offset_fixture_mm",
            ),
        )
        self.assertEqual(len(source_offsets), 1)
        self.assertEqual(len(source_offsets[0]), 2)
        self.assertLess(
            math.dist(offsets[0], source_offsets[0]),
            1.10,
            "the demo point must remain near the successful clean trace",
        )
        self.assertLess(
            radius + 0.250 - 4.200,
            0.750,
            "4.2 mm reach plus the CAD capture radius must cover the "
            "demo point after the declared touchdown-XY lock error",
        )

        modes_match = re.search(
            r"\b(?:methodModes|method_modes)\s*=\s*"
            r"\{(?P<body>.*?)\}\s*;",
            self.entry,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(
            modes_match,
            "missing explicit four-method campaign list",
        )
        assert modes_match is not None
        modes = re.findall(
            r"['\"](FULL_LOOP_[A-Z0-9_]+)['\"]",
            modes_match.group("body"),
        )
        self.assertEqual(
            tuple(modes),
            EXPECTED_METHOD_MODES,
            "method selector 1..4 must have one stable, documented mapping",
        )

        for randomizer in (
            "rand",
            "randn",
            "randi",
            "randperm",
            "rng",
        ):
            self.assertIsNone(
                re.search(
                    rf"(?<![A-Za-z0-9_]){randomizer}\s*\(",
                    self.entry,
                    flags=re.IGNORECASE,
                ),
                "offsets and method order must remain deterministic",
            )

    def test_one_top_level_selector_maps_directly_to_one_method(
        self,
    ) -> None:
        top = self._top_level_function()
        selectors = list(
            re.finditer(
                r"(?m)^\s*selectedMethodIndex\s*=\s*([1-4])\s*;"
                r"\s*(?:%[^\n]*)?$",
                top,
            )
        )
        self.assertEqual(
            len(selectors),
            1,
            "edit exactly one literal selectedMethodIndex = 1..4 at "
            "the top of the launcher",
        )
        selector = selectors[0]
        self.assertLess(
            selector.start(),
            top.index("realDir ="),
            "the operator-facing method selector must be visible before "
            "launcher setup",
        )
        self.assertRegex(
            top,
            r"trialIndex\s*=\s*selectedMethodIndex\s*;",
            "the selected method owns its stable campaign record directly",
        )
        self.assertRegex(
            top,
            r"trial\s*=\s*campaign\.trials\s*\(\s*"
            r"trialIndex\s*\)\s*;",
        )
        self.assertNotRegex(
            top,
            r"methodIndex\s*=\s*(?:double\s*\(\s*)?"
            r"methodOrder\s*\(",
            "the selected method must not come from legacy method order",
        )
        self.assertNotRegex(
            top,
            r"trialIndex\s*=\s*find\s*\(\s*~\s*completed",
            "the launcher must not silently choose the next old-order trial",
        )
        self.assertNotIn(
            "M12Demo:PairedCampaignComplete",
            top,
            "a prior COMPLETE record must not prevent rerunning the selected "
            "method",
        )
        validator = self._matlab_function(
            self.entry, "validatePairedCampaign"
        )
        self.assertRegex(
            validator,
            r"double\(trial\.method_index\)\s*==\s*methodIndex",
            "a selected persisted slot is only safe when its method index "
            "is bound to that same stable slot",
        )
        self.assertRegex(
            validator,
            r"strcmp\(char\(trial\.method_mode\),\s*"
            r"methodModes\{methodIndex\}\)",
            "the persisted method string must retain the selector's exact "
            "1..4 mapping",
        )

    def test_single_offset_is_selected_once_outside_method_trials(
        self,
    ) -> None:
        top = self._top_level_function()
        selected = re.search(
            r"offsetFixtureMm\s*=\s*offsetsFixtureMm\s*"
            r"\(\s*offsetIndex\s*,\s*:\s*\)\s*;",
            top,
        )
        self.assertIsNotNone(
            selected,
            "select the one literal fixture offset once",
        )
        assert selected is not None
        self.assertLess(
            selected.start(),
            top.index("execute_m12_contact_engagement_policy("),
            "select the shared fixed offset before direct motion",
        )
        self.assertRegex(
            top,
            r"privateTrialSpec\.offset_fixture_mm\s*=\s*"
            r"offsetFixtureMm\s*;",
        )
        self.assertNotRegex(
            top,
            r"offsetsFixtureMm\s*\(\s*"
            r"(?:selectedMethodIndex|methodIndex|orderPosition)",
            "changing method 1..4 must never change the paired start offset",
        )
        selections = re.findall(
            r"offsetFixtureMm\s*=\s*offsetsFixtureMm\s*"
            r"\(\s*offsetIndex\s*,\s*:\s*\)\s*;",
            top,
        )
        self.assertGreaterEqual(len(selections), 1)
        self.assertLessEqual(
            len(selections),
            2,
            "the fixed offset may be materialized during new-campaign "
            "creation and selected-method execution, but nowhere else",
        )
        self.assertEqual(
            len(re.findall(r"\bfor\s+offsetIndex\b", top)),
            0,
            "the one-point demo must not retain an offset loop",
        )

    def test_controller_receives_an_explicit_private_trial_spec(self) -> None:
        top = self._top_level_function()
        controller_calls = list(
            re.finditer(
                r"execute_m12_contact_engagement_policy\s*\(",
                top,
            )
        )
        self.assertEqual(
            len(controller_calls),
            1,
            "the inner loop must have one canonical shared-controller call",
        )
        for field in (
            "schema",
            "campaign_id",
            "trial_index",
            "offset_index",
            "offset_fixture_mm",
            "method_index",
            "method_mode",
        ):
            self.assertRegex(
                top,
                rf"privateTrialSpec\.{field}\s*=",
            )
        self.assertIn(
            "M12_PAIRED_FULL_LOOP_PRIVATE_TRIAL_V2",
            top,
        )
        self.assertIn("privateTrialSpec.demo_design_id", top)
        self.assertIn("privateTrialSpec.demo_seed_token", top)
        self.assertIn("privateTrialSpec.source_sample_id", top)
        call = re.search(
            r"execute_m12_contact_engagement_policy\s*\(\s*"
            r"\.\.\.\s*\n\s*"
            r"(?:executionMode|methodMode|mode)\s*,\s*"
            r"privateTrialSpec\s*\)",
            top,
        )
        self.assertIsNotNone(
            call,
            "controller must receive mode plus the explicit private trial spec",
        )
        for wrapper in (
            "run_m12_full_loop_proposed_plus_entropy_grid(",
            "run_m12_full_loop_proposed_local_spiral(",
            "run_m12_full_loop_baseline_pure_outward_spiral(",
            "run_m12_full_loop_baseline_improved_direct_x3(",
        ):
            self.assertNotIn(wrapper, top)

    def test_json_campaign_is_checkpointed_before_after_and_on_failure(
        self,
    ) -> None:
        top = self._top_level_function()
        for token in (
            "'paired_full_loop_demo_campaign_v2.json'",
            "campaign.schema",
            "campaign.campaign_id",
            "campaign.status",
            "campaign.offsets_fixture_mm",
            "campaign.method_modes",
            "campaign.demo_design_id",
            "campaign.demo_seed_token",
            "campaign.source_sample_id",
            "campaign.source_sample_offset_fixture_mm",
            "campaign.worst_case_capture_margin_mm",
            "campaign.trials",
        ):
            self.assertIn(token, top)
        self.assertIn(
            "M12_PAIRED_FULL_LOOP_DEMO_CAMPAIGN_V2",
            top,
        )
        self.assertNotIn(
            "'paired_full_loop_campaign.json'",
            top,
            "the V2 launcher must not open or overwrite the legacy campaign",
        )

        controller = top.index(
            "execute_m12_contact_engagement_policy("
        )
        checkpoint_pattern = re.compile(
            r"writePairedCampaignCheckpoint\s*\(\s*"
            r"campaignPath\s*,\s*campaign\s*\)"
        )
        self.assertRegex(top[:controller], checkpoint_pattern)
        self.assertRegex(top[controller:], checkpoint_pattern)
        catch = top.index("catch trialError", controller)
        self.assertRegex(
            top[catch:],
            checkpoint_pattern,
            "a failed physical trial must remain in persistent JSON",
        )
        complete = top.index("campaign.status = 'COMPLETE'", catch)
        self.assertRegex(
            top[complete:],
            checkpoint_pattern,
            "the completed four-method demo must be persisted",
        )

        writer = self._matlab_function(
            self.entry, "writePairedCampaignCheckpoint"
        )
        for token in (
            "jsonencode(campaign",
            "fopen(",
            "fwrite(",
            "fclose(",
            "movefile(",
        ):
            self.assertIn(token, writer)
        self.assertRegex(
            writer,
            r"(?:tempname\s*\(|\.tmp)",
            "campaign JSON must be replaced through a temporary file",
        )

    def test_task_outcome_kind_and_explicit_selector_rerun_contract(
        self,
    ) -> None:
        top = self._top_level_function()
        self.assertRegex(
            top,
            r"'outcome_kind'\s*,\s*'PENDING'",
            "every new trial must start with an explicit pending outcome",
        )
        self.assertRegex(
            top,
            r"campaign\.trials\(trialIndex\)\.outcome_kind\s*=\s*"
            r"'TASK_SUCCESS'",
            "a capture-and-insertion return must persist task success",
        )
        self.assertIn(
            "finiteDirectExhaustion = strcmp(executionMode, ...",
            top,
        )
        self.assertRegex(
            top,
            r"campaign\.trials\(trialIndex\)\.outcome_kind\s*=\s*"
            r"(?:\.\.\.\s*)?"
            r"'METHOD_EXHAUSTED_NO_CAPTURE'",
            "three clean Direct misses are a completed finite baseline outcome",
        )

        catch_start = top.index("catch trialError")
        catch_end = top.index("\nend\n\ncampaign =", catch_start)
        catch = top[catch_start:catch_end]
        self.assertRegex(
            catch,
            r"\.status\s*=\s*'RETRY_PENDING'",
        )
        self.assertRegex(
            catch,
            r"\.outcome_kind\s*=\s*"
            r"(?:\.\.\.\s*)?"
            r"'HARD_STATE_OR_IMPLEMENTATION_RETRY_REQUIRED'",
        )
        self.assertNotRegex(
            catch,
            r"campaign\.trials\(trialIndex\)\.status\s*=\s*'COMPLETE'",
            "no exception may be relabelled as normal finite-method completion",
        )
        self.assertNotIn(
            "isCompletedTaskOutcomeError",
            self.entry,
            "finite Direct exhaustion is a normal return, not an exception "
            "identifier classifier",
        )
        self.assertNotRegex(
            top,
            r"next_trial_index\s*=\s*min\s*\(\s*\.\.\.\s*\n\s*"
            r"numel\(campaign\.trials\)\s*,\s*trialIndex\s*\+\s*1\s*\)",
            "completion must not silently advance away from the explicit "
            "top-level method selector",
        )
        self.assertNotRegex(
            top,
            r"trialIndex\s*=\s*find\s*\(\s*~\s*completed",
            "COMPLETE/RETRY_PENDING are records, not method selectors",
        )
        self.assertLess(
            catch.index("writePairedCampaignCheckpoint("),
            catch.index("rethrow(trialError)"),
            "the classified outcome must be checkpointed before rethrow",
        )

        # Catch handling is uniform: no identifier allowlist may convert an
        # exception into task completion.  Representative hard-state names
        # therefore need no special branch in the catch block.
        for retry_identifier in (
            "M12Demo:HardForceGuard",
            "M12Demo:ConicalHardForceGuard",
            "M12Demo:HardTorqueGuard",
            "M12Demo:ConicalHardTorqueGuard",
            "M12Demo:BotaWrenchTimeout",
            "M12Demo:WrenchNonfinite",
            "M12Demo:DirectServoStateUnknown",
            "M12Demo:KstStateUnknown",
        ):
            self.assertNotIn(retry_identifier, catch)

    def test_controller_rebases_before_official_baseline_and_touchdown(
        self,
    ) -> None:
        controller = CONTROLLER.read_text(encoding="utf-8")
        preposition = controller.index(
            "executePairedAirPreposition("
        )
        official_baseline = controller.index(
            "measureBaseline(ctx, 'START_BASELINE')"
        )
        touchdown = controller.index(
            "guardedTouchdown(ctx,"
        )
        self.assertLess(preposition, official_baseline)
        self.assertLess(official_baseline, touchdown)
        between = controller[preposition:official_baseline]
        self.assertIn("ctx.start_pose = trialStartPose;", controller)
        self.assertIn("ctx.start_joints = trialStartJoints;", controller)
        self.assertIn(
            "pairedTrialSpec = struct('enabled', true",
            between,
            "oracle-bearing offset spec must be erased before task baseline",
        )

    def test_learned_input_has_no_paired_or_offset_oracle(self) -> None:
        controller = CONTROLLER.read_text(encoding="utf-8")
        learned_input = self._matlab_function(
            controller, "buildLearnedModelInput"
        ).lower()
        for forbidden in (
            "pairedtrial",
            "campaign_center",
            "offset_fixture",
            "taughtcenter",
        ):
            self.assertNotIn(forbidden, learned_input)


if __name__ == "__main__":
    unittest.main()
