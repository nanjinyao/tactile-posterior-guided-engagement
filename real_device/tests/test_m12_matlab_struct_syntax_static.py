"""Guard the MATLAB construct that the string-matching contract tests cannot see.

MATLAB rejects a comment line inside a parenthesised continuation.  Writing

    cfg.surface = struct( ...
        'a', 1, ...
        % explanation
        'b', 2);

fails to parse with "Unexpected end of line", and because every other test in
this suite matches literal substrings rather than parsing, nothing caught it.
On 2026-07-29 that cost a full real-robot setup cycle: the trial launched,
homed and only then died inside makeConfig.  Explanatory notes belong above the
assignment, never between its arguments.
"""

import re
import unittest
from pathlib import Path

REAL_DIR = Path(__file__).resolve().parent.parent
MATLAB_SOURCES = [
    "execute_m12_contact_engagement_policy.m",
    "m12_learned_postcone_dashboard.m",
    "m12_sampling_topview_dashboard.m",
    "run_m12_paired_full_loop_trial.m",
    "run_m10_proposed_plus_entropy_trial.m",
]
STRUCT_OPEN = re.compile(r"=\s*struct\(\s*\.\.\.\s*$")


class TestMatlabStructSyntaxStatic(unittest.TestCase):
    def test_no_comment_lines_inside_struct_continuations(self):
        for name in MATLAB_SOURCES:
            path = REAL_DIR / name
            if not path.is_file():
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
            for index, line in enumerate(lines):
                if not STRUCT_OPEN.search(line):
                    continue
                offenders = []
                for cursor in range(index + 1, len(lines)):
                    body = lines[cursor]
                    if body.strip().startswith("%"):
                        offenders.append((cursor + 1, body.strip()[:60]))
                    if body.rstrip().endswith(");"):
                        break
                self.assertEqual(
                    offenders,
                    [],
                    f"{name}: struct( starting at line {index + 1} contains "
                    f"comment lines, which MATLAB cannot parse: {offenders}.  "
                    f"Move the note above the assignment.",
                )

    def test_guard_detects_a_known_bad_pattern(self):
        # The check is only worth having if it actually fires, so prove it does.
        bad = [
            "cfg.surface = struct( ...",
            "    'a', 1, ...",
            "    % this comment is illegal here",
            "    'b', 2);",
        ]
        offenders = []
        for cursor in range(1, len(bad)):
            if bad[cursor].strip().startswith("%"):
                offenders.append(cursor + 1)
            if bad[cursor].rstrip().endswith(");"):
                break
        self.assertEqual(offenders, [3])


if __name__ == "__main__":
    unittest.main()
