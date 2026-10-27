#!/usr/bin/env python3
"""Generate the M10 26-point acquisition plan from the printed CAD."""

from generate_m12_printed_cad_sampling_plan import configure_target, main


if __name__ == "__main__":
    configure_target("m10")
    raise SystemExit(main())
