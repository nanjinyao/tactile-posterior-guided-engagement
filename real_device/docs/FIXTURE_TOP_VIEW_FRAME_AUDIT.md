# Fixture top-view frame audit

Date: 2026-07-24

## Conclusion

The former sampling-plan figure drew `+Y_F` upward.  That arrow was reversed
relative to the declared right-handed fixture frame.

The fixture frame is defined by

- `+X_F`: from the M10 centre toward the M12 centre, which is leftward in the
  reviewed photograph;
- `+Z_F`: the fitted upward normal of the fixture top datum, from the table
  toward an overhead viewer;
- `+Y_F = +Z_F x +X_F`.

For a displayed top view with horizontal screen-right basis `e_u`, vertical
screen-up basis `e_v`, and outward basis `e_o` toward the viewer,
`e_u x e_v = e_o`.  Because `X_F = -e_u` and `Z_F = e_o`,

```text
Y_F = Z_F x X_F = e_o x (-e_u) = -e_v.
```

Therefore `+Y_F` points **downward** in the displayed top view.  The
unambiguous mappings are

```text
mathematical display axes:  u_right = -x_F,  v_up   = -y_F
image pixel axes:           u_right = -x_F,  v_down = +y_F
```

The previous label `v=y_F` was valid only for a computer-vision pixel axis
whose positive vertical direction is downward; it was not valid on the
Matplotlib graph, where the vertical coordinate increases upward.

## Scope of the correction

`tools/generate_fixture_sampling_plan.py` now:

- applies `(u_right, v_up)=(-x_F,-y_F)` to displayed points;
- draws `+Y_F` downward and marks `+Z_F` out of the table/toward the viewer;
- states the view direction and both display-axis and pixel-axis conventions
  in the generated JSON coordinate contract;
- validates the display mapping for every generated point.

The correction changes only presentation coordinates and generated
figure/JSON/CSV metadata.  It does **not** change any
`local_offset_fixture_mm`, `fixture_xy_mm`,
`true_direction_to_centre_fixture_unit`, as-built rebasing, or
`WORLD <- fixture` motion transform used by the MATLAB controller.

