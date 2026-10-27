# M8 / M10 / M12 Allen Socket Model Inventory

All dimensions are generated from `src/allen_socket_model.py`.
The Allen key across-flats and screw-head scale are nominal size parameters.
Clearance and micro-chamfer are controlled simulation variables, not manufacturing tolerance claims.

|size|key_or_bit_af_mm|socket_wall_af_mm|clearance_mm|chamfer_w/d_mm|socket_top_af_mm|r_cap_wall/top_mm|socket_depth_mm|head_diameter/height_mm|visual_R_default_mm|
|---|---|---|---|---|---|---|---|---|---|
|M8|6.00|6.20|0.10|0.15/0.15|6.50|0.10/0.25|4.00|13.0/8.0|3.0|
|M10|8.00|8.24|0.12|0.18/0.18|8.60|0.12/0.30|5.00|16.5/10.0|4.0|
|M12|10.00|10.30|0.15|0.22/0.22|10.74|0.15/0.37|6.00|18.5/12.0|5.0|

Claim boundary:

```text
Generated XMLs are fixed Allen socket-head engagement scenes.
They do not include threads, screw extraction, tool changer, visual detector, or full unscrewing dynamics.
M10/M12 are model-ready scale variants; current first evidence should remain M8-first.
```
