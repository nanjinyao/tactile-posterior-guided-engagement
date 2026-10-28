# Experiment media gallery

[Back to the project overview](../README.md#experiments-in-motion) · [Download all six MP4s](https://github.com/nanjinyao/tactile-posterior-guided-engagement/releases/tag/media-gallery-2026-10-28) · [Source and checksum manifest](media/manifest.json)

These are viewing copies of existing project recordings, published on 28 October 2026. They introduce no new simulation or hardware trials. The full MP4s preserve the complete source timeline at the source playback rate. Hardware footage is converted to broadly compatible H.264, with audio and embedded metadata removed. No action sequence is rearranged or cropped.

GIF previews loop, omit audio, and sample fewer frames for faster loading. The banner shows the playback multiplier relative to the source video. In simulation, the source video's own simulation clock remains the reference for simulated elapsed time. Do not time a trial from an accelerated animation.

## 1. Clear tactile cue: four methods from the same simulated start

[![M8 clear-cue comparison, Sobol 000](media/simulation-clear-cue.gif)](https://github.com/nanjinyao/tactile-posterior-guided-engagement/releases/download/media-gallery-2026-10-28/simulation-clear-cue.mp4)

The four panels compare posterior-guided grid search, posterior-guided outward search, outward search without a fingerprint, and a four-channel direct-cue method. In this selected case, a concentrated direction cue supports a longer shortcut. Follow the trajectory, changing contact state, and terminal annotations in each panel.

**Source:** M8 `sobol_000`, [original rendering manifest](../mujoco_unscrewing_probe_new/outputs/current/active_comparison/m8_retry3_selected_media/sobol_000/comparison_clear_retry3_2x2_manifest.json). **Full video:** 35.40 s. **GIF:** 2x source playback. [Full MP4](https://github.com/nanjinyao/tactile-posterior-guided-engagement/releases/download/media-gallery-2026-10-28/simulation-clear-cue.mp4).

## 2. Muted tactile cue: why residual search matters

[![M8 muted-cue comparison, Sobol 016](media/simulation-muted-cue.gif)](https://github.com/nanjinyao/tactile-posterior-guided-engagement/releases/download/media-gallery-2026-10-28/simulation-muted-cue.mp4)

The same four methods receive a more ambiguous tactile case. The weaker posterior supports a shorter shortcut; the direct-cue arm misses its three physical rays in this selected example. The contrast with the clear-cue clip illustrates the role of uncertainty and subsequent search.

**Source:** M8 `sobol_016`, [original rendering manifest](../mujoco_unscrewing_probe_new/outputs/current/active_comparison/m8_retry3_selected_media/sobol_016/comparison_fuzzy_retry3_2x2_manifest.json). **Full video:** 43.73 s. **GIF:** 2x source playback. [Full MP4](https://github.com/nanjinyao/tactile-posterior-guided-engagement/releases/download/media-gallery-2026-10-28/simulation-muted-cue.mp4).

Both simulations are selected mechanism examples. Use the [paired 24-case benchmark](../README.md#4-use-simulation-to-test-mechanisms) to assess comparative performance rather than extrapolating from these two clips.

## 3. Real robot: conical tactile probing

[![Physical conical-probing sequence at the printed M12 fixture](media/hardware-conical-probe.gif)](https://github.com/nanjinyao/tactile-posterior-guided-engagement/releases/download/media-gallery-2026-10-28/hardware-conical-probe.mp4)

Controlled changes in tool orientation create a contact response that can be used for tactile inference. The camera shows the robot's tool, force/torque sensor assembly, and printed multi-size fixture. This clip illustrates the acquisition motion; the reported 78 completed scans refer to the separate acquisition records, rather than 78 engagement successes or 78 scans visible in this video.

**Source:** local experimental-results presentation, slide 1, embedded `ppt/media/media1.mp4`. **Full video:** 172.90 s. **GIF:** 8x source playback. [Full MP4](https://github.com/nanjinyao/tactile-posterior-guided-engagement/releases/download/media-gallery-2026-10-28/hardware-conical-probe.mp4).

## 4. Real robot: M12 shortcut and spiral search

[![Physical M12 shortcut and spiral-search trial](media/hardware-m12-shortcut-spiral.gif)](https://github.com/nanjinyao/tactile-posterior-guided-engagement/releases/download/media-gallery-2026-10-28/hardware-m12-shortcut-spiral.mp4)

The sequence connects physical probing to subsequent fine motion at the M12 socket. The source slide is titled “M12 Shortcut Spiral (Successful)”. The full clip preserves setup, motion, and the later portion of the recording, so the reader can inspect the sequence beyond the animated overview.

**Source:** experimental-results presentation, slide 3, embedded `ppt/media/media3.mp4`. **Full video:** 198.52 s. **GIF:** 8x source playback. [Full MP4](https://github.com/nanjinyao/tactile-posterior-guided-engagement/releases/download/media-gallery-2026-10-28/hardware-m12-shortcut-spiral.mp4).

## 5. Real robot: M10 grid search

[![Physical M10 grid-search trial at the middle socket](media/hardware-m10-grid.gif)](https://github.com/nanjinyao/tactile-posterior-guided-engagement/releases/download/media-gallery-2026-10-28/hardware-m10-grid.mp4)

The middle socket of the fixture is used for this M10 recording. The source slide is titled “M10 Entropy Grid (Successful)”. This provides a second physical size and search example alongside the M12 footage.

**Source:** experimental-results presentation, slide 7, embedded `ppt/media/media9.mp4`. **Full video:** 155.12 s. **GIF:** 8x source playback. [Full MP4](https://github.com/nanjinyao/tactile-posterior-guided-engagement/releases/download/media-gallery-2026-10-28/hardware-m10-grid.mp4).

## 6. Additional full-length recording: M12 grid search

[Download the M12 grid-search MP4](https://github.com/nanjinyao/tactile-posterior-guided-engagement/releases/download/media-gallery-2026-10-28/hardware-m12-grid.mp4) (136.00 s). The source slide is titled “M12 Entropy Grid (Successful)”: experimental-results presentation, slide 2, embedded `ppt/media/media2.mp4`.

## Reading the hardware evidence

Method names and success descriptions above follow the original experimental slide captions. The video images do not independently measure force thresholds, insertion depth, dwell criteria, or controller state; no new frame-to-CSV synchronization is asserted. Consult [hardware documentation and records](../real_device), the [research evidence guide](../REPRODUCIBILITY.md), and the [README hardware discussion](../README.md#6-connect-models-software-and-hardware) for those details.

These selected feasibility recordings show socket-engagement investigations. They do not establish a randomized success rate, an aggregate ranking of hardware methods, or completion of thread release and screw extraction.

## Provenance and downloads

The hardware source is `real_device/docs/实验结果.pptx`. Only the selected embedded videos are published here; the large source slide deck is not duplicated. The [media manifest](media/manifest.json) records each source member, its SHA-256, duration and resolution, the full viewing-copy checksum, and each GIF's speed and checksum. Simulation source videos also remain available in the frozen scientific release.

Download the viewing copies with GitHub CLI:

```bash
gh release download media-gallery-2026-10-28 \
  --repo nanjinyao/tactile-posterior-guided-engagement \
  --pattern '*.mp4' --pattern SHA256SUMS.txt --dir media-downloads
```

The [media release](https://github.com/nanjinyao/tactile-posterior-guided-engagement/releases/tag/media-gallery-2026-10-28) is separate from `research-snapshot-2026-10-27`. The original scientific snapshot, data, and six archives remain unchanged.
