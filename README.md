<div align="center">

# AVTR-1

[![Project Page](https://img.shields.io/badge/Project%20Page-AVTR--1-8A2BE2)](https://avaturn-live.github.io/avtr-1-projectpage/)
[![Managed API](https://img.shields.io/badge/Managed%20API-avaturn.live-blue?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAQAAADZc7J/AAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAACYktHRAD/h4/MvwAAAAd0SU1FB+oFFA8fAzOfxeUAAAOsSURBVEjHfZVPSFxXFMa/+wYpJGqofxCrhBKcpBMTozGRdFNCSDYtpRFCFu0mWYRgNor54za7JILgspa6SALZl9JCVgndFGqqVWmpmkKjUWGCCiGOzrx376+L++Y5o5Oe1TDvfueec+53vs+oJDAKjJVo1ef6TCf0keokbehfzeg3/WCWJVJyBlUKAkmig+/ZoFJkGaN75+RueEpiP8PkAbBEOJzDgcMSYQHYYoQaf3ovPMMkAKFHRcnVMdoRAjBFelcKUhI9LHmwgwS8RS5JAg7rbAgs0lOSgkCijRUoRDuHs9ylnQCRYSjO7aMQASsciWeBIaCaaY+z5AiBpzQgTnKDPk4hGnkKhORwxRtm2EeA8eWPAAXLAw4hWrlKLZ1MJDOYoYMa+jmISDNKwZczLBEIQyc57KbrRZzhJhcQ4jqL5LE4tnnFFYQ4Rz9diF62HZYCGT+DMSAaRdwD4DHiBEIcIMMxqlH8zyMA7iHu+zbGJUOL+yOot0qbek0IvdVhNep3LehHzWpVTq06ri+U1mm90Z+ql/SpVvWSlOFtcFT0AXYTMUAEDCKeVSTic8QdIOIW4h1Y4Jp4CEQRTZwF5qjiYkydEIvDxb8scAnDHHCeBkLfxBMxCaGF24gXfEOKl7iYt6VhccxTxWUmEQNA6IBZse5ryVJHK2KwhIflEQFDiCaaWcajWNPOx28Rn5DDebJUrGGDDOK7kktKFjOQZJSXVHndjaQPhCRXtodrvphlmmlCDP1vC4OIFurIFltYF7N+HAOISS5Txfx7h/gPKb7mBeI2fvRMiSdAFNLAeWAOwyXA7nnGCLhIFX8DZ2ki8iU9FNcA+w5xiwi4g3hesYVnyRsNIDZ9D32GFveXqbW0mWb9KmlN7WrUhBb0k2b1WoGadVxfKq1uvdG8amV0WmtaICW3FnRKYhyI7vsVAR4ly1TNMTIcSJbpcbJMo76BMf8OGQrYbdeL6KKfcwhxhVds47DkWeQ6QlzgJmcQvWw6LDm6MF7QhoGwwChpxEH6qaGDmaT7CTqp5SqtiEM8wEIBGJFIeUnbx4x/aJdIWiPiFH3c4CSiIZE0W6TENNUEmKKoHmF1R1RDYIkhMoiAdu6SLaFXIQJWaSsxmFjWF8GGtkzWc2yVyXrsDUtlsp6kOMxUXIAjpk4R7GDHWCbJvM+bahiJr/RmVrQ2l1hbnmH2V4CXmGs3Y2QrUnGDcTp2m6upaO8t+ko96tDH+lDSulY0rV/0s3m9197/A6bOU5Zxdgd9AAAAAElFTkSuQmCC)](https://avaturn.live)
[![Weights](https://img.shields.io/badge/HuggingFace-Weights-orange?logo=huggingface)](https://huggingface.co/avaturn-live/avtr-1)
[![Demo](https://img.shields.io/badge/Demo-Try%20Now-brightgreen?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAQAAADZc7J/AAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAACYktHRAD/h4/MvwAAAAd0SU1FB+oFFA8fAzOfxeUAAAOsSURBVEjHfZVPSFxXFMa/+wYpJGqofxCrhBKcpBMTozGRdFNCSDYtpRFCFu0mWYRgNor54za7JILgspa6SALZl9JCVgndFGqqVWmpmkKjUWGCCiGOzrx376+L++Y5o5Oe1TDvfueec+53vs+oJDAKjJVo1ef6TCf0keokbehfzeg3/WCWJVJyBlUKAkmig+/ZoFJkGaN75+RueEpiP8PkAbBEOJzDgcMSYQHYYoQaf3ovPMMkAKFHRcnVMdoRAjBFelcKUhI9LHmwgwS8RS5JAg7rbAgs0lOSgkCijRUoRDuHs9ylnQCRYSjO7aMQASsciWeBIaCaaY+z5AiBpzQgTnKDPk4hGnkKhORwxRtm2EeA8eWPAAXLAw4hWrlKLZ1MJDOYoYMa+jmISDNKwZczLBEIQyc57KbrRZzhJhcQ4jqL5LE4tnnFFYQ4Rz9diF62HZYCGT+DMSAaRdwD4DHiBEIcIMMxqlH8zyMA7iHu+zbGJUOL+yOot0qbek0IvdVhNep3LehHzWpVTq06ri+U1mm90Z+ql/SpVvWSlOFtcFT0AXYTMUAEDCKeVSTic8QdIOIW4h1Y4Jp4CEQRTZwF5qjiYkydEIvDxb8scAnDHHCeBkLfxBMxCaGF24gXfEOKl7iYt6VhccxTxWUmEQNA6IBZse5ryVJHK2KwhIflEQFDiCaaWcajWNPOx28Rn5DDebJUrGGDDOK7kktKFjOQZJSXVHndjaQPhCRXtodrvphlmmlCDP1vC4OIFurIFltYF7N+HAOISS5Txfx7h/gPKb7mBeI2fvRMiSdAFNLAeWAOwyXA7nnGCLhIFX8DZ2ki8iU9FNcA+w5xiwi4g3hesYVnyRsNIDZ9D32GFveXqbW0mWb9KmlN7WrUhBb0k2b1WoGadVxfKq1uvdG8amV0WmtaICW3FnRKYhyI7vsVAR4ly1TNMTIcSJbpcbJMo76BMf8OGQrYbdeL6KKfcwhxhVds47DkWeQ6QlzgJmcQvWw6LDm6MF7QhoGwwChpxEH6qaGDmaT7CTqp5SqtiEM8wEIBGJFIeUnbx4x/aJdIWiPiFH3c4CSiIZE0W6TENNUEmKKoHmF1R1RDYIkhMoiAdu6SLaFXIQJWaSsxmFjWF8GGtkzWc2yVyXrsDUtlsp6kOMxUXIAjpk4R7GDHWCbJvM+bahiJr/RmVrQ2l1hbnmH2V4CXmGs3Y2QrUnGDcTp2m6upaO8t+ko96tDH+lDSulY0rV/0s3m9197/A6bOU5Zxdgd9AAAAAElFTkSuQmCC)](https://avaturn.live/demo)

</div>

**AVTR-1** is a flow-matching-based autoregressive model for live dialogue. Given a portrait image and dual-stream audio, it renders lip-synced speech and active listening at 25 fps on a single GPU. Built for production deployment: model weights, TensorRT-accelerated inference, and the live-session backend - available as an API or fully self-hosted

<div align="center">
  <video src="https://github.com/user-attachments/assets/5e4f96af-973a-4aa4-a8be-e0ce6f44a5d1" muted=false width="50%"></video>
</div>

---

## 📑 What's included

- [x] Model weights
- [x] Inference code
- [x] Interactive streaming demo
- [ ] Technical report (Coming soon)
- [ ] Production-ready back-end (Coming soon)

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Performance](#2-performance)
3. [Troubleshooting](#troubleshooting)

---

## 1. 🚀 Quick Start

### Prerequisites

- Linux
- NVIDIA GPU (Ampere or later recommended)
- CUDA 12.x + TensorRT 10.x
- [pixi](https://prefix.dev/) — `curl -fsSL https://pixi.sh/install.sh | sh`

### Install

```bash
git clone https://github.com/avaturn-live/avtr-1.git
cd avtr-1
pixi install
```

### Set storage path (Optional)

```bash
export AVTR1_LOCAL_STORAGE=/path/to/avtr1_storage
```

All downloaded weights and built engines go here. Defaults to `<project_root>/artifacts/` (the repo checkout, not the caller's working directory) when unset.

### Download weights

```bash
pixi run download
```

First run will prompt for a HuggingFace login via `hf auth login`
(automatically invoked as a dependency of `download`).

### Build TRT engines

Weights are pulled from two public HF repos by the previous step:
AVTR-1 weights from [avaturn-live/avtr-1](https://huggingface.co/avaturn-live/avtr-1)
and LivePortrait weights repackaged as ONNX graphs from
[digital-avatar/ditto-talkinghead](https://huggingface.co/digital-avatar/ditto-talkinghead).
TRT engines are compute-capability specific and built locally — run the scripts
below once per machine; outputs land under `$AVTR1_LOCAL_STORAGE`.

```bash
# Build everything at once
pixi run build-trt-engines

# Or individually
pixi run build-trt-engines-avtr1
pixi run build-trt-engines-renderer
pixi run build-trt-engines-hubert
```

### Run interactive demo
```bash
pixi run interactive-demo
```


### Run offline generation

**Single speaker.** Avatar lip-syncs the given audio track.

```bash
pixi run generate_offline --speech example/speaker_1.ogg

# with a custom avatar and background:
pixi run generate_offline --speech example/speaker_1.ogg --avatar maria --bg minimal_office
```

**Two-speaker dialogue.** Avatar voices `--speech` and reacts (active listening) to the peer audio on `--listen`. Run twice with the tracks swapped to render both sides of the conversation.

```bash
# avatar = speaker 1 (elena)
pixi run generate_offline --speech example/speaker_1.ogg --listen example/speaker_2.ogg --avatar elena  --out elena.mp4
# avatar = speaker 2 (marcus)
pixi run generate_offline --speech example/speaker_2.ogg --listen example/speaker_1.ogg --avatar marcus --out marcus.mp4

# stitch both sides into a single side-by-side video:
ffmpeg -i elena.mp4 -i marcus.mp4 -filter_complex \
  "[0:v][1:v]hstack=inputs=2[v];[0:a][1:a]amix=inputs=2[a]" \
  -map "[v]" -map "[a]" dialogue.mp4
```

**Silence / idle motion.** No audio — renders idle micro-motion for the given duration.

```bash
pixi run generate_offline --duration 10
```

### Motion controls and diagnostics

Offline inference exposes the existing runtime guidance, noise, integration,
and source-crop controls directly:

```bash
pixi run generate_offline \
  --speech example/speaker_1.ogg \
  --avatar maria \
  --bg plain_white \
  --cfg-self-audio 2.0 \
  --cfg-other-audio 2.0 \
  --cfg-kp 3.0 \
  --noise-alpha 2.0 \
  --noise-trunc-z 1.2 \
  --ode-steps 5 \
  --seed 1234
```

Use opt-in diagnostics to investigate head jitter, facial geometry changes,
stitch corrections, and matte occupancy:

```bash
pixi run generate_offline \
  --speech example/speaker_1.ogg \
  --avatar maria \
  --bg plain_white \
  --seed 1234 \
  --motion-debug-jsonl motion-debug.jsonl
```

Each JSONL event includes a per-chunk `trace_id`. The main stages are
`avatar_registration`, `audio_chunk`, `motion_prediction`,
`motion_stabilization`, `turn_guidance`, `motion_trajectory`, `keypoint_geometry`,
`stage_artifact`, and `render_alpha`. Diagnostic
metrics synchronize CUDA and reduce throughput, so do not use them for
performance measurements.

Turn-aware other-audio guidance is also opt-in. In automatic mode it classifies
the current speech/listen audio with separate on/off RMS thresholds, requires a
stable observation before changing state, and interpolates the CFG value across
chunks. Overlap and silence hold the previous speaking/listening target:

```bash
pixi run generate_offline \
  --speech example/speaker_1.ogg \
  --listen example/listener_1.ogg \
  --turn-guidance auto \
  --speaking-cfg-other-audio 1.0 \
  --listening-cfg-other-audio 2.0 \
  --turn-hysteresis-chunks 2 \
  --turn-transition-chunks 2
```

Use `--turn-guidance speaking` or `listening` when a trusted upstream turn
controller supplies the state. The controller state is included in the normal
safetensors session blob. `disabled` remains the default and uses
`--cfg-other-audio` exactly as before.

### Experimental motion stabilization

Temporal guards are disabled by default and do not alter existing inference.
The rotation guard suppresses isolated one-frame reversals in the predicted
head rotation while leaving sustained motion untouched:

```bash
pixi run generate_offline \
  --speech example/speaker_1.ogg \
  --avatar maria \
  --bg plain_white \
  --motion-stabilization rotation \
  --rotation-acceleration-threshold-deg 0.75 \
  --rotation-max-correction-deg 1.5 \
  --motion-debug-jsonl rotation-debug.jsonl \
  --out rotation-guard.mp4
```

Expression coordinates are learned and do not have safe anatomical labels.
Generate sensitivity artifacts before enabling expression stabilization:

```bash
pixi run analyze-expression-sensitivity \
  --avatars maria,elena,marcus \
  --perturbation-z 0.5 \
  --out expression-sensitivity
```

The analyzer renders positive/negative perturbations for every coordinate,
writes pixel heatmaps plus post-stitch keypoint and face-radius metrics, and creates
`expression_profile.review.json`. Its 39 weights intentionally start at zero;
review the contact sheets and set only coordinates that are safe to constrain.
Then run expression-only or combined experiments:

```bash
pixi run generate_offline \
  --speech example/speaker_1.ogg \
  --avatar maria \
  --bg plain_white \
  --motion-stabilization both \
  --expression-profile expression-sensitivity/expression_profile.review.json \
  --motion-debug-jsonl combined-debug.jsonl \
  --out combined-guards.mp4
```

Stabilization corrects only the motion copy sent to the renderer; the model's
autoregressive history remains unchanged. Filter carry is serialized with the
normal renderer state, so corrections remain continuous across API chunks.

The spike guard, continuous One Euro filter, and acceleration/jerk limiter are
independent layers. For example, this runs the current moderate experimental
rotation candidate without the spike guard or kinematic limiters:

```bash
pixi run generate_offline \
  --speech example/speaker_1.ogg \
  --avatar maria \
  --bg plain_white \
  --cfg-self-audio 2.0 \
  --cfg-other-audio 1.0 \
  --cfg-kp 3.0 \
  --noise-alpha 2.0 \
  --noise-trunc-z 1.2 \
  --ode-steps 5 \
  --seed 1234 \
  --motion-stabilization rotation \
  --no-rotation-spike-guard \
  --rotation-temporal-filter one_euro \
  --rotation-one-euro-min-cutoff-hz 2.0 \
  --rotation-one-euro-beta 0.1 \
  --rotation-temporal-max-correction-deg 0.5 \
  --motion-debug-jsonl filtered-motion.jsonl \
  --out filtered.mp4
```

On one 15.2-second Maria speaking sample, this setting reduced within-trace
head step by `17.0%`, acceleration by `44.5%`, jerk by `41.9%`, and
chunk-boundary movement by `33.7%`. It corrected every generated frame, reached
the `0.5 degree` cap on `15.3%` of frames, and showed approximately one frame
of pose lag. These are experimental results, not production defaults. The same
run did not materially improve face/lipsync-region size variation.

Continuous expression filtering uses the same reviewed 39-coordinate profile
as the expression spike guard. Zero-weight coordinates remain byte-for-byte
unchanged. The One Euro filter raises its cutoff during fast motion, preserving
more speech articulation than a fixed low-pass filter. Each layer is bounded by
its configured per-frame maximum; when layers are combined, their final summed
correction can exceed one layer's cap.

### Deterministic morphing-stage analysis

Independent seeded TensorRT runs can produce different raw trajectories. Do
not use them to attribute small rendered-geometry changes. Capture the raw
normalized trajectory once:

```bash
pixi run generate_offline \
  --speech example/speaker_1.ogg \
  --avatar maria \
  --bg plain_white \
  --cfg-self-audio 2.0 \
  --cfg-other-audio 1.0 \
  --cfg-kp 3.0 \
  --seed 1234 \
  --motion-capture artifacts/maria-motion.safetensors \
  --motion-debug-jsonl artifacts/capture.jsonl \
  --out artifacts/capture.mp4
```

Replay validates the trajectory fingerprint, audio, avatar registration,
chunking, and raw motion options before replacing the generated motion at the
single pre-stabilization boundary. Capture the baseline stages losslessly:

```bash
pixi run generate_offline \
  --speech example/speaker_1.ogg \
  --avatar maria \
  --bg plain_white \
  --cfg-self-audio 2.0 \
  --cfg-other-audio 1.0 \
  --cfg-kp 3.0 \
  --seed 1234 \
  --motion-replay artifacts/maria-motion.safetensors \
  --stage-capture-dir artifacts/baseline-stages \
  --stage-capture-pixels \
  --motion-debug-jsonl artifacts/baseline.jsonl \
  --out artifacts/baseline.mp4
```

The stage directory contains a manifest, raw/network/final keypoint tensors,
stitch corrections, and—when requested—lossless decoded-face and final-frame
PNG sequences. Pixel capture is intentionally expensive and is not a
throughput mode.

Stitch controls are independent and disabled by default. For example, test one
predeclared stitch strength against the same trajectory:

```bash
pixi run generate_offline \
  --speech example/speaker_1.ogg \
  --avatar maria \
  --bg plain_white \
  --cfg-self-audio 2.0 \
  --cfg-other-audio 1.0 \
  --cfg-kp 3.0 \
  --seed 1234 \
  --motion-replay artifacts/maria-motion.safetensors \
  --stitch-strength 0.75 \
  --stage-capture-dir artifacts/stitch-075-stages \
  --stage-capture-pixels \
  --motion-debug-jsonl artifacts/stitch-075.jsonl \
  --out artifacts/stitch-075.mp4
```

Then rigid-align learned-keypoint geometry, validate every artifact hash, and
compare only captures with the same raw-motion fingerprint:

```bash
pixi run analyze-morphing-stages \
  --baseline artifacts/baseline-stages \
  --candidate stitch075=artifacts/stitch-075-stages \
  --out artifacts/morphing-stage-analysis.json
```

The analyzer reports raw, stitch-network, and final keypoint metrics for all,
lipsync, and source-locked subsets. It removes translation, rotation, and
uniform scale before non-rigid measurements, separates speaking, listening,
overlap, and silence from captured per-frame audio RMS, and adds secondary
optical-flow metrics for decoded and composited pixels. Its `10%` geometry and `5%`
preservation fields are gates, not automatic approval; full-resolution visual
review, mouth range, and an external lip-sync evaluator remain required.

The post-stitch prototype is also disabled by default. It filters only an
explicit reviewed keypoint subset (or the existing source-locked subset),
restores the original per-frame 3D similarity transform, clamps every
per-keypoint correction, and leaves all unselected/lipsync keypoints unchanged.
Do not enable it until deterministic stage analysis implicates final keypoint
geometry.

If final stitched keypoints are stable but decoded pixels remain unstable,
repeat the identical replay with `--renderer-backend onnx`. This forces the
ONNX renderer models while keeping the AVTR-1 motion engines unchanged, so the
stage report can compare the normal TensorRT/FP16 renderer path with a feasible
higher-precision reference. Treat this as localization evidence, not an
assumption that precision is the cause.

### Temporal-filter sweep

Run the predefined baseline, low-threshold spike, adaptive filter, kinematic
limiter, and layered rotation configurations in one reproducible experiment:

```bash
pixi run sweep-rotation-stabilization \
  --speech example/speaker_1.ogg \
  --avatar maria \
  --bg plain_white \
  --out rotation-sweep
```

The sweep writes each complete video separately, preserves each JSONL trace and
console log, and creates `rotation_sweep_summary.csv` plus JSON configuration
and metric manifests. Head step, acceleration, jerk, and chunk-boundary metrics
are calculated over the complete frame timeline. Its selection score penalizes
excessive correction and loss of expression activity; lower lipsync movement is
not treated as an unconditional improvement. Pass a reviewed profile with at
least one nonzero weight to append expression-only and combined runs:

```bash
pixi run sweep-rotation-stabilization \
  --speech example/speaker_1.ogg \
  --avatar maria \
  --bg plain_white \
  --expression-profile expression-sensitivity/expression_profile.review.json \
  --out combined-sweep
```

Motion diagnostics additionally include raw, spike-corrected and final motion;
requested and applied corrections for every layer; per-keypoint XYZ tracks;
per-keypoint temporal steps; and stitch correction by keypoint. Learned
keypoints are deliberately logged by numeric ID rather than assigned
unsupported anatomical labels.

The completed mechanism experiment indicated that continuous One Euro rotation
filtering was effective, low-threshold spike-only presets were weak, and the
tested acceleration/jerk limiters increased undesirable head step or boundary
movement. Layered spike plus temporal correction can exceed the temporal cap
because each layer is bounded independently. Expression filtering remains
experimental: coordinate ablations changed the intended learned coordinates,
but did not establish a causal reduction in visible facial morphing.

Available avatars are the filenames (without `.png`) inside
`$AVTR1_LOCAL_STORAGE/v1/avatars_artifacts/reference_frames/` after downloading.

### Native-size portraits

The default offline canvas is 1280x720. Portraits with a different aspect
ratio are resized to that canvas by the renderer. For testing an arbitrary
portrait without changing its aspect ratio, render at the portrait's native
size:

```bash
pixi run generate_offline \
  --speech example/speaker_1.ogg \
  --avatar custom_portrait \
  --bg plain_white \
  --native-size \
  --out custom_portrait.mp4
```

The portrait must exist as `custom_portrait.png` in the `reference_frames`
artifact directory. Native width and height are used as the output canvas;
odd dimensions are reduced by one pixel because YUV420 video requires even
dimensions. Large source images increase renderer memory usage and output
encoding cost.

---

## 2. Performance

### Per-chunk latency

AVTR-1 generates motion in 5-frame chunks end-to-end. At 25 fps that's 200 ms
of output per chunk, so any GPU under that line runs in real-time.

| GPU         | Latency / 5-frame chunk | Real-time factor |
| ----------- | ----------------------- | ---------------- |
| L40         | 84 ms                   | 2.4×             |
| A100        | 91 ms                   | 2.2×             |
| RTX 4060 Ti | 166 ms                  | 1.2×             |
| RTX 3070    | 181 ms                  | 1.1×             |
| L4          | 202 ms                  | 0.99×            |
| RTX 3060 Ti | 206 ms                  | 0.97×            |
| RTX 4060    | 232 ms                  | 0.86×            |

Real-time factor = 200 ms / latency. ≥ 1.0× means the GPU keeps up with 25 fps.

---

## Troubleshooting

<details>
<summary><b>TURN server setup</b> (optional)</summary>

ICE tries direct UDP first (host candidates + STUN-reflexive
candidates from a public STUN server) and only needs a TURN relay when the
network in between can't pass UDP between browser and streamer — typical
when the streamer lives on a cloud VM whose security group blocks inbound
UDP, or when one peer is behind symmetric NAT.

If direct UDP works for your setup you can skip this section entirely. The
browser's connectivity card after the engine dropdown tells you which path
ICE actually picked, and the same UI links back here when the verdict is
"only TURN works" or "nothing worked".

The project is wired for **Cloudflare's Realtime TURN**. The free tier is
generous enough for development; no credit card required.

**1. Create a TURN application on Cloudflare**

- Sign in to [dash.cloudflare.com](https://dash.cloudflare.com).
- Navigate to **Realtime → TURN Server**.
- Click **Create TURN App**, give it a name (e.g. `avtr1-dev`), and submit.

**2. Copy the two credential values**

On the application's detail page you'll see:

- **Turn Key ID** — short identifier (looks like a UUID without dashes).
- **API Token** — long secret shown only once at creation. Save it before
  navigating away.

**3. Put them in `.env`**

```dotenv
CLOUDFLARE_TURN_KEY_ID="<Turn Key ID>"
CLOUDFLARE_TURN_KEY_TOKEN="<API Token>"
```

That's it. On the next `/ice-servers` request the streamer mints a fresh,
short-lived TURN credential per session via Cloudflare's
`/v1/turn/keys/{kid}/credentials/generate` endpoint — the long-lived API
token never leaves the server. You can verify it picked up the keys by
watching the streamer log for `ice: using Cloudflare TURN` on the first
browser request.

The browser-side connectivity probe (the small status card under the
controls) tells you which ICE path actually wins:

- ✓ **host** — the browser saw its own local interface; always present.
- ✓ **server-reflexive via STUN** — the browser learned its public IP via
  STUN; doesn't prove the streamer is reachable on UDP from the browser.
- ✓ **relay via TURN** — the browser successfully allocated a Cloudflare
  TURN relay; required when direct UDP can't traverse the network in
  between.

If the relay check fails while TURN is configured the most likely cause is
wrong credentials — re-check that you copied the full **API Token** (not
the Key ID twice) into `CLOUDFLARE_TURN_KEY_TOKEN`.

**Alternatives.** Anything that speaks the standard TURN protocol works.
Set `TURN_URL` (and optionally `TURN_USERNAME` / `TURN_CREDENTIAL`) instead
of the Cloudflare variables and `resolve_ice_servers()` will use it
verbatim — e.g. a self-hosted [coturn](https://github.com/coturn/coturn) on
a small VM. STUN-only also works *if* you can open the appropriate UDP
port range inbound on whatever firewall sits in front of the streamer.

</details>

---

## License

This repository contains three separately licensed components:

- **`scripts/`** — build and demo tooling, released under the **AVTR-1
  Community License** ([LICENSE-MODEL.md](LICENSE-MODEL.md)). Permits
  commercial use by entities under USD 10M annual revenue; entities at or
  above that threshold need a commercial agreement. The same license governs
  the AVTR-1 weights distributed at
  [avaturn-live/avtr-1](https://huggingface.co/avaturn-live/avtr-1).
- **`src/avtr1_renderer/`** — Avaturn Renderer (inference pipeline), released
  under the **PolyForm Noncommercial License 1.0.0** with a Required Notice
  ([LICENSE-RENDERER.md](LICENSE-RENDERER.md)). **Noncommercial use only**,
  regardless of revenue; any commercial use needs a separate Renderer
  Commercial License.
- **`src/avaturn_live_streamer/`** — Avaturn Streamer (orchestration
  backend), released under the **PolyForm Noncommercial License 1.0.0** with
  a Required Notice and patent reservation
  ([LICENSE-STREAMER.md](LICENSE-STREAMER.md),
  [PATENTS.md](PATENTS.md)). **Noncommercial use only**, regardless of
  revenue; any commercial use needs a separate Streamer Commercial License.

See [LICENSE.md](LICENSE.md) for the full component map and the consequences
of the multi-license structure. In any conflict between this summary and the
underlying license files, the license files control.

### Non-commercial dependency

The pipeline uses InsightFace's pretrained SCRFD detector and 2D106 landmark
model, which are licensed for **non-commercial research use only**. To use
AVTR-1 commercially you must either obtain a commercial license from
InsightFace (deepinsight@gmail.com) or replace these models with
permissively-licensed alternatives (e.g., MediaPipe). See
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the full picture.

**Commercial inquiries:** hello@avaturn.me
