# AVTR-1 Motion Stability Handoff

Last updated: 2026-07-20
Repository: `https://github.com/Yuvrajxms09/avtr-1`
Branch: `main`
Current implementation commit: `b53f2c7` (`Fix post-stitch keypoint indexing`)
Current working tree: clean and pushed. Deterministic replay, geometry/pixel
stage capture and analysis, stitch controls, post-stitch controls, and
turn-aware CFG guidance have executed successfully on the real Colab
CUDA/TensorRT path. The new unit/static test suite has still not been run in
that environment.

## 1. Purpose of this document

This document captures the investigation into two quality problems reported in
AVTR-1 output:

1. The head sometimes moves in a jittery or unnatural way.
2. The apparent size or shape of facial regions changes over time, creating a
   morphing effect.

It records what has been observed, what was tested, which configuration is the
current candidate, what has and has not been proven, the diagnostic and
stabilization work now present in the repository, and the recommended next
steps toward a production solution.

The main distinction to preserve is:

- `cfg_self_audio=2, cfg_other_audio=1` remains the best tested **speaking
  guidance** configuration.
- The current logs-only **speaking head-stability** candidate keeps that CFG
  configuration and adds rotation-only One Euro filtering with cutoff `2.0
  Hz`, beta `0.10`, and a `0.5 degree` temporal correction cap.
- The default `cfg_other_audio=2` remains better during **listening** in the
  completed dual-audio experiment. The rotation filter has not been validated
  for listening, interruption, or overlap.
- The One Euro candidate materially reduces measured head jitter. It has not
  materially reduced face/lipsync-region size variation, so the facial
  morphing complaint remains unresolved.
- Deterministic stage isolation now shows that the raw source-locked geometry
  is effectively static and that the stitch network is the first stage that
  introduces the measured source-locked non-rigid motion.
- Reducing stitch strength improves geometry metrics causally on the identical
  raw trajectory. Full stitch bypass (`stitch_strength=0`) produced the largest
  numerical morphing reduction, but the resulting video was visually assessed
  as not good and is rejected. Metric improvement is therefore not an accepted
  morphing fix.

Therefore there is no single globally proven production configuration yet.

### Agent operating brief

This handoff is intended to be executable by another engineer or AI agent. Use
the following status vocabulary exactly:

- **Observed:** directly present in a completed video, trace, or metric output.
- **Tested candidate:** evaluated against baseline, but not necessarily ready
  for production.
- **Source-implemented, unverified:** code exists in the current working tree,
  but no build, test, or CUDA execution has been run for those changes.
- **Implemented, not validated:** code exists and passed local CPU/static tests,
  but the real CUDA/TensorRT experiment has not run.
- **Hypothesis:** a possible explanation that still needs an isolating test.
- **Production-approved:** passed the acceptance checklist in Section 19. No
  current stabilization configuration has this status.

Current state snapshot:

| Item | Status | What it means |
| --- | --- | --- |
| Jitter and facial morphing complaints | Observed | These are the two target defects. |
| `self=2, other=1` during avatar speech | Tested guidance candidate | Better speaking stability than repository-default `other=2`; it remains the base configuration for stabilization tests. |
| `self=2, other=2` during listening | Tested candidate/default | Better listening stability than `other=1` in the completed dual-audio test. |
| Initial `0.75 degree` rotation spike guard | Tested and ineffective | It corrected zero frames; the threshold was too high or the jitter was not isolated-spike behavior. |
| Maria expression sensitivity map | Observed sensitivity, not temporal causation | It identifies coordinates capable of changing geometry, not coordinates proven to cause runtime morphing. |
| Lower-threshold rotation spike guards | Tested and weak | They changed only 5-16 of 380 frames, did not reduce head step or boundaries, and were not the main solution. |
| Rotation One Euro filter | Tested candidate | Moderate filtering reduced within-run head step `17.0%`, acceleration `44.5%`, jerk `41.9%`, and chunk-boundary movement `33.7%`. |
| Rotation acceleration/jerk limiters | Tested and rejected at current settings | They reduced selected derivatives by increasing head step or boundary movement. |
| Expression temporal filters and four-coordinate profile | Tested, not accepted | The filters changed the intended coordinates, but rendered face/lipsync improvements were too small and confounded by raw-run variation. |
| Facial morphing fix | Not established; runtime path near saturation | The defect was localized to stitch-induced geometry on one deterministic sample. Stitch bypass improved geometry metrics but failed visual review. Temporal stitch/post-stitch filters primarily reduced jitter and did not establish a visual morphing fix. |
| Deterministic replay and stage capture | Executed on CUDA/TensorRT | Capture/replay, strict fingerprints, geometry/pixel artifacts, and the rigid-aligned analyzer completed across baseline and all tested branches with fingerprint `c83a1d3576d9e052a1400de9ca5eb7993bf9bef3a34286f7208dbdec0a61956a`. |
| Stitch/post-stitch controls | Tested candidates, none accepted | Strengths `0.00`, `0.25`, `0.50`, `0.75`, and `1.00`, stitch One Euro, and source-locked post-stitch filtering were evaluated against one identical trajectory. Strength `0.00` was visually rejected; no candidate is production-approved. |
| Turn-aware CFG switching | Partially runtime-validated | Automatic guidance completed on speech plus a silent listen track: 61 speaking and 15 silence chunks, one stable-state transition, and no runtime failure. Real listening, overlap, interruption, and serialized API-state validation remain open. |

The head-jitter parameter/filter investigation is near saturation. Moderate
rotation-only One Euro is the current head-stability candidate; more broad
rotation sweeps are unlikely to produce a material gain without adding lag or
making intentional motion look damped. The expression-coordinate investigation
is also near saturation as a morphing solution. It remains useful for diagnosis,
but it did not materially reduce the visible defect and must not be repeated as
another broad sweep.

The deterministic morphing-stage isolation task is complete for Maria and
`demoday15.mp3`. Do not repeat the same broad candidate matrix. The next
decision is whether to perform one final visual review of the already-generated
`stitch_strength=0.50` compromise across more samples. If that does not produce
an obvious visual gain without alignment/expression regressions, stop runtime
morphing-filter work and scope stitch/renderer retraining or replacement.

Non-negotiable instructions for the next agent:

1. Do not report lower motion as improved naturalness without visual and
   motion-preservation evidence.
2. Do not call expression coordinates anatomical regions. Use coordinate,
   learned-keypoint, and axis IDs.
3. Do not enable expression filtering from the all-zero review profile.
4. Do not combine rotation and expression changes before each is evaluated
   independently.
5. Do not preprocess portraits with notebook thumbnails, forced 1280x720
   resizing, padding, or recompression. Use official registration and
   `--native-size` for arbitrary portraits.
6. Do not use a comparison-grid encoding as the primary quality artifact.
   Inspect complete original-color videos first.
7. Do not infer that a filter worked unless its trace records nonzero applied
   correction.
8. Keep the seed, audio, avatar, output sizing, and non-tested parameters fixed
   within an A/B group.
9. Separate speaking, listening, overlap, and silence metrics.
10. Preserve the default-disabled behavior until a candidate passes the stated
    production checklist.
11. Do not repeat broad expression-coordinate or layered-expression sweeps.
    Coordinates `34`, `37`, `4`, and `28` are sensitivity probes, not accepted
    stabilization targets.
12. Do not optimize only whole-face or lipsync-subset radius. Those aggregates
    can improve while local shape, identity, or mouth articulation regresses.
13. Visual rejection overrides a passing geometry gate. Do not call
    `stitch_strength=0` a morphing fix; its metric improvement did not survive
    full-resolution visual review.
14. Do not use generic whole-frame smoothing as evidence of a local morphing
    fix. Global stabilization and non-rigid facial stability are separate
    problems.

## 2. Product and quality context

AVTR-1 is being evaluated as a real-time avatar pipeline. It generates five
frames per motion chunk at 25 fps, so one model chunk represents 200 ms of
video. The renderer turns audio-conditioned motion into head rotation and 39
learned expression coordinates, stitches those predictions onto the source
portrait geometry, and renders the final face.

The quality target is not simply minimum movement. The avatar must remain:

- Smooth without looking frozen.
- Responsive while listening.
- Expressive and synchronized while speaking.
- Stable in identity and facial geometry.
- Continuous across five-frame chunk boundaries.
- Fast enough for real-time serving.

This matters because a configuration can score as "smoother" merely by
suppressing valid speech motion. That would improve a naive stability metric
while making the avatar less natural. Every stabilization result must therefore
be evaluated against motion range, listening activity, and lip synchronization.

## 3. Current runtime flow

The relevant runtime path is:

```text
speech audio + listening audio
        |
        v
HuBERT audio features
        |
        v
AVTR-1 condition encoder
        |
        v
AVTR-1 guided ODE decoder
  - past-motion condition
  - self/speaking-audio condition
  - other/listening-audio condition
  - portrait-keypoint condition
        |
        v
normalized motion [3 rotation + 39 expression values]
        |
        +------------------------------+
        | raw motion remains in the    |
        | autoregressive model history |
        +------------------------------+
        |
        v
optional render-only temporal stabilization
        |
        v
de-normalization and SO(3) rotation conversion
        |
        v
LivePortrait motion stitching
        |
        v
warp + face decoder + pasteback + matting
        |
        v
25 fps output video
```

The stabilization layer deliberately edits only the copy sent to the renderer.
Raw model output continues into autoregressive history. Feeding corrected motion
back into the model could create an unmeasured distribution shift and cause
future chunks to behave differently from training.

The guidance parameters are runtime TensorRT inputs and do not require an
engine rebuild:

- `cfg_self_audio`: guidance from the avatar's speaking track.
- `cfg_other_audio`: guidance from the other/user/listening track.
- `cfg_kp`: portrait/keypoint guidance.

Even when the other track is silence, changing `cfg_other_audio` changes the
guided decode result empirically. The exact learned interaction is inside the
shipped model; it should not be described as a simple, isolated "listening
strength" control.

## 4. Manager feedback and observed symptoms

The original feedback was that:

- Facial objects or regions appeared to change size.
- Head movement looked jittery or unnatural.
- The result was less smooth than a real human performance.

The CTO later clarified that the requested direction is to continue improving
the current real-time pipeline, especially morphing and jitter. The mention of
a human performance plus character replacement was a quality reference for how
natural motion can look, not a request to introduce a human actor into every
real-time session.

The two complaints can be related but should not be treated as identical:

- A head-rotation jump changes perspective and can make face regions appear to
  expand, contract, or shift.
- Expression-coordinate instability can deform local geometry independently of
  head rotation.
- Motion stitching can amplify or compensate for predicted geometry.
- Matting and pasteback can create boundary flicker that resembles geometric
  instability without changing the underlying keypoints.

## 5. Initial diagnostic findings

A 60-second Maria run produced 301 chunks and 1,505 frames. The important
distribution values were:

| Metric | p50 | p95 | p99 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Head step, degrees | 0.2564 | 0.9852 | 1.4756 | 1.7034 |
| Chunk-boundary head jump, normalized | 0.0181 | 0.0802 | 0.1392 | 0.1599 |
| Expression step, normalized | 1.0846 | 2.3541 | 2.9599 | 3.8244 |
| Expression boundary jump, normalized | 0.4403 | 1.8870 | 2.8713 | 3.1588 |
| Stitched face-radius variation, percent | 0.1446 | 0.4369 | 0.7471 | 0.9902 |
| Stitched lipsync-region variation, percent | 0.2731 | 0.8752 | 1.3693 | 1.8576 |
| Stitch correction | 0.0179 | 0.0482 | 0.0614 | 0.0681 |
| Matte coverage swing, percentage points | 0.0265 | 0.0990 | 0.1351 | 0.1576 |

The measured correlation between head movement and stitched face-size
variation was approximately `0.623`. The correlation between stitch correction
and stitched face-size variation was approximately `0.372`.

These are correlations, not proof of causation, but they support the working
hypothesis that head motion contributes materially to perceived face-size
change while not explaining all of it. Expression geometry and stitching still
need separate analysis.

The worst face-radius and lipsync-radius events often occurred in the same
chunks, for example around 22.6 seconds, 24.2 seconds, and 53.6 seconds in that
run. This is useful for targeted frame-level review.

## 6. Guidance-parameter experiments

### 6.1 Initial one-factor motion sweep

The first broad sweep changed one inference variable at a time around the
repository defaults. It covered the controls most likely to affect temporal
motion:

- `cfg_self_audio`.
- `cfg_other_audio`.
- `cfg_kp`.
- `noise_alpha`.
- `noise_trunc_z`.
- `ode_steps`.

The seed was held at `1234` so that changes were more directly comparable. The
main result was that `cfg_other_audio` had the strongest favorable relationship
with the measured head and geometry stability. In that speech-only run:

| Configuration | Stability score | Head p95 | Face-radius p95 | Lipsync-radius p95 |
| --- | ---: | ---: | ---: | ---: |
| Default baseline | 1.0000 | 1.0083 | 0.4512 | 0.8369 |
| `cfg_other_audio=0` | 0.6507 | 0.5403 | 0.3107 | 0.6089 |
| `cfg_other_audio=1` | 0.7978 | 0.7533 | 0.3706 | 0.6906 |
| `cfg_self_audio=1` | 0.8379 | 0.7723 | 0.4019 | 0.7173 |

The worst measured configurations in that sweep included:

| Configuration | Stability score | Head p95 | Face-radius p95 | Lipsync-radius p95 |
| --- | ---: | ---: | ---: | ---: |
| `noise_alpha=0` | 1.6299 | 1.5787 | 0.7316 | 1.4250 |
| `noise_alpha=1` | 1.3832 | 1.1436 | 0.6750 | 1.2715 |

This supported retaining the default `noise_alpha=2`. It also showed why a
purely metric-ranked optimum is unsafe: `cfg_other_audio=0` produced the lowest
variation, but it removes that guidance contribution entirely and may suppress
desirable motion. Visual differences among the leading configurations were
subtle. The more conservative `cfg_other_audio=1` was carried forward for
multi-avatar and dual-audio validation.

### 6.2 Important metric limitation

Early sweeps ranked configurations over an entire video. In a dual-audio test,
most chunks contained avatar speech and relatively few contained user speech.
An overall score was therefore dominated by the speaking section and could not
establish the best listening configuration.

Later analysis split chunks into:

- Avatar speaking, user silent.
- User speaking, avatar listening.
- Both speaking.
- Both silent.

All future evaluation should keep this segmentation.

### 6.3 `cfg_other_audio` dual-audio sweep

The test used:

- Avatar speech with 15 seconds of leading silence.
- A roughly 14-second user/listening track during that leading silence.
- No overlap between speaking and listening tracks.
- `cfg_self_audio=2` held fixed.
- `cfg_other_audio` swept through `0.5, 1.0, 1.5, 2.0, 2.5, 3.0`.
- Default `cfg_other_audio=2` treated as baseline.

Overall results, where lower is smoother:

| `cfg_other_audio` | Stability score | Head p95 | Face-radius p95 | Lipsync-radius p95 |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.7082 | 0.5718 | 0.3800 | 0.7086 |
| 1.0 | 0.8243 | 0.6642 | 0.4443 | 0.8226 |
| 1.5 | 0.9264 | 0.7739 | 0.4854 | 0.9201 |
| 2.0 | 1.0000 | 0.8703 | 0.5174 | 0.9666 |
| 2.5 | 1.1162 | 0.9770 | 0.5690 | 1.0886 |
| 3.0 | 1.2245 | 1.0718 | 0.6269 | 1.1892 |

This trend alone does not mean `0.5` is best. Lower guidance also reduces valid
motion and may make the avatar under-reactive.

During avatar speaking, the comparison between the current candidate and
baseline was:

| Speaking metric | `other=1` | `other=2` baseline | Relative reduction |
| --- | ---: | ---: | ---: |
| Head-step p95, degrees | 0.7329 | 0.9402 | about 22.0% |
| Face-radius p95, percent | 0.4944 | 0.5947 | about 16.9% |
| Lipsync-radius p95, percent | 0.8994 | 1.1349 | about 20.8% |

During listening, baseline remained better on the measured geometry metrics:

| Listening metric | `other=1` | `other=2` baseline |
| --- | ---: | ---: |
| Head-step p95, degrees | 0.3301 | 0.2597 |
| Head-activity standard deviation, degrees | 0.5150 | 0.5428 |
| Face-radius p95, percent | 0.2331 | 0.1897 |
| Lipsync-radius p95, percent | 0.4191 | 0.3593 |

The baseline had lower listening head-step and geometry variation while
retaining slightly more head activity. This is why `other=1` should not be used
as a universal replacement for `other=2`.

### 6.4 `cfg_self_audio` sweep

With `cfg_other_audio=2` fixed, `cfg_self_audio` was swept from `0` through
`3`. Lower values produced much lower movement metrics; for example,
`cfg_self_audio=0` had a speaking stability score of `0.6191` relative to the
baseline score of `1.0`.

This did not establish `cfg_self_audio=0` as a quality improvement. It largely
demonstrated that removing speaking guidance suppresses movement. That is not
the objective. The current candidate therefore keeps the repository default:

```text
cfg_self_audio = 2.0
```

### 6.5 Multi-avatar speech-only A/B validation

The candidate `self=2, other=1` was compared against baseline `self=2,
other=2` using three arbitrary portraits and speech-only input.

For the first audio sample:

| Portrait | Stability ratio | Head improvement | Face improvement | Lipsync-region improvement |
| --- | ---: | ---: | ---: | ---: |
| `modern_2025_14.png` | 0.8045 | 21.83% | 20.65% | 16.18% |
| `rdj_close_up.jpg` | 0.8086 | 10.42% | 24.28% | 22.72% |
| `leo_image.png` | 0.9083 | 5.10% | 11.77% | 10.64% |
| Mean | 0.8404 | 12.45% | 18.90% | 16.51% |

For the second audio sample:

| Portrait | Stability ratio | Head improvement | Face improvement | Lipsync-region improvement |
| --- | ---: | ---: | ---: | ---: |
| `modern_2025_14.png` | 0.7118 | 31.11% | 30.22% | 25.12% |
| `leo_image.png` | 0.7667 | 29.81% | 19.63% | 20.54% |
| `rdj_close_up.jpg` | 0.8943 | 20.22% | 6.92% | 4.57% |
| Mean stability ratio | 0.7910 | | | |

A positive improvement means the candidate produced lower measured variation.
The effect generalized in sign across all tested portraits, but its magnitude
depended strongly on the portrait and audio.

The metric direction repeated across these portraits, but the visual difference
was subtle and was not treated as decisive evidence. These earlier tests did
not eliminate morphing or establish fully human-natural motion.

## 7. Current candidate configurations

The current speaking guidance candidate is:

```text
cfg_self_audio = 2.0
cfg_other_audio = 1.0
cfg_kp = 3.0
noise_alpha = 2.0
noise_trunc_z = 1.2
ode_steps = 5
seed = 1234  # offline reproducibility, not itself a quality optimum
```

The current speaking head-stability candidate adds:

```text
motion_stabilization                   = rotation
rotation_spike_guard                   = false
rotation_temporal_filter               = one_euro
rotation_one_euro_min_cutoff_hz        = 2.0
rotation_one_euro_beta                 = 0.10
rotation_one_euro_derivative_cutoff_hz = 1.0
rotation_temporal_max_correction_deg   = 0.5
rotation_max_acceleration_deg          = 0.0
rotation_max_jerk_deg                  = 0.0
expression stabilization              = disabled
```

The conservative head-stability fallback uses cutoff `3.0`, beta `0.20`, and
cap `0.4 degrees` instead.

The current listening candidate remains the repository defaults:

```text
cfg_self_audio = 2.0
cfg_other_audio = 2.0
cfg_kp = 3.0
noise_alpha = 2.0
noise_trunc_z = 1.2
ode_steps = 5
```

Recommended terminology:

- Call `self=2, other=1` the **current speaking guidance candidate**.
- Call moderate One Euro plus that CFG the **current speaking head-stability
  candidate**.
- Do not call it the globally optimal AVTR-1 configuration.
- Call `self=2, other=2` the **current listening/default configuration**.
- Do not call expression filtering a morphing fix.

### Production use of both configurations

The model already accepts CFG values per chunk, so using both values in one
session is technically straightforward. A production controller can choose
`cfg_other_audio` from the speaking/listening state:

```text
avatar speaking                  -> cfg_other_audio = 1.0
avatar listening                 -> cfg_other_audio = 2.0
overlap or uncertain transition  -> hold previous state or interpolate
```

This should not be implemented as an unfiltered hard switch on every audio
frame. Use a chunk-level state machine with:

- Voice activity detection or known turn state.
- Hysteresis to avoid rapid toggling.
- A minimum state duration.
- A short transition or interpolation over one or more five-frame chunks.
- Explicit testing at speech start, speech end, interruption, and overlap.

The current decode engine accepts one guidance value for the five-frame chunk.
Per-frame switching would require a different model/export interface and is not
needed for the first production prototype.

## 8. First stabilization attempt and its result

An initial isolated-rotation-spike guard was tested with:

```text
rotation acceleration threshold = 0.75 degrees
rotation maximum correction = 1.5 degrees
rotation strength = 1.0
```

It examined 365 frames and corrected zero frames:

```text
intervention rate = 0.00%
correction p95 = 0.000 degrees
maximum correction = 0.000 degrees
```

Raw and corrected direct-motion values were identical:

| Metric | Raw | Corrected | Improvement |
| --- | ---: | ---: | ---: |
| Head-step p95 | 0.44357 degrees | 0.44357 degrees | 0.00% |
| Head-acceleration p95 | 0.36968 degrees | 0.36968 degrees | 0.00% |

Rendered metrics were slightly worse in the guard run:

| Rendered metric | Baseline | Rotation-guard run |
| --- | ---: | ---: |
| Head-step p95 | 0.44900 | 0.45017 |
| Expression-step p95 | 1.63882 | 1.63862 |
| Face-radius variation p95 | 0.54081% | 0.56973% |
| Lipsync-radius variation p95 | 1.02318% | 1.06833% |
| Stitch correction p95 | 0.02379 | 0.02385 |
| Matte swing p95 | 0.06851 pp | 0.07298 pp |

Because the guard applied no correction, those small differences cannot be
credited to stabilization. They are run-to-run/render variation. The useful
conclusion is that `0.75 degrees` was too high for this sample and that an
isolated-spike detector alone is not sufficient for lower-amplitude continuous
jitter.

## 9. Expression sensitivity analysis

The model predicts 39 normalized expression coordinates. They correspond to
selected axes of learned LivePortrait keypoints, but they do not have reliable
semantic labels such as "left cheek" or "mouth corner".

The implemented sensitivity analyzer perturbs each coordinate by `-0.5z` and
`+0.5z`, renders through the production stitch/warp/decoder path, and records:

- Pixel change magnitude and spatial region.
- Keypoint displacement.
- Face-radius change.
- Lipsync-keypoint-subset radius change.
- Width and height change.
- Contact sheets and heatmaps.

The completed sensitivity run used Maria only, with 78 renders: two
perturbations for each of 39 coordinates. It has not yet established that the
same ranking holds across different portraits.

The strongest measured geometry coordinates were:

| Coordinate | Learned keypoint/axis | Dominant image region | Mean absolute RGB change | Face-radius effect | Lipsync-radius effect |
| ---: | --- | --- | ---: | ---: | ---: |
| 34 | keypoint 19, y | lower center | 0.002121 | 0.1445% | 0.2766% |
| 37 | keypoint 20, y | middle center | 0.001993 | 0.1002% | 0.1831% |
| 4 | keypoint 2, y | middle center | 0.003368 | 0.0984% | 0.1733% |
| 28 | keypoint 17, y | middle center | 0.001577 | 0.0715% | 0.1304% |
| 0 | keypoint 1, x | upper center | 0.001456 | 0.0413% | 0.0714% |

Interpretation:

- Coordinates 34, 37, 4, and 28 can materially change measured geometry.
- Coordinate 34 is currently the strongest candidate for explaining local
  size change.
- A large lipsync-radius effect is also a warning that aggressive filtering
  could damage valid mouth motion.
- A neutral `+/-0.5z` perturbation proves sensitivity, not that a coordinate is
  temporally unstable in real inference.
- The spatial regions are coarse difference-map regions, not anatomical labels.

The generated `expression_profile.review.json` intentionally contains 39 zero
weights. This is a safety feature. Expression filtering must not be enabled
until contact sheets and temporal traces have been reviewed and selected
weights have been written explicitly.

## 10. Code added during the investigation

### 10.1 Native-size custom portrait rendering

Commit: `cc70949`

The default offline canvas is 1280x720. Forcing arbitrary square or portrait
images into that canvas caused severe face distortion and invalidated visual
comparisons. `--native-size` now preserves the portrait dimensions, reducing
odd dimensions by one pixel for YUV420 compatibility.

This fix is important for evaluation but is separate from temporal stability.
Do not resize, thumbnail, pad, or recompress test portraits in ad hoc notebook
code. Register the original image and use official inference with
`--native-size`.

### 10.2 Diagnostics and first spike guard

Commit: `e757a5f`

Added:

- Direct exposure of motion/guidance parameters in offline inference and API.
- JSONL diagnostics with per-chunk trace IDs.
- Raw motion, head, expression, keypoint, stitching, and alpha metrics.
- Stateful isolated-spike guards for rotation and reviewed expression
  coordinates.
- Expression sensitivity analyzer and review profile.
- Serialization of filter state across API requests.

The diagnostic path is opt-in because extracting Python values from CUDA
tensors synchronizes the GPU and reduces throughput.

### 10.3 Layered temporal filters and sweep tooling

Commit: `429af51`

Added three independent render-only layers for both rotation and expression:

1. **Isolated spike guard**
   - Detects a one-frame reversal relative to adjacent raw values.
   - Removes only the excess above a threshold.
   - Appropriate for outliers, not continuous low-level jitter.

2. **One Euro adaptive low-pass filter**
   - Uses stronger smoothing when motion is slow.
   - Raises its cutoff as motion speed increases.
   - Intended to reduce jitter while preserving intentional fast motion.

3. **Acceleration and jerk limiter**
   - Constrains abrupt changes in angular or expression velocity.
   - Can preserve a sustained nod while limiting sudden direction changes.
   - Corrections remain bounded by a separate maximum.

The rotation path operates on model-native SO(3) rotation vectors. The observed
head-pose range is far from the 180-degree rotation-vector branch singularity,
so this is simpler and appropriate for the current data. If the pipeline later
supports extreme rotations, quaternion-domain filtering should be reconsidered.

Expression filtering is coordinate-weighted. A zero-weight coordinate remains
unchanged. This prevents an unreviewed filter from silently modifying all mouth
motion.

The implementation records:

- Raw, spike-corrected, and final values.
- Requested and applied correction from each layer.
- Velocity, acceleration, and full-timeline angular jerk.
- Chunk-boundary head and expression jumps.
- Filter intervention rate and correction magnitude.
- Per-keypoint XYZ tracks and temporal steps.
- Per-keypoint stitch correction.
- Face and lipsync-subset geometry variation.
- Expression activity retention.
- Diagnostic filter timing.

Baseline behavior remains unchanged when `motion_stabilization=none`.

### 10.4 Relevant file map

| File | Responsibility |
| --- | --- |
| `src/avtr1_renderer/types.py` | Public render and stabilization options plus validation. |
| `src/avtr1_renderer/motion_stabilizer.py` | Stateful spike, One Euro, acceleration, and jerk correction logic. |
| `src/avtr1_renderer/avtr1_motion_generator.py` | Raw model generation, render-only filter insertion point, and state serialization. |
| `src/avtr1_renderer/diagnostics.py` | JSONL motion, keypoint, stitching, matting, and filter diagnostics. |
| `src/avtr1_renderer/components/liveportrait/motion_stitch.py` | Converts predicted rotation/expression into source/driving keypoints and invokes geometry diagnostics. |
| `scripts/generate_offline.py` | Official offline CLI and all opt-in stabilization flags. |
| `scripts/analyze_expression_sensitivity.py` | Offline coordinate perturbation, rendering, metrics, and review-profile generation. |
| `scripts/sweep_rotation_stabilization.py` | Reproducible 15-run rotation sweep, optional expression runs, and ranking. |
| `src/avtr1_renderer/api/app.py` | Per-request production API exposure of guidance and filter options. |
| `tests/test_motion_stabilizer.py` | CPU filter behavior, bounds, coordinate isolation, and chunk continuity. |
| `tests/test_rotation_stabilization_sweep.py` | Full-timeline rotation metric and sweep-manifest checks. |
| `tests/test_stabilizer_state_codec.py` | Stabilizer carry serialization compatibility. |

### 10.5 Diagnostic event contract

JSONL diagnostics are stage-oriented. A next agent should parse by the `stage`
field and join per-chunk events through `trace_id` where available.

| Stage | Expected frequency | Primary purpose |
| --- | --- | --- |
| `session` | Once per offline run | Records input/output paths, avatar, crop/output settings, seed, and session metadata. |
| `avatar_registration` | Once per avatar load | Records source size, crop parameters, source pose, scale, and translation. |
| `options` | Once per processed chunk | Records CFG, noise, streaming, and every active stabilization setting. |
| `audio_chunk` | Once per processed chunk | Records speech/listen RMS and peak; use it to classify speaking/listening/silence. |
| `motion_prediction` | Once per processed chunk | Records raw head rotvecs, raw normalized expression, step metrics, and boundary context. |
| `motion_stabilization` | Once per filtered chunk only | Records raw/intermediate/final values, layer corrections, kinematics, interventions, and filter time. Baseline intentionally lacks this stage. |
| `keypoint_geometry` | Once per processed chunk | Records source, raw driving, stitched driving, subsets, per-keypoint tracks, and stitch correction. |
| `render_alpha` | Once per processed chunk | Records alpha mean and foreground coverage for matting/pasteback diagnosis. |

Before comparing metrics, verify:

- All runs have the same number of audio, prediction, geometry, and render
  events.
- Filtered runs have the same number of stabilization and prediction events.
- The session metadata matches except for the parameter intentionally changed.
- The same audio chunks are classified into the same speaking/listening states.
- No inference log contains a failure followed by a partially written video or
  trace.

Do not compare old and new diagnostic schemas blindly. Older traces may not
contain full-timeline raw rotvecs, expression arrays, jerk, or per-keypoint
tracks. Re-run baseline with the current commit when using the new sweep.

### 10.6 Important functionality that is not implemented yet

The current repository still does **not** contain:

- An automatic analyzer that correlates all 39 real-time expression coordinate
  traces with per-keypoint or region-level morphing events.
- A trusted mapping from learned coordinates/keypoints to anatomical names.
- Automatic creation of a nonzero expression filter profile.
- An external audio/video lip-sync confidence evaluator.
- Automatic production rollout, feature flagging, or A/B telemetry.
- A model-training fix or access to retraining losses/data.
- Face-aware post-render stabilization or optical-flow restoration.

The repository now contains:

- Fingerprinted raw normalized-motion capture/replay with strict metadata and
  complete-consumption checks.
- Hash-verified raw, stitch-network, and final keypoint artifacts.
- Optional lossless decoded-face and final-composite frame capture.
- A rigid-aligned geometry and secondary pixel-motion comparison analyzer.
- Default-disabled stitch strength, stitch-correction One Euro filtering, and
  post-stitch source-locked residual filtering with serialized carry.
- Default-disabled turn-aware `cfg_other_audio` selection with RMS Schmitt
  thresholds, chunk hysteresis/interpolation, and serialized carry.

The capture/replay, geometry/pixel artifact, analyzer, stitch-strength,
stitch-filter, post-stitch, and speech/silence turn-guidance paths completed on
the real Colab CUDA/TensorRT environment on 2026-07-20. This validates the
runtime plumbing on one avatar/audio sample; it does not validate production
quality, multiple avatars, real listening/overlap, state serialization, or the
new unit/static test suite.

## 11. Completed comprehensive temporal-stability experiment

The broad CUDA/TensorRT experiment was completed on 2026-07-17 at commit
`429af51`. A Colab orchestration cell used the official
`scripts/generate_offline.py` path to run 29 configurations:

- One unfiltered baseline using the prior speaking CFG candidate.
- Five rotation spike variants.
- Three rotation One Euro variants.
- Two rotation acceleration/jerk limiter variants.
- Four layered rotation variants.
- Two expression spike variants.
- Two expression One Euro variants.
- Two expression limiter variants.
- Two layered expression variants.
- Two combined rotation/expression variants.
- Four single-coordinate expression ablations for coordinates `34`, `37`, `4`,
  and `28` with weight `0.2`.

### 11.1 Experiment controls and integrity

```text
commit              = 429af51
avatar              = maria
background          = plain_white
speech              = demoday15.mp3
duration            = 15.2 seconds
frames              = 380 at 25 fps
cfg_self_audio      = 2.0
cfg_other_audio     = 1.0
cfg_kp              = 3.0
noise_alpha         = 2.0
noise_trunc_z       = 1.2
ode_steps           = 5
seed                = 1234
output              = 1280x720
```

All 29 videos and diagnostics completed. Stage counts were internally
consistent. However, all 29 raw prediction fingerprints were different despite
the fixed seed. Exact output-to-output comparisons are therefore confounded by
small TensorRT/run-level variation.

The raw metric variation was much smaller than the useful rotation effects:

| Raw metric across runs | Range relative to mean | Coefficient of variation |
| --- | ---: | ---: |
| Head step | 3.32% | 0.86% |
| Head acceleration | 6.22% | 1.63% |
| Head jerk | 10.17% | 2.76% |
| Expression step | 3.05% | 0.60% |
| Expression acceleration | 6.22% | 2.03% |
| Expression jerk | 4.22% | 0.96% |

Consequently, large rotation changes were evaluated by comparing raw and
corrected motion inside the same JSONL trace. Small 1-3% rendered-geometry
changes were not treated as causal.

### 11.2 Rotation mechanism result

Within-run raw-to-corrected results were:

| Configuration | Head step | Acceleration | Jerk | Chunk boundary |
| --- | ---: | ---: | ---: | ---: |
| One Euro strong, cutoff `1.5`, beta `0.05`, cap `0.5` | 15.3% better | 44.4% better | 32.6% better | 27.2% better |
| **One Euro moderate, cutoff `2.0`, beta `0.10`, cap `0.5`** | **17.0% better** | **44.5% better** | **41.9% better** | **33.7% better** |
| One Euro light, cutoff `3.0`, beta `0.20`, cap `0.4` | 14.4% better | 38.9% better | 44.5% better | 23.2% better |
| Spike plus One Euro moderate | 16.6% better | 44.3% better | 48.4% better | 30.8% better |
| Limiter light | 10.3% worse | 18.2% better | 20.7% better | 5.4% better |
| Limiter moderate | 25.0% worse | 5.4% worse | 20.6% better | 7.7% worse |

This classifies the observed head defect as primarily continuous
high-frequency pose fluctuation rather than sparse one-frame spikes. One Euro
is the useful mechanism. The tested acceleration/jerk limiter settings are not
acceptable because they reduce selected derivatives by increasing position
step or boundary movement.

### 11.3 Layer attribution and correction safety

For `12_spike_one_euro_moderate`, the same trajectory measured:

| Stage | Step | Acceleration | Jerk | Boundary |
| --- | ---: | ---: | ---: | ---: |
| Raw | 0.60642 | 0.49453 | 0.59085 | 0.59494 |
| After spike guard | 0.60642 | 0.48842 | 0.59631 | 0.59494 |
| After One Euro | 0.50599 | 0.27561 | 0.30514 | 0.41159 |

The spike layer intervened on only 14 frames, did not improve step or boundary
movement, and slightly worsened jerk before the One Euro layer. Nearly all
useful correction came from One Euro. The pure One Euro configuration is
therefore preferred over the nominally top-ranked layered result.

The layered result also exposed a safety detail: the `0.5 degree` temporal cap
applies to the temporal layer, not to the final sum of temporal and spike
corrections. Four frames exceeded `0.5 degrees`, with a final maximum of
`0.74961 degrees`. A final global correction cap is required before layered
filters could be considered production-safe.

For the preferred moderate One Euro run:

```text
frames corrected                  = 379 / 379
median correction                 = 0.2412 degrees
temporal-cap frames               = 58 / 380 (15.3%)
maximum correction                = 0.5 degrees
best raw/corrected alignment      = one frame
approximate pose lag at 25 fps    = 40 ms
```

The light One Euro run is the conservative fallback. Its median correction was
`0.1859 degrees`, its cap was `0.4 degrees`, and it retained substantial
head-jerk reduction.

### 11.4 Expression and facial-morphing result

The four single-coordinate One Euro ablations changed only the intended
coordinate and reduced its own derivatives:

| Coordinate | Step | Acceleration | Jerk |
| ---: | ---: | ---: | ---: |
| 34 | 3.7% better | 8.4% better | 7.9% better |
| 37 | 5.7% better | 4.4% better | 8.2% better |
| 4 | 3.8% better | 9.9% better | 6.6% better |
| 28 | 6.4% better | 9.6% better | 13.5% better |

This verifies that the coordinate-weighted implementation functions. It does
not establish a morphing fix. Reported face/lipsync-radius changes were mostly
0-3%, which is inside the scale of independent raw-run variation. Stronger
layered expression filtering reduced overall expression jerk by only `5.9%`,
retained `97%` of expression activity, and made face-radius and lipsync-radius
metrics `3.31%` and `3.79%` worse than baseline respectively.

Expression filtering is therefore not an accepted candidate. Coordinate `34`
remains useful for controlled investigation because it had the strongest
sensitivity and the best nominal face-radius result, but its strong
lipsync-subset effect makes it risky to filter without deterministic replay and
lip-sync validation.

### 11.5 Encoded-video motion assessment

Three complete, original-color 1280x720 videos were also analyzed directly:

1. Repository default baseline: `self=2, other=2`, no stabilization.
2. Previous speaking optimum: `self=2, other=1`, no stabilization.
3. Moderate One Euro: `self=2, other=1`, rotation-only filtering.

Post-hoc OpenCV diagnostics tracked upper-face features with pyramidal optical
flow, estimated a robust per-frame similarity transform, and measured face-box
and upper/lower-region changes. These are secondary encoded-video measurements,
not canonical model metrics.

| Video-derived p95 metric | Repository baseline | Old speaking optimum | Moderate One Euro |
| --- | ---: | ---: | ---: |
| Global translation, pixels | 4.4482 | 3.6200 | 2.8292 |
| Global rotation, degrees | 0.2234 | 0.1440 | 0.1019 |
| Rotation acceleration, degrees | 0.2525 | 0.1861 | 0.1178 |
| Rotation jerk, degrees | 0.3992 | 0.3349 | 0.2248 |
| Global scale step | 0.4438% | 0.3815% | 0.3400% |
| Face-box center step, pixels | 3.8079 | 3.6056 | 3.1623 |
| Face-box area step | 2.7595% | 2.0970% | 2.0941% |
| Upper/lower local-scale difference | 2.0228% | 1.5494% | 1.5687% |
| Lower-face non-rigid residual, pixels | 20.0665 | 13.9634 | 13.5422 |

Relative to the old speaking optimum, One Euro reduced encoded-video global
translation `21.84%`, rotation `29.27%`, rotation acceleration `36.71%`,
rotation jerk `32.88%`, global scale fluctuation `10.88%`, and face-box center
step `12.29%`.

It did not materially change local facial-size stability: face-box area step
improved only `0.14%`, upper/lower scale inconsistency worsened `1.24%`, and
lower-face residual improved `3.02%`. Synchronized contact sheets and
five-frame sequences were consistent with smoother global pose but did not
show a defensible reduction in facial morphing. No visual claim stronger than
that is supported.

### 11.6 Completed-experiment conclusion

- The previous CFG-only speaking optimum was improved further for head
  stability by rotation-only One Euro filtering.
- The current logs-only candidate is moderate One Euro. The light One Euro
  preset is the safer fallback.
- Spike-only and limiter-only mechanisms are rejected at their tested values.
- Layering spike and One Euro is not justified and can exceed the temporal cap.
- Facial morphing remains unresolved. Do not present the expression experiments
  as a fix.
- None of these settings is production-approved because only one avatar, one
  speaking audio, and one turn state were used, and the moderate candidate
  reaches its correction cap frequently.
- Further broad rotation or expression-coordinate sweeps are not the active
  direction. The remaining morphing work must first identify the stage that
  introduces non-rigid facial instability.

## 12. Recommended immediate experiment

### 12.1 Add deterministic raw-motion replay

Generate and persist one raw head/expression trajectory. The capture must
include:

- Raw rotation matrices or rotation vectors and all 39 normalized expression
  coordinates for every frame.
- Seed, audio digest, avatar/source registration identity, frame rate, chunk
  boundaries, duration, turn classification, motion configuration, repository
  commit, and model/artifact identifiers.
- A stable fingerprint over the raw arrays and metadata used by every replay.

Then render that exact trajectory through:

1. No stabilization.
2. Moderate One Euro: cutoff `2.0`, beta `0.10`, cap `0.5`.
3. Light One Euro: cutoff `3.0`, beta `0.20`, cap `0.4`.
4. The stage-localization captures defined in Section 13.6: raw driving
   keypoints, stitched keypoints, decoded face crops, and final frames.

The renderer must consume identical raw arrays in every branch. Compare both
motion and rendered geometry from that shared input. Independent seeded
inference is not sufficient for deciding whether a 1-3% facial metric change
is causal. Abort an A/B comparison if the raw-motion fingerprints differ.

Coordinates `34`, `37`, `4`, and `28` remain available as controlled probes
only if rigid-aligned raw driving geometry is identified as the first unstable
stage. They are not part of the initial morphing replay.

### 12.2 Current speaking candidate for validation

```text
cfg_self_audio                         = 2.0
cfg_other_audio                        = 1.0
cfg_kp                                 = 3.0
noise_alpha                            = 2.0
noise_trunc_z                          = 1.2
ode_steps                              = 5

motion_stabilization                   = rotation
rotation_spike_guard                   = false
rotation_temporal_filter               = one_euro
rotation_one_euro_min_cutoff_hz        = 2.0
rotation_one_euro_beta                 = 0.10
rotation_one_euro_derivative_cutoff_hz = 1.0
rotation_temporal_max_correction_deg   = 0.5
rotation_max_acceleration_deg          = 0.0
rotation_max_jerk_deg                  = 0.0
expression stabilization              = disabled
```

This is a tested candidate, not a new default.

### 12.3 Validation sequence

After deterministic replay, compare baseline, moderate One Euro, and light One
Euro across:

- At least three avatars with different face shape and registration geometry.
- At least three speaking samples with different energy and cadence.
- Multiple seeds or captured raw trajectories.
- One long sample to expose filter drift.
- Listening, silence, interruption, and overlap using the dual-audio path.
- External lip-sync confidence and mouth-range preservation.

The working tree now contains the default-disabled turn-aware serving
prototype:

```text
avatar speaking  -> cfg_other_audio = 1.0
avatar listening -> cfg_other_audio = 2.0
```

Its transition state is serialized and uses chunk hysteresis/interpolation so a
turn boundary does not create a hard CFG discontinuity. It still requires the
same CUDA validation across interruption, overlap, silence, and restored API
state before production use.

### 12.4 Confounders to control

- Use captured raw motion for causal filter A/B tests.
- Keep source image, crop, output-size mode, audio, and renderer artifacts fixed.
- Compare on the same GPU/runtime/engine cache.
- Use the same diagnostic mode in every candidate.
- Inspect original generated videos; comparison encodes are presentation-only.
- Record per-layer and final correction caps separately.
- Do not infer expression causation from sensitivity ranking alone.
- Do not optimize only global face radius; preserve mouth range and lip sync.

## 13. Expression evidence and conditional investigation

This section preserves the completed reverse-engineering method and the
remaining conditional analysis. It is not the active morphing-fix plan. The
all-coordinate sensitivity run, temporal filters, layered profiles, and
single-coordinate ablations have already shown that expression filtering is
not a demonstrated fix.

Do not repeat the following sequence unless deterministic stage localization
shows that morphing is already present in rigid-aligned raw driving keypoints.
If that condition is met, use coordinates `34`, `37`, `4`, and `28` as probes
and continue with this bounded sequence:

1. Confirm the existing neutral sensitivity result across the affected avatars;
   do not rerun all 39 coordinates when a smaller probe set answers the stated
   hypothesis.
2. Inspect contact sheets for identity changes and mouth involvement.
3. During real speech inference, correlate each coordinate's temporal velocity,
   acceleration, and jerk with:
   - Per-keypoint motion.
   - Face-radius change.
   - Lipsync-subset radius change.
   - Stitch correction.
4. Identify coordinates that are both visually sensitive and temporally noisy.
5. Start with weak weights, for example `0.1-0.25`, on one coordinate at a time.
6. Run expression-only tests before combining with rotation filtering.
7. Reject any profile that reduces mouth range, creates lip-sync lag, or changes
   identity.

The initial profile should be versioned with:

- Coordinate weights.
- Source sensitivity run.
- Avatars used.
- Perturbation size.
- Audio validation set.
- Date and reviewer.

### 13.1 What expression analysis is trying to reverse engineer

The expression investigation has two different stages that must not be
conflated:

#### Stage 1: sensitivity or controllability

Perturbing one normalized coordinate while holding everything else fixed asks:

```text
If this coordinate changes, what rendered pixels and keypoint geometry can it
change, and by how much?
```

This produces a coordinate-to-visible-effect map. It identifies potentially
dangerous coordinates and coordinates that may be useful control points. It
does **not** show whether those coordinates fluctuate abnormally during real
speech.

#### Stage 2: temporal responsibility

Logging real inference asks:

```text
When the manager-visible morphing occurs, which raw expression coordinates,
driving keypoints, stitch corrections, and head-pose values changed at the same
time?
```

This identifies coordinates associated with actual runtime defects. A
coordinate should become a filter candidate only when it is both:

1. Capable of changing the affected geometry in sensitivity renders.
2. Temporally abnormal or strongly associated with real morphing events.

This is still observational. The final causal test is an expression-only A/B
run where that coordinate receives a weak filter weight and the target defect
improves without harming mouth motion.

The intended evidence chain is:

```text
neutral perturbation sensitivity
        |
        v
coordinate can affect target geometry
        |
        +--- real temporal logs show no abnormal activity ---> do not filter
        |
        v
real temporal logs associate coordinate with defect timestamps
        |
        v
weak one-coordinate expression-only A/B
        |
        +--- morphing unchanged ------------------------------> reject hypothesis
        |
        +--- lip sync / identity regresses -------------------> unsafe coordinate
        |
        v
target morphing improves and motion is preserved
        |
        v
repeat across avatars/audio, then add to reviewed profile
```

### 13.2 How to construct a reviewed expression profile

The required file contract is:

```json
{
  "version": 1,
  "coordinate_weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

There must be exactly 39 values in `[0, 1]`. The value is a correction blend:

- `0.0`: coordinate is unchanged.
- A small value such as `0.1-0.25`: only a weak share of the proposed
  correction is applied.
- `1.0`: full proposed correction is allowed, still subject to the configured
  correction cap. This is not recommended for the first review profile.

Build the first active profile conservatively:

1. Copy the generated review profile rather than editing the only source
   artifact.
2. Keep 38 coordinates at zero.
3. Activate one reviewed coordinate with a weak weight.
4. Record the sensitivity artifact and temporal evidence that justified it.
5. Run expression-only inference, not `both`.
6. Measure target geometry, global expression activity, mouth range, and lip
   sync.
7. Increase weight or add a second coordinate only after the first coordinate
   passes across multiple samples.

Coordinates 34, 37, 4, and 28 are investigation priorities, not a ready-made
profile. Their relatively large lipsync-subset effects make aggressive weights
especially risky.

### 13.3 How the reverse-engineered map will be used later

The final coordinate map can support more than one implementation:

- **Targeted temporal filtering:** stronger high-frequency suppression on
  geometry-sensitive, nonessential coordinates; zero or weak filtering on
  mouth-critical coordinates.
- **Coordinate-specific spike thresholds:** lower thresholds for coordinates
  proven to cause identity/size jumps and higher thresholds for expressive
  coordinates.
- **Region-aware diagnostics:** translate a visible defect timestamp into a
  shortlist of learned coordinates and keypoints rather than inspecting all 39.
- **Training priorities:** if the same coordinates fail across avatars, add
  temporal or identity-consistency losses around the corresponding predicted
  dimensions.
- **Avatar-specific profiles:** only if evidence shows a stable portrait
  dependency that cannot be handled by one conservative global profile. Avoid
  this complexity unless required.

The map must not be used to rename learned coordinates as human anatomy without
external landmark evidence.

### 13.4 Expression artifact interpretation

The sensitivity analyzer writes:

| Artifact | How to use it |
| --- | --- |
| `sensitivity_per_render.csv` | Inspect each avatar, coordinate, and `+/-z` perturbation independently. Asymmetry can reveal nonlinear behavior hidden by averaging. |
| `sensitivity_by_coordinate.csv` | Rank aggregate pixel and geometry sensitivity. Do not use alone to select filter weights. |
| `sensitivity_by_coordinate.json` | Machine-readable version for later analysis scripts or agents. |
| `expression_profile.review.json` | Disabled 39-weight template. It is not an active solution. |
| Coordinate contact sheets | Determine whether changes affect mouth, contour, identity, or unrelated regions. |
| Difference heatmaps | Locate visual impact; verify the coarse dominant-region label. |

The temporal inference JSONL then contributes:

- `motion_prediction.expression_normalized`: raw 39-coordinate values.
- `motion_stabilization.expression`: raw, spike-corrected, final, and
  per-layer corrections when filtering is enabled.
- `keypoint_geometry.driving_raw`: geometry before stitching correction.
- `keypoint_geometry.driving_stitched`: geometry sent to rendering after
  stitching.
- Per-keypoint temporal steps and stitch corrections.

Use raw versus stitched comparisons to distinguish likely causes:

- Defect already present in `driving_raw`: model pose/expression is implicated.
- Defect small in raw but large after stitch: stitching is amplifying it.
- Keypoints stable but rendered pixels unstable: investigate warp, decoder,
  pasteback, matting, or texture flicker rather than motion coordinates.
- Head pose and many keypoints move coherently: likely rigid/perspective motion.
- A small keypoint subset changes while head pose is stable: likely local
  expression geometry.

### 13.5 Proposed temporal correlation analyzer

This analyzer is not implemented yet. A clean first version should consume one
or more current-schema JSONL files and produce evidence, not automatic filter
weights.

For each frame, reconstruct aligned series for:

- Raw 3D head rotation.
- Each of 39 raw normalized expression coordinates.
- First and second temporal differences for each expression coordinate.
- Raw and stitched per-keypoint XYZ positions.
- Whole-face and lipsync-subset radius/extent.
- Per-keypoint stitch correction.
- Audio speaking/listening class.

Recommended analysis logic:

1. Concatenate chunks in frame order, preserving boundary indices and
   timestamps.
2. Calculate coordinate velocity, acceleration, and jerk using the same frame
   rate and boundary handling as the renderer diagnostics.
3. Detect geometry outliers with robust per-run statistics such as median and
   median absolute deviation, while also retaining p95/p99/max values.
4. Mark morphing events separately for raw keypoints and stitched keypoints.
5. Compute coordinate-to-geometry association over a small lag window, for
   example `-2` to `+2` frames. A filter can otherwise appear unrelated if the
   visible renderer response is delayed by a frame.
6. Control for rigid head motion. At minimum, report correlations both before
   and after excluding frames with high angular step/jerk. A stronger version
   can regress geometry metrics on head pose first and analyze residuals.
7. Compare speaking, listening, and silence separately; mouth coordinates are
   expected to move during speech.
8. Cross-reference temporal associations with the neutral sensitivity table.
9. Emit top timestamps and coordinate/keypoint IDs so a reviewer can inspect
   the exact frames rather than trusting a coefficient.
10. Repeat across avatars and retain only associations that generalize or are
    explicitly classified as avatar-specific.

A useful coordinate-priority report should contain at least:

```text
coordinate ID
flattened expression coordinate
learned keypoint and axis
neutral face/lipsync geometry sensitivity
real-run velocity/acceleration/jerk percentiles
association with raw geometry outliers
association with stitched geometry outliers
best temporal lag
speaking versus listening association
mouth-risk/manual-review status
supporting timestamps and avatars
```

Do not multiply these fields into an unexplained single score and call the top
coordinate causal. The output should shortlist hypotheses. The weak
single-coordinate A/B described above is the causal confirmation step.

### 13.6 Morphing root-cause isolation and remediation protocol

This is the active investigation for the manager-reported facial-region size
changes. It supersedes broader expression-coordinate tuning as the primary
morphing direction. The diagnostic capture/analyzer and bounded runtime
controls below are implemented and have been executed on one deterministic
CUDA/TensorRT sample. No morphing fix has been accepted; Section 13.6.8 records
the completed evidence and visual rejection.

#### 13.6.1 Pipeline and stage-localization order

Use one deterministic replay and inspect the pipeline in this order:

```text
captured raw rotation/expression trajectory
        |
        v
raw driving keypoints
        |
        v
stitch correction
        |
        v
final stitched keypoints
        |
        v
warp network
        |
        v
decoder
        |
        v
compositing/matting
```

Classify the first stage where the defect becomes measurable:

- Morphing already present in rigid-aligned raw driving keypoints implicates
  model pose/expression generation.
- Raw geometry stable but stitched geometry unstable implicates the stitch
  network or its correction magnitude.
- Stitched geometry stable but decoded face crops unstable implicates the warp
  network or decoder.
- Decoded crops stable but final frames unstable implicates pasteback, matting,
  crop transforms, or compositing.

Do not compare these stages across independently generated trajectories. Every
branch must pass the raw-motion fingerprint assertion from Section 12.1.

#### 13.6.2 Geometry analysis

Before measuring local deformation, align each 21x3 keypoint shape to the
registered source or a fixed replay reference with a 3D similarity transform.
The alignment must remove translation, rotation, and one uniform global scale.
Measure the remaining non-rigid shape change rather than interpreting rigid
head motion or perspective-driven scale as morphing.

Report separately for the existing lipsync and source-locked learned-keypoint
subsets:

- Pairwise-distance change.
- Aligned width, height, area, and radius change.
- Per-keypoint non-rigid residual.
- Temporal step, acceleration, and jerk.
- p95, p99, maximum, and exact event timestamps.
- Speaking, listening, overlap, and silence distributions.

The subset names describe model wiring, not anatomy. Do not rename learned
keypoints as cheeks, eyes, jaw, or contour without independent landmark or
rendered-region evidence.

#### 13.6.3 Stitch-network control experiment

From the same raw trajectory, obtain the stitch network output once and test:

```text
final = raw + stitch_strength * (stitched - raw)
```

Use exactly these strengths first:

```text
0.00, 0.25, 0.50, 0.75, 1.00
```

Add one separate candidate that applies weak temporal filtering to the stitch
correction before blending. Do not combine this first experiment with rotation
or expression changes beyond the fixed validated speaking configuration.

Record raw, network-produced, blended, and final keypoints; per-keypoint stitch
correction; correction velocity, acceleration, and jerk; rigid-aligned shape
residuals; mouth range; external lip-sync confidence; and timestamps of cap or
filter intervention. Review decoded frames for mouth seams, identity changes,
and warp discontinuities.

A lower stitch strength is viable only if it reduces non-rigid morphing without
exposing mouth seams, weakening articulation, or reducing lip-sync confidence
beyond the acceptance limits below. A correlation between stitch correction
and morphing is not enough; the replayed strength A/B is the causal test.

#### 13.6.4 Post-stitch non-rigid stabilization prototype

Prototype this only if stage localization implicates stitched/final keypoint
geometry. It is one bounded experiment, not a new family of broad sweeps:

1. Start from the existing source-locked keypoint subset and exclude every
   point shown by sensitivity or replay analysis to affect mouth motion.
2. Estimate the per-frame 3D similarity transform from the registered source
   shape to the current stitched stable subset.
3. Remove that transform and compute non-rigid residuals in aligned space.
4. Apply a conservative temporal filter only to high-frequency residuals of
   the reviewed non-lipsync subset.
5. Leave lipsync keypoints untouched in the first implementation.
6. Restore the original per-frame rigid transform.
7. Send the corrected keypoints to the unchanged warp network.

Do not smooth global head rotation, translation, or scale in this layer. Head
pose is handled by the rotation stabilizer; this layer exists only to suppress
local shape changes after stitching.

The prototype must remain disabled by default and expose explicit cutoff,
strength, and maximum-correction options. Log the reviewed keypoint mask,
aligned residuals, raw and corrected keypoints, intervention rate, correction
p50/p95/max, cap hits, and estimated temporal delay. Reject it if it introduces
rigidity, asymmetry, seams, mouth lag, or identity drift.

#### 13.6.5 Renderer-level isolation

If final stitched keypoints are stable while decoded pixels still morph, stop
tuning motion coordinates. Replay identical keypoints through:

1. The existing TensorRT FP16 stitch/warp/decoder path.
2. A higher-precision ONNX/FP32 reference or rebuilt TensorRT reference where
   the model/plugin path makes that feasible.

Capture decoded face crops before pasteback and matting. Track stable rendered
features with local optical flow and compare local scale, shape, texture, and
boundary variation. Then compare the same decoded crops with final composited
frames to separate warp/decoder behavior from matting or pasteback behavior.

If stable keypoints morph in both precision paths, runtime keypoint filtering
cannot reliably solve the defect. Escalate to renderer retraining or replacement
with temporal identity, non-mouth shape-consistency, landmark, and optical-flow
consistency objectives. Precision comparison is a localization test, not an
assumption that FP16 is the cause.

#### 13.6.6 Rejected repeats and optional post-processing

Do not repeat the following without new causal evidence:

- Broad sweeps over all 39 expression coordinates.
- Stronger layered expression filtering.
- Generic whole-face or whole-frame temporal smoothing.
- A/B claims from independently sampled raw trajectories.
- Candidate ranking based only on face-radius or lipsync-radius reduction.

Face-aware optical-flow post-processing may be tested later as a polish layer.
It would track rendered stable regions, apply weak local warps, and leave the
mouth mostly untouched. It is not the primary fix because it adds latency and
can create ghosting, identity drift, or background wobble. Generic FFmpeg
stabilization can correct global frame transforms but cannot correct local
non-rigid facial geometry.

#### 13.6.7 Advancement and stopping rules

Validate on at least three avatars and three speaking samples, plus listening
and silence cases. Advance a runtime morphing candidate only when:

- Target rigid-aligned non-rigid geometry improves by at least `10%` on the
  majority of cases.
- The improvement appears in p95 and in worst-event frame review, rather than
  only in an aggregate mean.
- Mouth range and external lip-sync confidence each regress by no more than
  `5%` relative to the deterministic baseline.
- Identity, seams, texture stability, responsiveness, and intended motion
  remain acceptable in original full-resolution videos.
- The result reproduces from the same captured trajectories and then
  generalizes to additional captures.

Treat `1-3%` metric changes as inconclusive unless they repeat consistently
across the validation set and correspond to visible improvement. Do not accept
lower motion by itself as proof of better naturalness.

Stop runtime morphing-filter work if the stitch-control experiment and one
conservative post-stitch prototype both fail these gates. Record the runtime
architecture as saturated for this defect and recommend renderer/model
retraining or replacement rather than adding more filter combinations.

#### 13.6.8 Completed deterministic stage experiment (2026-07-20)

The protocol above was executed on commit `b53f2c7` with Maria,
`demoday15.mp3`, seed `1234`, `self=2`, `other=1`, and the real auto-selected
TensorRT path. All geometry branches replayed the same captured raw trajectory:

```text
c83a1d3576d9e052a1400de9ca5eb7993bf9bef3a34286f7208dbdec0a61956a
```

The baseline source-locked temporal-step p95 was approximately
`4.95e-8` before stitching and `3.45e-4` after stitching. The first material
measured amplification was therefore the stitch network. The very large
percentage amplification reported by the analyzer must not be quoted as an
effect size because the raw denominator is nearly zero; the stage ordering is
the useful result.

The first fixed candidate matrix produced:

| Candidate | Source-locked step | Acceleration | Jerk | Boundary step | Lipsync step | Final pixel residual | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Post-stitch source-locked | `-49.13%` | `-64.94%` | `-69.34%` | `-55.65%` | `+0.01%` | `+0.83%` | Strong geometry smoothing, no accepted visual morphing fix. |
| Stitch One Euro | `-49.09%` | `-64.91%` | `-69.42%` | `-57.73%` | `-0.29%` | `+1.13%` | Strong geometry smoothing, primarily a jitter result. |
| Stitch strength `0.75` | `-26.25%` | `-28.17%` | `-27.58%` | `-29.88%` | `-0.20%` | `+0.25%` | Conservative attenuation; no established visual fix. |
| Rotation One Euro | `+3.24%` | `-3.79%` | `-3.96%` | `-2.93%` | `-0.28%` | `-4.14%` | Does not address source-locked morphing geometry. |
| Expression One Euro | `+5.63%` | `+0.16%` | `+0.39%` | `+0.14%` | `-4.34%` | `+0.36%` | Reject as a morphing direction. |

A follow-up deterministic strength ablation used `0.00`, `0.25`, and `0.50`
against the same baseline and fingerprint:

| Strength | Source step | Non-rigid residual | Pairwise deformation | Shape range | Lipsync step | Final pixel residual | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `0.00` | `-99.99%` | `-17.06%` | `-33.33%` | `-99.99%` | `-0.27%` | `-2.73%` | Largest numerical morphing reduction, but visually assessed as not good; rejected. |
| `0.25` | `-74.52%` | `-15.90%` | `-28.80%` | `-76.40%` | `-0.24%` | `-2.07%` | Not visually reviewed or accepted. Do not infer quality from interpolation metrics. |
| `0.50` | `-49.61%` | `-12.60%` | `-21.27%` | `-49.17%` | `-0.21%` | `-3.12%` | Best pixel metric and the only remaining bounded compromise worth visual review; not accepted. |

The `stitch_strength=0` result is the central negative finding. It proves that
stitch correction contributes causally to the measured deformation, but it
also proves that removing the learned correction is not a production solution:
the full-resolution video looked worse despite passing the geometry gates.
Visual quality outranks the proxy metrics.

The practical conclusion is:

- Jitter-like source-locked geometry can be reduced with stitch or post-stitch
  temporal controls.
- No tested runtime candidate has fixed the manager-visible facial morphing.
- The learned stitch correction is both part of the problem and necessary for
  acceptable alignment/rendering quality, creating a weight-level tradeoff.
- Do not add more broad filters. If strength `0.50` fails visual review across
  representative samples, move to stitch/warp/decoder retraining, replacement,
  or new weights with identity and non-mouth shape-consistency objectives.

## 14. Production solution direction

The most practical production path is multi-stage rather than relying on one
parameter:

### Phase A: turn-aware guidance

Use the current speaking and listening CFG values with a stable chunk-level
turn-state controller:

- `other=1` while the avatar speaks.
- `other=2` while the avatar listens.
- Hysteresis/interpolation around transitions.

This is low risk because it uses existing runtime inputs and has already shown
modest improvements across multiple portraits.

### Phase B: conservative head stabilization

Validate moderate and light rotation-only One Euro filtering through
deterministic replay, multiple avatars/audios, and all turn states. The current
evidence favors the pure adaptive filter; spike layering and the tested
acceleration/jerk limiters are not justified.

Production acceptance should require:

- Lower head jerk and boundary jumps.
- No obvious lag when a nod begins or ends.
- No frozen or robotic head.
- Low correction magnitude.
- Stable behavior across session-state serialization.
- Negligible hot-path latency with diagnostics disabled.

### Phase C: deterministic morphing-stage isolation

Capture one raw trajectory and locate the first unstable stage using Section
13.6. Expression coordinates `34`, `37`, `4`, and `28` remain diagnostic probes
only. Do not resume broad expression filtering unless replay evidence proves a
specific coordinate causes the target defect.

If stitching is implicated, run the fixed stitch-strength A/B and one weak
stitch-correction filter. If final keypoint geometry is implicated, run one
conservative post-stitch non-rigid stabilizer with lipsync keypoints untouched.
If keypoints remain stable while pixels morph, move to renderer-level isolation.

### Phase D: renderer/model improvement if runtime filtering is insufficient

Runtime stabilization can remove high-frequency artifacts but cannot make a
fundamentally implausible motion trajectory human-natural. If residual problems
remain, the robust long-term solutions are model/training changes such as:

- Temporal velocity, acceleration, or jerk losses.
- Identity and non-mouth geometry consistency losses.
- Better supervision for listening motion.
- Longer effective temporal context.
- Training data with more natural head and upper-body behavior.
- Explicit disentanglement of mouth motion, expression, and rigid head pose.

These are more expensive and require access to training code/data. They should
follow evidence that deterministic stitch control and bounded post-stitch
filtering cannot meet the quality target.

## 15. Post-processing options and their limits

### Conservative whole-frame stabilization

FFmpeg alone can apply transforms, but it does not provide the face-aware logic
needed to decide which motion is intentional. A practical prototype would use
OpenCV or a landmark tracker to:

1. Track stable upper-face landmarks.
2. Estimate translation and small rotation only.
3. Smooth the transform over time.
4. Clamp the correction.
5. Apply the transform to the complete frame.

This can reduce visible global jitter but may introduce moving borders,
background wobble, or robotic motion. It should be considered a fallback or
polish layer after stabilizing motion before rendering.

### Local facial-region filtering

A face-aware post-process could stabilize non-mouth regions more strongly than
the mouth. This requires landmark tracking, masks, and temporal warping or
blending. It is harder to make robust than filtering the model's expression
coordinates and can create ghosting.

### Optical-flow restoration

Optical-flow or video-restoration models may reduce skin shimmer, hair flicker,
and boundary instability. They generally cannot repair incorrect facial
geometry. They add latency, GPU cost, another model dependency, and possible
identity changes. They are not the first production recommendation for the two
reported issues.

## 16. Evaluation protocol

Every candidate should be evaluated with both objective and visual criteria.

### Metric definitions

- **Head step:** geodesic angular distance between consecutive SO(3) head
  rotations, in degrees. This measures frame-to-frame pose movement.
- **Angular acceleration:** change between consecutive angular-velocity
  vectors. This captures abrupt speed or direction changes that a scalar angle
  alone can miss.
- **Angular jerk:** change between consecutive angular-acceleration vectors.
  High jerk is a useful proxy for visibly unnatural starts, stops, and
  reversals.
- **Chunk-boundary jump:** the motion between the last frame of one five-frame
  chunk and the first frame of the next. It is reported separately because a
  filter can look smooth inside each chunk while still discontinuous between
  chunks.
- **Face radius:** mean 2D distance of the selected driving keypoints from their
  frame-wise center. Radius range is normalized by mean radius and reported as
  a percentage.
- **Lipsync-region radius:** the same measurement over keypoints touched by the
  39 predicted expression coordinates. It is a geometry measure, not a direct
  lip-sync score.
- **Expression activity retention:** corrected expression-step p95 divided by
  raw expression-step p95. Values far below `1.0` warn that a filter may be
  removing valid articulation.
- **Stitch correction:** difference between driving keypoints before and after
  the stitching network. It helps distinguish model prediction instability
  from renderer-side correction.
- **Matte coverage swing:** frame-wise change in the proportion of pixels with
  foreground alpha over `0.5`. This detects compositing instability, not motion
  geometry directly.

Percentile metrics summarize a distribution and can hide short severe events.
Always inspect p99, maximum, and the associated timestamps for the final
candidate.

### Head motion

- Angular step p95, p99, and maximum.
- Angular acceleration p95.
- Angular jerk p95.
- Chunk-boundary angular jump.
- Filter intervention rate.
- Correction p95 and maximum.

### Facial geometry

- Whole-face radius and width/height variation.
- Lipsync-keypoint-subset variation.
- Per-keypoint temporal steps.
- Per-keypoint stitch correction.
- Region-specific geometry instead of only global face radius.

### Motion preservation

- Expression activity retention.
- Mouth opening/range preservation.
- Audio/video lip-sync confidence, ideally with an external lip-sync evaluator.
- Listening head-activity range.
- Response to speech starts, stops, interruptions, and overlap.

### Visual review

- Use full-color, full-duration outputs.
- Do not resize or concatenate videos before the primary review.
- Compare synchronized baseline and candidate only after validating the raw
  individual outputs.
- Review at normal speed and frame-by-frame around metric outliers.
- Include multiple reviewers when selecting a production candidate.

An improvement should not be accepted if it only lowers motion magnitude.
Naturalness, responsiveness, identity, and lip sync are hard constraints.

## 17. Known risks and architectural tradeoffs

- The new filters are render-only. This avoids autoregressive distribution
  shift but means future model predictions do not know about corrections shown
  to the user.
- Rotation correction caps take priority over strict acceleration/jerk bounds.
  A large spike may therefore remain partly uncorrected instead of causing a
  visually dangerous transform. Requested and applied corrections are logged.
- Spike and temporal caps currently apply per layer. A layered run produced a
  `0.74961 degree` final correction despite a `0.5 degree` temporal cap, so a
  separate final global cap is required before layered production use.
- One Euro filtering can add apparent motion lag if its cutoff is too low.
- Expression filtering can damage lip sync even when geometry metrics improve.
- A metric that rewards lower lipsync-region variation can accidentally reward
  a frozen mouth. Expression-retention and lip-sync review are mandatory.
- Current expression sensitivity evidence comes from Maria only.
- Custom portrait quality depends on source pose, crop, resolution, lighting,
  occlusion, and how well the LivePortrait registration path handles it.
- TensorRT engine caches are tied to the relevant GPU/runtime compatibility and
  are operationally separate from motion-quality tuning.

## 18. Reproducibility notes

The Colab setup used:

```text
repository: /content/avtr-1
local storage root: /content/avtr1_storage
AVTR1_LOCAL_STORAGE=/content/avtr1_storage
```

Seven restored TensorRT engines were observed:

```text
main/renderer_runtime_artifacts_cc/decoder_b5_fp16.engine
main/renderer_runtime_artifacts_cc/modnet_b5_fp16.engine
main/renderer_runtime_artifacts_cc/stitch_network_b5_fp16.engine
main/renderer_runtime_artifacts_cc/warp_network_b5_fp16.engine
main/speech2motion_runtime_artifacts_cc/avtr1_decode_fp16.engine
main/speech2motion_runtime_artifacts_cc/avtr1_encode_fp16.engine
main/speech2motion_runtime_artifacts_cc/hubert_lbs_fp16.engine
```

The default offline output is 1280x720 at 25 fps. Use `--native-size` for
arbitrary custom portraits. The observed 5-frame processing time was roughly
93-104 ms per chunk in the initial Colab run, which is faster than the 200 ms
of video represented by a chunk. Diagnostic logging adds synchronization and
must not be used to claim production throughput.

Runtime lessons from the deterministic Colab run:

- Do not force `--renderer-backend onnx` for this artifact set. The ONNX warp
  graph uses `GridSample3D`, which the installed ONNX Runtime does not support.
  Use the default `auto` path so the TensorRT renderer engines are selected.
- Exact hashes of GPU-produced avatar registration floats were not stable
  across processes. Commit `4c9953d` fingerprints the stable loaded portrait
  tensor instead while retaining strict audio/config/trajectory checks.
- A tuple of source-locked keypoint IDs was initially interpreted as
  multi-axis PyTorch indexing. Commit `b53f2c7` uses explicit
  `index_select`/`index_copy_`; post-stitch replay then completed successfully.
- Failed captures and stage directories must not be reused as evidence. Start
  a new run root or remove only the explicitly identified incomplete candidate
  before retrying.

## 19. Current verification status

Completed locally for the earlier stabilization implementation:

- CPU behavior tests for disabled identity behavior.
- Isolated spike suppression.
- One Euro smoothing and correction bounds.
- Acceleration and jerk limits.
- Expression coordinate isolation.
- Cross-chunk filter continuity.
- Filter topology state reset.
- Quaternion-derived full-timeline sweep metrics.
- Strict JSON summary handling.
- Diagnostic JSON/per-keypoint smoke test.
- Python compilation, formatting, linting, and whitespace checks.

Still required:

- Run the newly added unit tests and static tooling for deterministic replay,
  stage artifacts/analysis, geometry stabilization, state carry, and turn
  guidance; none were executed while authoring the current source changes.
- Runtime state serialization test with the actual environment.
- Multi-avatar and multi-audio validation.
- External lip-sync and mouth-range validation.
- Production API transition/interrupt/overlap testing.

Completed in Colab on the real TensorRT path:

- Twenty-nine-run rotation, expression, combined, and coordinate-ablation
  experiment on Maria and `demoday15.mp3`.
- Within-trace raw-versus-corrected rotation analysis.
- Correction-cap and approximate lag analysis.
- Experimental nonzero four-coordinate profile and single-coordinate
  ablations.
- Encoded-video optical-flow assessment for repository baseline, prior CFG
  optimum, and moderate One Euro candidate.
- Deterministic raw-motion capture and replay across baseline plus rotation,
  expression, stitch, post-stitch, and combined branches using one verified
  fingerprint.
- Hash-verified geometry and lossless decoded/final pixel stage artifacts, plus
  the rigid-aligned morphing-stage analyzer.
- Stitch strengths `0.00`, `0.25`, `0.50`, `0.75`, and `1.00` on the identical
  trajectory.
- Runtime post-stitch source-locked stabilization after correcting tuple-index
  handling in commit `b53f2c7`.
- Speech/silence automatic turn guidance on 76 chunks. Real listening and
  overlap were not exercised because the listen track was silence.
- Full-resolution visual review of stitch bypass (`strength=0.00`), which was
  judged not good and rejected despite its numerical geometry improvements.

### Production acceptance checklist

A candidate should not be enabled globally until all of the following are true:

- It improves head jerk or visible jitter on the manager-reference samples.
- Improvement repeats across multiple avatars and audios rather than one seed.
- Chunk boundaries do not introduce visible snaps.
- Intentional nods and sustained head turns remain present.
- Mouth range and external lip-sync confidence do not regress materially.
- Listening motion remains responsive rather than frozen.
- Face-region morphing improves independently of any output resize or crop.
- A morphing candidate meets the Section 13.6 geometry, worst-event, and
  `10%` improvement/`5%` preservation gates.
- Corrections rarely reach their configured maximum.
- Filter state survives API serialization without a boundary discontinuity.
- Speaking/listening CFG transitions are stable under interruption and overlap.
- Latency remains inside the 200 ms budget per five-frame chunk with production
  diagnostics disabled.
- A rollback path exists and the feature remains disabled by default until the
  validation set passes.

## 20. Recommended next decisions

1. Do not repeat the completed Maria/`demoday15.mp3` deterministic candidate
   matrix or another broad expression/filter sweep.
2. Review the already-generated `stitch_strength=0.50` video only as a final
   bounded compromise. If it does not show an obvious visual morphing gain
   without alignment, expression, identity, or seam regressions, reject runtime
   stitch attenuation as a production direction.
3. If a runtime compromise survives visual review, validate it on at least
   three avatars and three speaking audios before changing any defaults. Require
   original-resolution review and external lip-sync/mouth-range evidence.
4. Otherwise stop runtime morphing-filter work. Scope new stitch/warp/decoder
   weights or retraining with temporal identity and non-mouth shape-consistency
   objectives. The current evidence shows a learned-weight tradeoff, not a
   missing temporal-filter combination.
5. Keep rotation One Euro separate as a head-jitter candidate. Validate moderate
   and light settings across multiple captured trajectories and intentional
   head-turn samples; do not present it as a morphing fix.
6. Validate turn-aware `cfg_other_audio` with a real listen track covering
   listening, interruption, overlap, and silence, then verify serialized API
   state across turn boundaries. Keep it disabled by default.
7. Run the new unit tests, lint, type checks, and state-codec tests in the Linux
   renderer environment. Runtime execution has succeeded, but the new automated
   suite is still unexecuted.
8. Preserve the current default renderer behavior and rollback path. No
   geometry or turn-guidance candidate is production-approved.

The current evidence supports a substantial measured reduction in global head
jitter on one speaking sample, not completion. The moderate One Euro candidate
is the strongest clean setting; the light preset is the conservative fallback.
The face-morphing complaint remains open. Stage localization implicates the
stitch correction, but complete bypass was visually rejected and no runtime
candidate has passed the visual production gate.

### Required handoff update after each experiment

The next agent should append a result using this structure:

```text
Experiment ID and date:
Commit and environment:
Avatar/audio/turn segmentation:
Baseline exact configuration:
Candidate exact configuration:
Artifacts and paths:
Raw-motion capture path and fingerprint:
Source registration/model artifact identifiers:
Stage-count/data-integrity check:
First stage where the target defect is measurable:
Absolute baseline metrics:
Absolute candidate metrics:
Relative changes:
Rigid-aligned non-rigid geometry metrics:
Intervention rate and correction distribution:
Correction-cap saturation:
Motion-preservation/lip-sync checks:
Decoded-crop versus final-composite assessment:
Normal-speed visual assessment:
Worst-frame/boundary assessment:
Conclusion: accepted / rejected / needs more evidence
Accepted hypothesis and supporting evidence:
Rejected hypotheses and evidence:
Precisely authorized next experiment:
```

Never replace prior evidence with only the newest aggregate score. Preserve the
experiment history so later agents can distinguish a real improvement from a
different input, output-size mode, seed, or metric implementation.
