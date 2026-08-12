# Avatar quality assessment

This document separates measurements collected from the deployed avatar worker
from the fuller study needed to make claims about perceived avatar quality.
Latency alone cannot establish whether an avatar looks natural, retains a
person's likeness, or moves its mouth in time with speech.

## What was measured

The current automated check exercises the same binary `PHS1` stream used by
the web client. It sends one fixed-seed request at a time to a warmed, deployed
FastAPI worker, decodes the returned audio and JPEG frames, and evaluates the
rendered output. It does not retain generated audio or video.

The evaluation currently measures the following diagnostics.

| Diagnostic | What it checks | Interpretation |
| --- | --- | --- |
| Face-landmark detection rate | Whether MediaPipe finds a face and mouth in generated frames | A low value indicates a severe rendering failure. A value near 1.0 is necessary, not sufficient, for quality. |
| Audio-to-mouth timing proxy | Cross-correlation of generated-audio RMS with detected mouth aperture, with the best frame offset | A coarse timing diagnostic. It is not phoneme-level lip-sync accuracy. |
| Lower-face continuity | Mouth-centre displacement and high-frequency aperture variation | Higher values can indicate jitter or unstable lower-face placement, but legitimate speech motion also contributes. |
| Pause stability | Mouth-patch motion only during sustained low-energy audio regions | Helps reveal unwanted movement during a genuine pause. Short word boundaries are excluded. |
| Mouth-detail retention | Laplacian variance in a scale-normalized mouth patch, compared descriptively with the source clip at the same frame edge | A relative sharpness signal only. It is not a percent-realism or teeth-quality score. |

The implementation is in [`voice_cloning/src/evaluate_musetalk.py`](../voice_cloning/src/evaluate_musetalk.py).

## Controlled deployment result

The following is a renderer comparison, not a human-subject study. It was run
on the deployment worker against an isolated `Test` avatar profile. The worker
was warmed for the selected profile before the requests. Chatterbox generated
the speech and MuseTalk rendered the frames. The assistant/LLM stage was not
used, so provider-network time cannot affect these results.

Configuration:

- 3 articulation-oriented prompts, each rendered 3 times sequentially
- 9 output streams per coordinate track
- fixed seed `1234`, `realistic` preset, 25 FPS, and a 768-pixel maximum frame edge
- direct loopback connection to the deployed worker
- no output media retained

The repeated runs are primarily a deterministic-pipeline check. With a fixed
seed, the visual diagnostic values reproduced for the same prompt. The ranges
below therefore mainly reflect the three different prompts, not independent
sampling of the model's stochastic variation.

| Metric | Production `legacy` face track | Alternate `baked` face track | Direction / caveat |
| --- | ---: | ---: | --- |
| Valid face landmarks | 100% of frames | 100% of frames | Higher is better, but this only confirms detectable faces. |
| Audio-mouth correlation, median | 0.379 | 0.263 | Higher is usually better for this envelope proxy. |
| Best audio-mouth offset, median | -40 ms | +200 ms | Near zero is desirable. The estimate is limited to 25-FPS frame resolution and does not prove phoneme alignment. |
| Best audio-mouth offset, 95th percentile | -40 ms | +280 ms | A large positive delay is unfavorable. |
| Mouth-centre jitter, p95 | 0.311 | 0.173 | Lower is smoother, in normalized landmark units. |
| High-frequency aperture variation, median | 0.0473 | 0.0192 | Lower is smoother, but excessive smoothing can reduce articulation. |
| Pause mouth-motion, p95 | 0.0155 | 0.0096 | Lower is better for sustained low-energy regions. |
| Voiced mouth-motion, p95 | 0.0320 | 0.0153 | This must be read together with sync. Low motion is not automatically better. |
| Mouth-detail diagnostic, median | 21.52 | 31.19 | Higher is sharper under this diagnostic only. |
| Source mouth-detail diagnostic | 36.85 | 36.85 | Same 60-frame browser-recorded source sample at the same 768-pixel cap. |
| First streamed media, median | 1.75 s | 1.67 s | Included for context, not a visual-quality metric. |

The `legacy` track is the current public default. It remains the better choice
for the active application because it has the better timing proxy and a
near-zero median offset. The `baked` track improves static detail and
continuity diagnostics, but it falls behind the generated audio by 2–7 frames
in this test and its timing correlation is less reliable. It should not replace
the production path without a lip-sync improvement.

The source/detail numbers should not be read as “58%” or “85% realistic.” The
source and generated clips have different speech content, compression, pose,
and lighting. The result only supports the narrow conclusion that the current
generated mouth patch contains less high-frequency detail than this test
source, and that the baked track retains more of that diagnostic than legacy.

## What this does and does not establish

The controlled result establishes that the current production renderer streams
valid faces and that the legacy-versus-baked tradeoff is measurable. It also
gives regression baselines for future compositor changes.

It does not establish that PixelHolo matches human lip movements, that the
generated face looks indistinguishable from the person, that Chatterbox sounds
like the source speaker, or that the output is preferred by users. Those are
different questions and require ground truth or human judgment. The current
metrics are intentionally treated as engineering diagnostics, not a MOS,
identity, or commercial-product comparison.

## Rigorous quality study to run next

### Dataset and protocol

Use consented participants only. For each participant, collect:

1. A 25-second front-lit profile clip used to create the avatar.
2. A separate held-out recording of the same person speaking 8–12 short,
   prewritten evaluation sentences. The held-out sentences must not occur in
   the profile clip.
3. A separate target audio recording of those sentences, or an exact text
   target rendered through the selected TTS system, depending on whether the
   study evaluates audio-driven animation or the complete cloned avatar.

Use at least 12–20 speakers with varied lighting, skin tones, facial hair,
glasses, and camera quality. Keep the evaluation clips in a controlled
front-facing subset for primary scores. Report a separate robustness split for
less favorable lighting and pose rather than mixing both conditions.

For every source profile, render each held-out sentence with the production
configuration and planned ablations. Randomize presentation order. Do not let
raters know which system or setting produced a clip.

### Objective metrics

| Quality question | Metric | Experimental use |
| --- | --- | --- |
| Are lips synchronized to the audible speech? | SyncNet LSE-D and LSE-C on the final muxed audio-video output | Primary lip-sync metric. Report mean, median, and 95% confidence intervals. Lower LSE-D and higher LSE-C are generally better. |
| Does the mouth follow the correct spoken content? | Forced-alignment or viseme/phoneme timing error on a labeled subset | Compare visible-viseme timing with the known generated or recorded phoneme sequence. Use this to validate the coarse RMS/aperture proxy. |
| Does the face keep the source identity? | Face-embedding cosine similarity and temporal embedding variance | Compare frames to held-out real video frames after face alignment. Report identity retention separately from temporal stability. |
| Does the voice retain speaker identity? | ECAPA-TDNN or another calibrated speaker-verification cosine score | Compare generated speech with enrollment speech not reused as the reference condition. |
| Is the spoken content preserved? | ASR word error rate against the target text | Run a fixed ASR model on generated audio. This detects pronunciation or dropped-word failures that visual metrics miss. |
| Is the lower face temporally stable? | Optical-flow-warped temporal error and landmark jitter outside expected mouth motion | Measure chin/jaw stability separately from moving lips. This directly targets the previously observed lower-face double-layer issue. |
| Is visual detail retained? | Mouth-region LPIPS or DISTS against matched held-out recordings, plus no-reference sharpness as a secondary diagnostic | Use paired metrics only when pose, sentence, camera, and lighting are matched. Do not use a no-reference sharpness score as a realism claim. |

For each metric, report the number of profiles, prompts, frames, and failed
clips. Report bootstrap 95% confidence intervals across speakers, not merely
across frames, because frames from one person are not independent samples.

### Human evaluation

Automated metrics should be complemented by a blinded study. A practical first
study is 20 or more raters, each viewing a balanced random subset of clips and
scoring five questions on a 1–5 scale:

1. Lip-sync accuracy
2. Naturalness of mouth, teeth, and chin
3. Temporal stability, including pauses
4. Facial likeness to the reference person
5. Overall conversational-avatar quality

Add paired preference trials for proposed renderer changes. For every pair,
ask which clip has more natural lip motion and which more closely resembles the
reference. Randomize left/right order and include attention checks. Report the
mean opinion score with confidence intervals, pairwise win rate, rater count,
and inter-rater agreement.

### Ablations and acceptance criteria

At minimum, compare the current production configuration against each of the
following one variable at a time:

- `legacy` versus `baked` coordinate tracks
- detail-sharpen strength
- mouth and chin compositing boundaries
- temporal smoothing strength
- 768 versus 1280 maximum frame edge

Hold the TTS output, seed, source profile, and prompt set fixed for renderer
ablations. A change should become the default only if it improves the primary
lip-sync score or has no statistically meaningful degradation in it, while
also improving the relevant human rating. The current controlled result is an
example: baked coordinates improve the detail and stability diagnostics, but
their timing regression means they fail that criterion.

## Reproducing the engineering diagnostics

Run the evaluator on a GPU worker with a test workspace and an explicitly
chosen profile. Do not use another person's production profile for evaluation.

```bash
cd voice_cloning
PYTHONPATH="$PWD" .venv/bin/python src/evaluate_musetalk.py \
  --api-base http://127.0.0.1:8000 \
  --workspace-id <test-workspace-id> \
  --profile <test-profile> \
  --presets realistic \
  --coord-sources legacy,baked \
  --avatar-max-frame-edge 768 \
  --repeats 3 \
  --no-save-media \
  --output-dir outputs/avatar_quality_eval
```

The command writes machine-readable JSON and CSV summaries. It is intended for
regression checks and renderer comparisons. Add SyncNet, speaker-verification,
and human-study outputs before making a broader quality claim.
