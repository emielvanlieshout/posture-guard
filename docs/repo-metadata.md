# Repository metadata

The strings GitHub keeps outside the codebase, written down so they survive and
stay consistent with `pyproject.toml`.

## About — description

Paste into the repository's **About** box (GitHub allows 350 characters):

> Webcam posture monitor that targets shoulder protraction, not forward head
> posture. Calibrates against your own two postures, dims the screen while you
> slouch and clears the moment you sit up, and tightens the target every week.
> Runs entirely on-device.

248 characters. Four claims, in the order a stranger needs them:

1. **what it is** — a webcam posture monitor, so the category is clear in five words;
2. **what is different** — protraction rather than forward head posture. This is
   the whole reason the project exists: every comparable tool measures the
   ear-shoulder-torso angle. Anyone searching for the shoulder problem
   specifically will recognise it here and nowhere else;
3. **what it does to you** — calibrated to your own postures, dims while you
   slouch, clears when you sit up. The asymmetry is the mechanism, and it fits in
   a clause;
4. **the objection** — a camera watching you all day. "Runs entirely on-device"
   answers it before it is asked.

Deliberately left out: macOS (the badge and the README cover it, and the pipeline
is cross-platform), MediaPipe (an implementation detail nobody searches for when
looking for a posture tool), and the weekly ratchet's numbers.

### Shorter variant

If the box ever needs to be tighter:

> Webcam posture monitor for shoulder protraction, not forward head posture.
> Calibrated to you, escalating screen dim, fully on-device.

## Topics

```
posture  posture-corrector  ergonomics  rsi  mediapipe  pose-estimation
computer-vision  webcam  macos  menubar-app  privacy  on-device
```

Three groups, because people arrive from three directions. `posture`,
`posture-corrector`, `ergonomics` and `rsi` catch the ones with the problem.
`mediapipe`, `pose-estimation` and `computer-vision` catch the ones looking for
the technique. `macos`, `menubar-app`, `webcam`, `privacy` and `on-device` catch
the ones filtering on how it runs.

## Website

Leave empty. There is no project site, and pointing it at the README duplicates
the link GitHub already shows.

## Social preview

None yet. A worthwhile one would be the side-view diagram: ear, acromion and the
horizontal offset between them, over the sentence about perspective hiding
protraction from a frontal camera.
