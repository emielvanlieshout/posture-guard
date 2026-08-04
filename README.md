# posture-guard

A local webcam posture monitor built around **shoulder protraction** — shoulders
rolling forward — rather than forward head posture.

Everything runs on your machine. The camera feed is turned into a handful of
numbers inside the process and the image is discarded. Nothing is uploaded, and
the test suite enforces that.

---

## Why this exists

Every posture app I could find measures roughly the same thing: the angle
between ear, shoulder and torso, i.e. how far your head juts forward. Posturr,
SitApp and Slouch Sniper all do. That is a real problem, but it is a different
one from shoulders rolling forward, and none of them treat protraction as its
own signal.

The obvious way to add it seemed easy: shoulders rotating forward should shrink
the distance between them as seen head-on, while your head stays the same size.
Measure the ratio, alarm when it drops.

**That does not work at desk distance,** and finding out why shaped the rest of
this project.

Rotating your shoulders forward shrinks their projected width by `cos(θ)` — but
it also brings them *closer to the lens*, which magnifies them. Projected width
goes as `2a·cos(θ) / (D − a·sin(θ))`. The cosine term is second order in θ; the
magnification is first order. At small angles magnification wins, so the
shoulders appear to get *wider*. Narrowing only takes over past
`tan(θ) > a/D` — about **17°** at a 62 cm laptop distance.

Measured on the synthetic model in `posture_guard/synth.py`, with the head held
still so only the shoulders move:

| protraction | shoulder ÷ eye width (62 cm) | same, camera at 2.5 m |
|------------:|-----------------------------:|----------------------:|
| 0°          | 3.71                         | 4.10                  |
| 10°         | 3.86                         | 4.09                  |
| 20°         | **3.90** (peak)              | 3.95                  |
| 30°         | 3.80                         | 3.69                  |
| 35°         | 3.69                         | —                     |

Neutral and 35° of slouch land within 3% of each other. The signal is not weak,
it is *non-monotonic*: an alarm built on it would fire in the middle of the
range and go quiet at both ends. `tests/test_features.py` pins this down so it
cannot quietly get "fixed" back.

From the side, the same model gives a clean, near-linear measurement — the
horizontal offset between ear and acromion, which is also the plane clinicians
use for the sagittal shoulder angle:

| protraction | 0° | 5° | 10° | 15° | 20° | 25° | 30° |
|---|---:|---:|---:|---:|---:|---:|---:|
| shoulder ahead of ear (÷ face height) | 0.00 | 0.09 | 0.17 | 0.26 | 0.34 | 0.41 | 0.48 |

So posture-guard supports both, defaults to the side, and **tells you in numbers
whether your setup can actually see your slouch** instead of leaving you to find
out after three weeks.

### A laptop webcam is not a dead end

Width is the wrong thing to look at, but it is not the only thing. Protraction
does not merely roll the shoulders forward, it rides them *up* toward the ears,
and that is plainly visible head-on. Same model, head held perfectly still so
every bit of signal has to come from the shoulders:

| protraction | 0° | 5° | 10° | 15° | 20° | 25° | 30° |
|---|---:|---:|---:|---:|---:|---:|---:|
| ear-to-shoulder drop (÷ face height) | 2.19 | 2.15 | 2.11 | 2.07 | 2.03 | 1.98 | 1.94 |
| shoulder ÷ eye width | 3.71 | 3.80 | 3.86 | 3.89 | **3.90** | 3.86 | 3.80 |

The drop falls monotonically the whole way; the width peaks in the middle and
comes back. Calibrated on nothing but shoulder movement, that one feature takes
71% of the weight on its own and the resulting score is monotonic from 0 to 26
degrees.

So a frontal setup does watch your shoulders. It reads their *height* rather
than their width, and it cannot separate that from the head sinking — which is
what the rest of the feature set and the disagreement term are for.

### And why your hip has to be in the shot

The ear turns out to be a treacherous reference. Forward head posture moves the
ear the same way protraction moves the shoulder, so any measurement taken
between the two conflates them. Same model, `shoulder_ahead` at 30° of
protraction:

| head position | shoulder_ahead |
|---|---:|
| head stays put | **0.99** |
| head drifts forward with the shoulders | 0.11 |
| head runs further forward than the shoulders | **−0.80** |

The sign inverts. The feature ends up reporting "shoulders back" about someone
whose shoulders came forward.

Anything referenced to the pelvis is immune, because your pelvis does not move
when you crane your neck. `trunk_incline` responds to protraction and ignores
head travel completely; `head_over_hip` does the reverse. With both in play the
two motions are separable — which is why the side view wants your hip in frame,
and why calibration says so when it is missing.

---

## Install

macOS, Python 3.10 or newer.

```bash
git clone https://github.com/emielvanlieshout/posture-guard.git
cd posture-guard
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[macos]"

posture-guard setup      # creates the data directory, downloads the pose model
posture-guard doctor     # checks camera, permissions, dependencies
```

`setup` makes the only network request this project ever makes: it fetches
MediaPipe's ~6 MB pose model. After that you can run it offline forever.

On Linux or Windows the pipeline works and the console alerter works; the dim
overlay and the menu bar are macOS-only.

> **Camera permission on macOS** is granted to the app that *launches* Python —
> your terminal or IDE, not posture-guard, which macOS does not consider a thing
> that exists. `doctor` reports the authorisation status outright and raises the
> system prompt the first time. If it says `denied`, switch your terminal on
> under System Settings → Privacy & Security → Camera and then **quit that
> terminal completely (Cmd-Q) and reopen it** — a permission change does not
> reach a process that is already running.

> **`AVCaptureDeviceTypeExternal is deprecated for Continuity Cameras`** is
> macOS talking to itself, not a fault. It appears once when `doctor` lists your
> cameras. The deprecated call is kept on purpose: its ordering is the one
> OpenCV indexes by, and a correct index matters more than a tidy log line.

> **`incompatible architecture (have 'arm64', need 'x86_64')`** means this
> Python is running translated by Rosetta while its packages are compiled for
> Apple Silicon. Nothing is corrupt. `posture-guard doctor` reports the
> architecture on its third line and says what to do; `arch` in a shell tells
> you what that shell is. A universal Python inherits its architecture from
> whatever started it, which is why the same virtualenv can work from one
> terminal and fail from another — and why `install-app` pins the bundle to the
> architecture that was working when you ran it.

> **`CERTIFICATE_VERIFY_FAILED` during setup** means this Python has no CA
> bundle — normal on a fresh python.org, pyenv or conda install, which ignore
> the macOS keychain. `setup` uses `certifi` to avoid it, so
> `pip install --upgrade certifi` fixes it. Failing that, the error message
> prints a `curl` command that fetches the model using the system trust store
> instead.

Paste the commands without the trailing `# comments` — zsh does not treat `#`
as a comment in an interactive shell, so they arrive as arguments.

---

## Setting up the camera

**Side view (recommended).** Put a camera to your left or right, roughly at
shoulder height, far enough back to see your head, shoulder **and hip** in
profile. That last one matters — see above; without it, tucking your chin would
satisfy the profile while your shoulders stay exactly where they were, and
calibration will warn you. An old phone on a stand works; on macOS an iPhone via
Continuity Camera shows up as an ordinary camera. Use `posture-guard doctor` to
find its index and `posture-guard config --set camera_index=1` to select it.

**Frontal view.** Your built-in webcam, whole head and both shoulders in frame.
No second camera, no stand, and it is what most people will actually use:

```bash
posture-guard config --set view=frontal
posture-guard calibrate
```

It tracks the slouch complex — shoulders riding up, chin dropping, head drifting
forward — which travels with protraction without being the same thing.
Calibration says so explicitly rather than letting you assume otherwise.

Make the two poses differ as much as you can *in the shoulders*: back and down
hard for the first, rolled forward and up for the second. Weights follow
whatever separates your two poses, so the more of that difference is shoulders,
the more of the score is about shoulders.

It is not gameable in either obvious direction. Against the synthetic model, a
profile calibrated this way scores 1.50 for slouched shoulders with the chin
deliberately tucked, and 0.60 — over the alert threshold — for good shoulders
with the head craning forward.

Check the framing before you calibrate:

```bash
posture-guard preview
```

This draws the landmarks and the live feature values. It is the only command
that puts the camera image on screen; monitoring never does.

---

## Calibrating

```bash
posture-guard calibrate            # opens a window on macOS
posture-guard calibrate --terminal # countdown in the shell instead
```

On macOS this opens a window with the live camera in it, which matters more
than it sounds. Posing at a camera you cannot see means finding out twelve
seconds later that every frame was rejected; here the frame in front of you is
judged as you sit, in plain words — *no-one in view*, *face the camera
squarely*, *this camera is looking at your front* — and usable frames are
counted as they arrive. A badly placed camera becomes something you fix during
the countdown. The menu bar has a **Calibrate…** item that opens the same
window.

You hold two poses for 12 seconds each: the posture you want, then the slouch
you actually sit in. Everything is anchored on those two, so **0 means you are
sitting the way you demonstrated and 1 means you are back where you started**.
No assumptions about your build, your desk or the lens.

Feature weights are not hand-tuned. Each feature is scored on how far apart it
puts your two postures relative to its own noise — essentially *d′* — and
anything that fails to separate them gets weight zero. That is what lets one
codebase serve both camera placements without either needing magic numbers.

You will get a table like this, and it is worth reading:

```
feature                   good    slouch  separation   weight
trunk_incline            0.866     9.467       21.94     0.18
head_over_hip            0.039     0.452       18.02     0.18
ear_shoulder_hip       177.588   150.596       15.71     0.18
shoulder_ahead           0.038     0.426       17.31     0.09
neck_incline             1.541    19.951       11.02     0.09
head_pitch               5.961    -2.313        6.14     0.18
face_over_neck           0.717     0.798        3.31     0.09
```

Separation is not the only thing deciding those weights. It measures how well a
feature splits your two poses; it cannot tell whether the feature means what its
name says. The ear-referenced ones are halved on principle, because the geometry
above says they are ambiguous — and normalisation hands them the full weight back
anyway when the hips are out of frame and they are all there is.

Recalibrate about once a month. Your "good" posture should itself improve, and
the app reminds you after 30 days.

### Postures you never demonstrated

Averaging features places you somewhere on the line between your two calibrated
poses. That is the right answer for postures on that line, and a misleading one
for postures off it.

Craning at the screen with your shoulders back is the case that matters. Measured
against the ear, the shoulder now sits *behind* where it does in your good
posture, so `shoulder_ahead` reports better than perfect while `head_over_hip`
reports fully slouched. Averaged, they cancel: the first version of this scored
that posture **+0.12**, essentially "good".

What gives it away is not the average but the argument. Features that agree
within a few percent for every on-axis posture are suddenly a full scale apart,
so the spread across features is measured too, and anything beyond what
calibration saw is added to the score:

| posture | axis | disagreement | final |
|---|---:|---:|---:|
| your good pose | 0.01 | 0.05 | 0.01 |
| on-axis slouch, halfway | 0.56 | 0.06 | 0.56 |
| your slouch | 1.00 | 0.05 | 1.00 |
| **shoulders good, head 7 cm forward** | 0.12 | 0.71 | **0.84** |
| **shoulders slouched, chin tucked** | 0.79 | 0.83 | **1.43** |

On-axis postures are untouched — the ladder stays exactly as smooth as it was.
The tolerance comes from your own calibration frames rather than a constant, so
someone whose landmarks are noisy gets a correspondingly higher bar.

So yes, it watches your head. Not because head position was the goal, but
because a posture nobody demonstrated should not be assumed to be a good one.

---

## Running

```bash
posture-guard run          # menu bar app
posture-guard run --headless
```

The menu bar title is a single bar glyph that grows with your score, so you can
read your state without opening anything.

**What an alert feels like.** Cross the threshold and nothing happens for 8
seconds — reaching for a mug or turning to talk to someone never triggers it.
Stay there and the screen begins to dim, slowly, over 25 seconds, capped at 55%
so it stays perfectly usable. Sit up and it is gone in half a second.

That asymmetry is the whole mechanism: correcting pays out immediately, ignoring
it costs you gradually. The thresholds are just bookkeeping.

Three ways to call it off, because a screen you cannot un-dim is worse than bad
posture:

```bash
posture-guard pause 30     # works from any terminal, even mid-alert
posture-guard resume
```

plus the menu bar (the overlay ignores mouse events, so everything underneath
stays clickable), and quitting. If the capture thread hangs or dies, the main
thread notices the stale state within three seconds and lifts the dim by itself.

---

## The part that makes it stick

A fixed threshold stops working the moment you clear it. You sit at 86% good for
weeks and nothing more happens.

So the target moves. Once a week posture-guard looks at the past seven days:

- **≥ 85% of measured time in good posture** → the threshold tightens by 8%. A
  posture that passed last week now registers as a slouch.
- **≤ 55%** → it loosens by the same step. An alarm you cannot satisfy is one
  you learn to ignore.
- in between → it holds.

It needs a full week and at least 10 hours of measured time before it will move
at all, and it stops at 0.25 (where the score is mostly measurement noise) and
0.75 (where it stops asking anything of you). Every change is logged.

A realistic first two months:

| | what to do | what to expect |
|---|---|---|
| **Week 1** | `config --set alerters=console` and just let it watch | your honest baseline; most people are surprised |
| **Week 2** | switch the dim on, leave the threshold alone | frequent alerts, short good stretches |
| **Weeks 3–5** | nothing; let the ratchet work | good% climbs, first tightening around week 3 |
| **Week 5** | recalibrate | your "good" pose is better than it was; the scale resets |
| **Weeks 6–8** | nothing | mean score falls while the threshold keeps dropping |

Watch the **mean score**, not the daily percentage. Good posture holding steady
while the threshold keeps tightening is progress even when the percentage looks
flat. The percentage is measured against a moving target; the mean score is
anchored on your calibration.

```bash
posture-guard report                     # terminal
posture-guard report --html --open       # self-contained HTML, no external assets
```

One honest caveat: this trains awareness, and awareness is the part a camera can
help with. If protraction comes from short pecs or a weak mid-back, a dimming
screen will not fix that on its own.

---

## Making it a real app

Run from a terminal, this is not an application as far as macOS is concerned. It
has no identity, so the camera permission is granted to Terminal and listed
under that name; it has no Info.plist, so the prompt cannot say why it wants the
camera; it takes a Dock icon it has no use for; and it dies with the window you
started it in.

```bash
posture-guard install-app --login-item
```

That writes `~/Applications/PostureGuard.app` — a launcher plus an Info.plist,
not a vendored copy of Python, so the app and the `posture-guard` command remain
the same program. It runs in the menu bar with no Dock icon, asks for the camera
in its own name with a sentence explaining why, declares Continuity Camera
support (which is what that deprecation warning in the logs is asking for), and
logs to `~/Library/Logs/posture-guard.log`.

Open it once from Finder so macOS can ask for the camera. Rebuild it after
upgrading with the same command.

**When it appears not to start.** It has no window and no Dock icon by design,
and the menu bar item is one character wide — on a laptop with a notch and a
dozen other menu bar items it can be genuinely invisible. So the app posts a
notification when it comes up, and every startup step goes to the log:

```bash
posture-guard app-log            # what it wrote
posture-guard app-log -f         # keep watching
```

```
[09:14:02] posture-guard 0.1.0 starting  pid=8412  from the app bundle
[09:14:02] config: view=frontal camera=0 alerters=['dim']
[09:14:02] calibration: frontal view, 0 days old
[09:14:02] camera permission: authorized
[09:14:03] opening camera 0
[09:14:04] camera open, waiting for a pose
[09:14:05] first pose detected, score 0.31
[09:14:05] starting the menu bar
```

The last line is how far it got. `posture-guard run --debug` adds every frame's
score or rejection reason, which is more than you want in normal use and exactly
what you want when it is misbehaving.

Unsigned, so it is fine for your own machine and Gatekeeper will stop anyone
else opening it without a right-click → Open. Signing it properly needs a paid
Apple developer account.

## Keeping it running headless

If you would rather not have an app bundle, a launch agent works too. Drop this
in `~/Library/LaunchAgents/com.posture-guard.plist`
(adjust the paths) and run `launchctl load ~/Library/LaunchAgents/com.posture-guard.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.posture-guard</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/YOU/posture-guard/.venv/bin/posture-guard</string>
    <string>run</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict></plist>
```

A copy lives in `scripts/com.posture-guard.plist`.

---

## Settings

```bash
posture-guard config                       # show everything
posture-guard config --set dwell_s=15
posture-guard config --set max_dim=0.4
posture-guard config --set quiet_start=18:00 --set quiet_end=08:30
posture-guard config --set alerters=dim,notify
```

Worth knowing:

| setting | default | what it does |
|---|---|---|
| `view` | `side` | which camera placement the profile is for |
| `camera_index` | `0` | which camera; `doctor` lists them |
| `fps` | `6` | posture moves slowly; this keeps a core mostly idle |
| `enter` / `exit` | `0.55` / `0.35` | alert on and off thresholds (hysteresis) |
| `dwell_s` | `8` | how long a slouch must last before anything happens |
| `ramp_s` | `25` | how slowly the dim escalates |
| `release_s` | `0.5` | how fast it clears when you sit up |
| `max_dim` | `0.55` | opacity ceiling; values above 0.85 are refused |
| `alerters` | `dim` | any of `dim`, `notify`, `console` |
| `ratchet_enabled` | `true` | the weekly threshold progression |

Everything lives in `~/Library/Application Support/posture-guard/`: config,
profile, model and history. Uninstalling is deleting that directory.

---

## Privacy

- No frame is written to disk, kept in a buffer, or passed outside `capture.py`.
  `preview` is the single exception and it only draws to a window.
- The database stores a score, seconds above and below threshold, and event
  markers. `tests/test_privacy.py` asserts the schema holds nothing else.
- No runtime module may import `urllib`, `http`, `requests` or `socket`. A test
  parses every source file and fails if one does. Only `model.py` is exempt, and
  it runs during `setup`.
- The whole pipeline is exercised in a test with every socket monkeypatched to
  raise, so a network call would fail the suite rather than succeed quietly.
- The HTML report is self-contained: no CDN, no fonts, no script tags.

---

## Development

```bash
pip install -e ".[dev]"
pytest -q                            # 343 tests, no camera needed
posture-guard selftest               # end-to-end on synthetic data
```

`posture_guard/synth.py` is a parametric torso: shoulders are two points on a
circle around the spine, protraction rotates them forward, and the landmarks
come out of an ordinary perspective projection. That is what lets the tests
assert real properties — monotonicity, invariance to camera distance, that a
turned torso is rejected rather than mismeasured — without a webcam.

`selftest` runs the whole chain on that model and prints a verdict. If it passes
but the real thing misbehaves, the problem is the camera, a permission or the
calibration, not the pipeline.

```
[PASS] calibration: 90+90 frames, best separation 21.9 noise widths
[PASS] weight sits on protraction itself: 18% of the weight is on trunk_incline
[PASS] a tucked chin does not excuse forward shoulders: scores +1.43 (slouch is ~1.0)
[PASS] forward head posture is not scored as good posture: scores +0.84 (good is ~0.0)
[PASS] score anchors: good posture scores -0.01 (want ~0), slouch +1.00 (want ~1)
[PASS] monotonic in protraction: 0deg=-0.09, 10deg=+0.34, 20deg=+0.76, 30deg=+1.14
[PASS] distance invariance: score moves 0.02 across a +-25% change in camera distance
[PASS] alert clears on correction: cleared 0.8s after sitting up
```

---

## Limitations

- The synthetic model validates the geometry and the pipeline. It cannot tell
  you how accurate MediaPipe's landmarks are on *your* face in *your* lighting;
  the separation number from `calibrate` is what answers that.
- `shoulder_depth` uses MediaPipe's metric world landmarks. With hips off-screen
  those are extrapolated and usually noise, so it normally ends up weighted zero.
  It is left in because calibration decides, not me.
- A frontal setup cannot isolate protraction. See the top of this file.
- A side setup without your hip in frame cannot separate protraction from a
  craning neck either. Calibration warns you; the off-axis penalty catches the
  worst of it; neither is a substitute for moving the camera back.
- Multiple people in frame: only the most prominent pose is used.
- Sharing the camera with a video call works on macOS, but the calibration was
  made at your normal seating distance; expect noisier scores if the call app
  changes the resolution.

## Licence

MIT.
