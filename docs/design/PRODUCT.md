# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Three audiences, all confirmed by the owner:

- **Lab supervisor.** Watches a screen in or near the lab, sometimes from several metres away. Job: know immediately whether anyone is unsafe right now (no lab coat in a danger zone, fall, fire, smoke), see who and where, and verify the evidence.
- **Competition judges and demo audiences.** Meet the page for the first time on a projector or large screen. Job: understand within seconds what the system detects and that it is live.
- **The development team.** Tunes the system: reads FPS, redraws the danger zones when the camera moves, checks false alarms.

## Product Purpose

Lab Safety Edge AI runs on an Orange Pi 5 (RK3588S) with a camera and YOLO-based models on the NPU. It detects people, whether they wear a lab coat, whether they are inside a danger zone, falls, fire and smoke, and raises alarms with photo evidence. The dashboard is its web face: live annotated video, current counts, alarm history with evidence photos, a 15-minute timeline, and an in-page editor for the danger zones.

## Positioning

Edge-only and offline: the whole pipeline and this page run on the board with no cloud and no internet. Every alarm carries the evidence photo of the person and the full frame, so a supervisor can verify without opening another tool.

## Operating Context

- Served by the board's own stdlib HTTP server; reached on the viewer's laptop through an SSH tunnel to localhost (`make dashboard`, `make dashboard-tunnel`). The page is not meant to be reachable from the network.
- The board has no internet and no reliable clock; times are shown as ages relative to `server_now` and converted to the viewer's clock.
- The live system runs as the `labsafe-live` service; the dashboard also runs on recorded video on a laptop (`make dashboard-demo`).

## Capabilities and Constraints

- Single self-contained page: no CDN, no downloaded fonts, no external requests.
- Backend contract is fixed: `/stream` (MJPEG), `/status`, `/events`, `/snap/<id>`, `/crop/<id>`, `/timeline`, `GET/POST /zones`, `POST /zones/reset`. The redesign changes the page only.
- Vietnamese UI.
- The danger-zone editor stays on the page. Saving replaces the zones that define what counts as a violation, so it must be hard to do by accident.
- No login and no wider network exposure; the access model stays as is.
- Alarm acknowledgement, if offered, lives only in the browser page and is lost on reload; nothing is stored or sent to the board.
- Terminology in the product: "áo lab" (lab coat), "vùng nguy hiểm" (danger zone), "vi phạm" (violation), "cảnh báo" (alarm).

## Evidence on Hand

- Recorded lab video for demos: `data/interim/demo/boot014_demo.mp4`.
- Measured results and evidence photos are in `docs/results.md` and the run directories; no invented figures belong on the page.
- No brand assets (logo, school colours) are recorded. Do not fabricate any.

## Product Principles

- The first question is "is anything wrong right now?"; the page answers it before anything else.
- Calm must look calm: a quiet room reads as safe, not as three coloured warnings.
- Evidence beats assertion: show the person and the frame behind every alarm.
- Configuration that defines safety rules is guarded against accidental change.
- Works offline and on the board's constraints before it works pretty.

## Accessibility & Inclusion

No product-specific standard was set. Working baseline: text contrast at least WCAG AA, state never carried by colour alone, keyboard-operable controls, and alarms announced to assistive technology.
