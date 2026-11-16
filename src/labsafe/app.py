"""Entry point: build the system from config, run it, or benchmark it.

    python3 -m labsafe.app --source 0                  # live, with overlay
    python3 -m labsafe.app --bench --exp E1 --mode parallel --duration 900
    python3 -m labsafe.app --source clip.mp4 --no-display

    # first live run on a real RKNN model, COCO-80 checkpoint not fine-tuned
    # yet for this project's own classes (see KNOWN_ISSUES.md #20):
    python3 -m labsafe.app --source 0 --no-display --gpio off \\
        --config configs/exp/N1L_live_rknn_fp16_640.yaml \\
        --classes configs/classes_coco80.yaml \\
        --save-every 3 --save-max 20
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from .bench.runner import BenchConfig, BenchRunner, percentiles
from .config import ROOT, Config, load_yaml
from .io.alerts import AlertSink
from .io.camera import CameraSource
from .io.snapshots import FrameSaver
from .io.sysstats import SysStats
from .io.recording import AsyncVideoRecorder
from .models.backends import build_backend
from .pipeline.decision import worst_severity
from .pipeline.fall import FallConfig
from .pipeline.ppe import PPEChecker, PPEConfig
from .pipeline.router import AdaptiveRouter
from .pipeline.zones import ZoneStore, load_live_zones, load_zones
from .viz import draw


def _spin_warning(result: dict, tolerance: float = 2.0) -> str | None:
    """Flag a run whose loop iterated far more than it processed frames.

    Returns a warning string, or None when the two counters agree. The
    tolerance is generous on purpose: warmup iterations and a frame or two
    of slack are normal, five orders of magnitude are not.
    """
    iterations = result.get("iterations_total") or 0
    frames = (result.get("router") or {}).get("frames") or 0
    if iterations < 50 or frames <= 0:
        return None            # too short to judge, or no router in this run
    ratio = iterations / frames
    if ratio <= tolerance:
        return None
    return (
        f"loop spun {ratio:.0f}x more iterations ({iterations}) than frames "
        f"processed ({frames}) -- the source likely ended early and the "
        f"reported FPS measures the loop, not the model. stop_reason="
        f"{result.get('stop_reason', '?')}"
    )


def camera_stats(cam: CameraSource, frames_processed: int, read_ms: list[float]) -> dict:
    """Camera-side numbers, separate from the model-side ones in `router.stats`.

    `capture_fps` measures what the CAMERA delivered (frames_read / wall
    time since open) -- a different quantity from `fps_sustained`, which
    measures how fast the LOOP ran. `duplicate_ratio` is the bridge between
    them: see `CameraSource.read()` and KNOWN_ISSUES.md #6.
    """
    w, h = cam.resolution
    duplicate_reads = getattr(cam, "duplicate_reads", 0)
    t_open = getattr(cam, "t_open", None)
    capture_fps = (
        round(cam.frames_read / max(1e-9, time.time() - t_open), 2)
        if t_open is not None else None
    )
    return {
        "resolution": [w, h],
        "frames_captured": cam.frames_read,
        "capture_fps": capture_fps,
        "duplicate_reads": duplicate_reads,
        "duplicate_ratio": round(duplicate_reads / max(1, frames_processed), 4),
        "read_ms": percentiles(read_ms),
    }


def _duplicate_warning(cam_stats: dict, tolerance: float = 0.05) -> str | None:
    """Flag a run where a meaningful share of "processed" frames were
    actually the same camera frame handed back twice.

    Deliberately does NOT set `result_valid=False`: this is an
    interpretation aid for reading FPS numbers, not evidence the run's
    data is wrong the way `_spin_warning`/`liveness_warning` are.
    """
    ratio = cam_stats.get("duplicate_ratio")
    if ratio is None or ratio <= tolerance:
        return None
    fps = cam_stats.get("capture_fps")
    return (
        f"duplicate_ratio={ratio:.2%} of processed frames were the same camera "
        f"frame read more than once (see KNOWN_ISSUES.md #6 and "
        f"CameraSource.read()) -- the reported FPS is higher than real frame "
        f"throughput. Unique FPS is approximately fps * (1 - duplicate_ratio)"
        + (f"; camera capture_fps={fps}" if fps is not None else "") + "."
    )


def apply_affinity(cfg: Config) -> dict:
    """Pin the process per `affinity.cpus` before any inference happens.

    Applied at startup and left in place: an unpinned run on a big.LITTLE
    SoC lets the scheduler migrate the inference thread onto a LITTLE core
    mid-benchmark, which lands in p99 as variance nobody can attribute.
    The mask that was actually applied is returned so it can be recorded in
    the run's metadata -- a latency number whose core mask is unknown is
    not reproducible.
    """
    import os

    from .optimize.affinity import cluster_masks, supported
    from .optimize.arm_probe import probe_cpu

    want = cfg.pipeline.get("affinity", {}).get("cpus")
    if want in (None, "none", "auto", False):
        return {"affinity": None, "reason": "not requested"}
    if not supported():
        return {"affinity": None, "reason": "sched_setaffinity unavailable"}

    info = probe_cpu()
    if isinstance(want, str):
        masks = cluster_masks(info)
        if want not in masks:
            log.warning("affinity.cpus=%r not available (have %s); leaving unpinned",
                        want, sorted(masks))
            return {"affinity": None, "reason": f"mask {want!r} not found"}
        cpus = masks[want]
    else:
        cpus = [int(c) for c in want]

    try:
        os.sched_setaffinity(0, set(cpus))
    except OSError as e:
        log.warning("could not pin to %s: %s", cpus, e)
        return {"affinity": None, "reason": str(e)}

    log.info("pinned to cpus %s (%s)", cpus, want)
    return {"affinity": cpus, "affinity_name": str(want),
            "heterogeneous": info.heterogeneous}

log = logging.getLogger("labsafe")


def build_system(cfg: Config, mode: str, stats_window: int | None = None) -> tuple[AdaptiveRouter, AlertSink, dict]:
    from .optimize.engine_map import EnginePlacement, resolve_stage_engines

    placements = resolve_stage_engines(cfg.pipeline)

    def _with_core_mask(stage_cfg: dict, placement: EnginePlacement | None) -> dict:
        # A latency number whose engine is unknown is not reproducible --
        # `core_mask` must reach the backend that actually reads it
        # (RKNNBackend), not just live in the placement record.
        if placement is not None and placement.engine == "npu" and placement.core_mask:
            stage_cfg = {**stage_cfg, "core_mask": placement.core_mask}
        return stage_cfg

    names_a = cfg.classes["stage_a"]["names"]
    stage_a = build_backend(_with_core_mask(cfg.pipeline["stage_a"], placements.get("stage_a")), names_a)

    stage_b = None
    b_cfg = cfg.pipeline.get("stage_b", {})
    if b_cfg.get("enabled", False):
        names_b = cfg.classes["stage_b"]["names"]
        required = cfg.classes["stage_b"]["required"]
        unknown = set(required) - set(names_b)
        if unknown:
            raise ValueError(
                f"configs/classes.yaml: required PPE {sorted(unknown)} is not in "
                f"stage_b.names -- the detector can never report it, so every "
                f"person would be flagged non-compliant forever"
            )
        stage_b = PPEChecker(
            build_backend(_with_core_mask(b_cfg, placements.get("stage_b")), names_b),
            PPEConfig.from_dict(b_cfg, required),
        )

    stage_c = None
    c_cfg = cfg.pipeline.get("stage_c", {})
    if c_cfg.get("enabled", False):
        stage_c = build_backend(_with_core_mask(c_cfg, placements.get("stage_c")),
                                 cfg.classes["stage_c"]["names"])

    zones_cfg = cfg.pipeline.get("zones", {})
    zones = load_zones(cfg.zones) if zones_cfg.get("enabled") else []
    default_zones = list(zones)
    # Same guard as stage_b.required: a zone that requires an item stage A can never report would flag every person
    # who steps in, forever. Stage A must both know the class and keep it (keep_classes).
    keep_a = cfg.pipeline["stage_a"].get("keep_classes")
    allowed_requires = tuple(n for n in names_a if n != "person" and (keep_a is None or n in keep_a))
    if zones_cfg.get("enabled"):
        # zones drawn on site in the dashboard win over the shipped file (deploy-board must not undo them)
        live = load_live_zones(zones_cfg.get("live_file", "data/zones_live.yaml"), allowed_requires)
        if live is not None:
            log.info("using zones drawn on site: %s", zones_cfg.get("live_file", "data/zones_live.yaml"))
            zones = live
    for z in zones:
        bad = [r for r in z.requires if r not in names_a or (keep_a is not None and r not in keep_a)]
        if bad:
            raise ValueError(
                f"zone {z.name!r} requires {bad}, but stage A (names {names_a[:5]}..., keep_classes {keep_a}) "
                f"never outputs it -- every person entering would be flagged"
            )
    fall_cfg = FallConfig.from_dict(cfg.pipeline.get("fall", {}))
    fall_cfg.fps_estimate = float(cfg.pipeline.get("source", {}).get("fps", 15))

    router = AdaptiveRouter(
        stage_a, stage_b, stage_c, zones, fall_cfg,
        mode=mode,
        tracker_cfg=cfg.pipeline.get("tracker", {}),
        alerts_cfg=cfg.pipeline.get("alerts", {}),
        stage_c_interval_s=float(c_cfg.get("interval_s", 1.0)),
        async_stages=bool(cfg.pipeline.get("async_stages", False)),
        attire_memory_s=float(cfg.pipeline.get("zones", {}).get("attire_memory_s", 2.0)),
        attire_min_conf=float(cfg.pipeline.get("zones", {}).get("attire_min_conf", 0.0)),
        partial_min_h=float(cfg.pipeline.get("zones", {}).get("partial_min_h", 0.0)),
        visibility=cfg.pipeline.get("zones", {}).get("visibility"),
        stats_window=stats_window,
    )
    alerts = AlertSink(cfg.pipeline.get("alerts", {}))
    engine_placements = {k: vars(p) for k, p in placements.items()}
    router.zone_store = (ZoneStore(router, zones_cfg.get("live_file", "data/zones_live.yaml"), default_zones,
                                   allowed_requires) if zones_cfg.get("enabled") else None)
    return router, alerts, engine_placements


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lab safety edge monitor")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--classes", default="configs/classes.yaml")
    ap.add_argument("--source", default=None, help="camera index or video path")
    ap.add_argument("--mode", choices=["adaptive", "parallel"], default="adaptive")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--dashboard", action="store_true", help="force dashboard on")
    ap.add_argument("--bench", action="store_true", help="run under the bench harness")
    ap.add_argument("--exp", default="E0", help="experiment id for the run directory")
    ap.add_argument("--duration", type=float, default=None, help="seconds (bench)")
    ap.add_argument("--warmup", type=float, default=None, help="seconds discarded (bench)")
    ap.add_argument("--note", default="")
    ap.add_argument("--affinity", default=None,
                    help="big | little | all | none | '4,5,6,7' (overrides config)")
    ap.add_argument("--gpio", choices=["auto", "on", "off"], default=None,
                    help="override alerts.gpio from config -- use 'off' unless "
                         "buzzer/LED wiring and gpioinfo offsets are confirmed")
    ap.add_argument("--save-every", type=float, default=0.0,
                    help="seconds between saved annotated frames, 0=off")
    ap.add_argument("--save-max", type=int, default=30,
                    help="maximum number of frames to save with --save-every")
    ap.add_argument("--record-video", action="store_true",
                    help="save annotated.mp4 asynchronously (evidence runs; opt-in for benchmarks)")
    ap.add_argument("--record-fps", type=float, default=20.0,
                    help="playback FPS written into annotated.mp4 (default: 20)")
    ap.add_argument("--run-dir", default=None,
                    help="exact artifact directory; otherwise runs/<exp-or-live>-<timestamp>")
    ap.add_argument("--replay", action="store_true",
                    help="video file source for SUSTAINED benchmarks: play at the file's FPS (camera-like pacing) and loop "
                         "until --duration ends; identical content in every run. Recorded in command.json.")
    ap.add_argument("--demo-video", action="store_true",
                    help="video file source: play at its own FPS and loop forever (dashboard demos; never for --bench)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / (
        f"{args.exp}-{stamp}" if args.bench else f"live-{stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.json").write_text(json.dumps({
        "started": datetime.now().isoformat(timespec="seconds"),
        "argv": list(sys.argv[1:] if argv is None else argv),
        "config": args.config,
        "classes": args.classes,
        "source": args.source,
        "mode": args.mode,
        "bench": args.bench,
        "replay": bool(getattr(args, "replay", False)),
    }, indent=2))
    log_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), log_handler],
        force=True,
    )
    log.info("run artifacts: %s", run_dir.resolve())

    cfg = Config.load(args.config, args.classes)
    if args.affinity is not None:
        cfg.pipeline.setdefault("affinity", {})["cpus"] = (
            None if args.affinity == "none" else args.affinity
        )
    if args.gpio is not None:
        cfg.pipeline.setdefault("alerts", {})["gpio"] = args.gpio
    affinity_meta = apply_affinity(cfg)
    src_cfg = cfg.pipeline.get("source", {})
    device: int | str = src_cfg.get("device", 0)
    if args.source is not None:
        device = int(args.source) if args.source.isdigit() else args.source

    # Benchmarks keep every latency sample (the report needs them); live runs may last a day, so they keep a window.
    stats_window = None if args.bench else int(cfg.pipeline.get("stats_window", 18000))   # 18000 = 10 min at 30 FPS
    router, alerts, engine_placements = build_system(cfg, args.mode, stats_window)
    router.stage_a.warmup()
    if router.stage_c is not None:
        router.stage_c.warmup()

    dash = None
    d_cfg = cfg.pipeline.get("dashboard", {})
    if (d_cfg.get("enabled") and not args.bench) or args.dashboard:
        from .io.dashboard import Dashboard

        dash = Dashboard(d_cfg.get("host", "127.0.0.1"), int(d_cfg.get("port", 8000)),
                         int(d_cfg.get("jpeg_quality", 70)), zones=getattr(router, "zone_store", None))
        log.info("dashboard: %s", dash.url)

    # phone notification (Zalo), live runs only: a benchmark or a replayed video must not message anyone
    notifier = None
    if not args.bench and not getattr(args, "replay", False):
        from .io.notify import build_notifier

        notifier = build_notifier(cfg.pipeline.get("notify"))

    # LED + OLED status panel (live runs only); a missing device or permission is logged and skipped, never fatal
    panel = None
    if not args.bench:
        from .io.panel import build_panel

        panel = build_panel(cfg.pipeline.get("panel"))

    evidence = None
    e_cfg = cfg.pipeline.get("evidence", {})
    if e_cfg.get("enabled") and not args.bench:
        from .io.evidence import ViolationRecorder

        evidence = ViolationRecorder(
            e_cfg.get("dir") or (run_dir / "violations"),
            kinds=tuple(e_cfg.get("kinds", ["zone_ppe_violation", "zone_intrusion"])),
            padding=float(e_cfg.get("padding", 0.15)),
            max_files=int(e_cfg.get("max_files", 2000)),
        )
        log.info("violation photos: %s", evidence.dir.resolve())

    display = not args.no_display and not args.bench
    if args.demo_video and args.bench:
        ap.error("--demo-video paces and loops the file: it would make a benchmark meaningless")
    # --replay is the benchmark-legal twin of --demo-video: same pacing + loop, but it is an explicit, recorded choice
    # (a file played unpaced and stopped at its end is a different experiment, KNOWN_ISSUES #6).
    cam_kw = {"realtime": True, "loop": True} if (args.demo_video or args.replay or src_cfg.get("demo_video")) else {}   # source.demo_video: a video-file demo profile (configs/live/video_demo_board.yaml)
    cam = CameraSource(device, src_cfg.get("width", 640), src_cfg.get("height", 480),
                       src_cfg.get("fps", 30), **cam_kw)
    log.info("source %s @ %s, mode=%s, stage_a=%s", device, cam.resolution,
             args.mode, router.stage_a.name)

    state = {"stop": False, "last_fps_t": time.time(), "fps": 0.0, "n": 0, "last_vis_t": 0.0}
    dash_video_fps = float(d_cfg.get("video_fps", 15))   # 0 = every frame
    sysstats = SysStats() if dash is not None else None   # CPU / RAM / temperature / NPU gauges for the page
    read_ms = [] if stats_window is None else deque(maxlen=stats_window)
    # Mutable box so `step()` (a closure captured once, before either the
    # bench or the live save directory exists) can see a FrameSaver that
    # gets assigned later -- the bench branch only knows `runner.out` after
    # `BenchRunner` is constructed, well after `step` is defined.
    saver_box: list[FrameSaver | None] = [None]
    recorder: AsyncVideoRecorder | None = (
        AsyncVideoRecorder(run_dir / "annotated.mp4", cam.resolution, args.record_fps)
        if args.record_video else None
    )
    previous_sigint = None
    if not args.bench and threading.current_thread() is threading.main_thread():
        # Do not raise KeyboardInterrupt in the middle of RKNN's native
        # inference call: rknn-lite catches it internally, returns None and
        # leaves Python to fail later with a misleading TypeError. Request a
        # clean stop and let the current frame finish instead.
        previous_sigint = signal.getsignal(signal.SIGINT)

        def _request_clean_stop(_signum, _frame):
            state["stop"] = True
            log.info("stop requested -- finishing current frame")

        signal.signal(signal.SIGINT, _request_clean_stop)

    def step() -> bool | None:
        """Returns False when the frame source is exhausted.

        The bench runner stops on False. Without it a finished video file
        turns the remaining minutes of the run into no-op iterations and
        the reported FPS becomes a property of the loop, not the model.
        """
        t_r0 = time.perf_counter()
        ok, frame = cam.read(wait_new=True)   # one inference per camera frame, not per loop spin
        read_ms.append((time.perf_counter() - t_r0) * 1000.0)
        if not ok or frame is None:
            if cam.is_file:
                state["stop"] = True
                return False
            return None
        events, alarming = router.process(frame)
        crops = None
        if evidence is not None and alarming:
            # crop from the RAW frame (no boxes drawn on the person), keep an annotated frame for context
            crops = evidence.record(frame, draw(frame, router.tracks, router.zones, events, coat_scores=getattr(router, "_coat_scores", {}),
                                                  memory_s=router.attire_memory_s), alarming, router.tracks)
        for ev in alarming:
            alerts.alert(ev.severity, f"{ev.kind} {ev.detail} track={ev.track_id}")
        if notifier is not None:
            notifier.on_alarm(alarming, crops)     # every frame: alarms held back by its min interval go out later
        if not alarming and not events:
            alerts.clear()

        state["n"] += 1
        now = time.time()
        if now - state["last_fps_t"] >= 1.0:
            state["fps"] = state["n"] / (now - state["last_fps_t"])
            state["n"], state["last_fps_t"] = 0, now
        if panel is not None:
            if now - state.get("panel_t", 0.0) >= 0.2:      # the people counts need no more than 5 updates a second
                state["panel_sum"], state["panel_t"] = router.people_summary(events), now
            panel.update(state.get("panel_sum"), events, state["fps"])

        saver = saver_box[0]
        saver_due = saver is not None and saver.due(now)
        if display or dash is not None or saver_due or recorder is not None:
            sev = worst_severity(events)
            hud = (
                f"mode={router.mode} fps={state['fps']:.1f} "
                f"A={router.stage_a.last_latency_ms:.0f}ms "
                f"occup={router.stats.occupancy:.2f}\n"
                f"B/frame={router.stats.stage_calls['b'] / max(1, router.stats.frames):.2f} "
                f"severity={sev}"
            )
            # Drawing + JPEG-encoding the page's video costs a few ms per frame in this single-threaded loop; a person watching the
            # page does not need 30 FPS of it. The status / alarms still update every frame; frames that raise an alarm, are
            # displayed, recorded or saved are always drawn.
            dash_due = dash is not None and now - state["last_vis_t"] >= 1.0 / max(1.0, dash_video_fps)
            need_vis = display or recorder is not None or saver_due or bool(alarming) or dash_due or dash_video_fps <= 0
            # the engineering HUD (mode/fps/latency) is for the local window and recordings; the dashboard shows FPS itself
            hud_txt = hud if (display or recorder is not None or saver_due) else ""
            vis = draw(frame, router.tracks, router.zones, events, hud_txt, coat_scores=getattr(router, "_coat_scores", {}),
                       memory_s=router.attire_memory_s) if need_vis else None
            if dash_due or dash_video_fps <= 0:
                state["last_vis_t"] = now
            if dash is not None:
                dash.publish(vis if (dash_due or alarming or dash_video_fps <= 0) else None, {
                    "severity": sev,
                    "fps": round(state["fps"], 1),
                    "occupancy": round(router.stats.occupancy, 3),
                    "tracks": len(router.tracks),
                    "summary": router.people_summary(events),
                    "events": [e.as_dict() for e in events],
                    "stage_calls": router.stats.stage_calls,
                    "evidence_saved": evidence.saved if evidence is not None else None,
                    "sys": sysstats.update(state["fps"], router.stage_a.last_latency_ms),
                }, alarming, crops)
            if display:
                cv2.imshow("labsafe", vis)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    state["stop"] = True
            if saver_due:
                meta = {
                    "frame_index": router.stats.frames,
                    "tracks": [
                        {"track_id": tr.track_id, **tr.det.as_dict()} for tr in router.tracks
                    ],
                    "events": [e.as_dict() for e in events],
                    "stage_a_latency_ms": round(router.stage_a.last_latency_ms, 2),
                    "stage_a_timing": dict(getattr(router.stage_a, "last_timing", {})),
                }
                saver.save(now, vis, meta)
            if recorder is not None:
                recorder.write(vis)

    try:
        if args.bench:
            bcfg = BenchConfig.from_dict(load_yaml("configs/bench.yaml"))
            bcfg.exp_id, bcfg.note = args.exp, args.note
            if args.duration is not None:
                bcfg.duration_s = args.duration
            if args.warmup is not None:
                bcfg.warmup_s = args.warmup
            if bcfg.warmup_s >= bcfg.duration_s:
                ap.error(
                    f"warmup ({bcfg.warmup_s}s) >= duration ({bcfg.duration_s}s): "
                    "the run would measure nothing"
                )
            if bcfg.duration_s < 300:
                log.warning(
                    "duration %.0fs is a burst, not a sustained measurement -- "
                    "thermal throttling on a Pi 4 needs >=900s (see "
                    "docs/measurement-protocol.md). Fine for a smoke test only.",
                    bcfg.duration_s,
                )
            stage_a_card = getattr(router.stage_a, "card", None)
            extra_meta = {
                "mode": args.mode,
                "backend": router.stage_a.name,
                "imgsz": router.stage_a.imgsz,
                "source": str(device),
                "pipeline_config": cfg.pipeline,
                "engine_placements": engine_placements,
                **affinity_meta,
            }
            if stage_a_card is not None:
                # So a number in metrics.json can always be traced back to
                # the exact model file + precision it came from, not just
                # "backend: rknn".
                extra_meta["stage_a_model_card"] = stage_a_card.as_dict()
            runner = BenchRunner(bcfg, extra_meta=extra_meta, out=run_dir)
            if args.save_every > 0:
                saver_box[0] = FrameSaver(runner.out / "frames", args.save_every, args.save_max)

            is_tty = sys.stderr.isatty()
            last_print = [0.0]

            def tick(n: int, elapsed: float) -> None:
                # Once a second, not once every 50 frames: at 300 FPS the
                # latter writes 6 lines a second into a redirected log and
                # turns a 15-minute run into megabytes of progress bar.
                if elapsed - last_print[0] < 1.0:
                    return
                last_print[0] = elapsed
                end, flush = ("\r", True) if is_tty else ("\n", False)
                print(f"  {elapsed:6.0f}s / {bcfg.duration_s:.0f}s  frames={n}",
                      end=end, flush=flush, file=sys.stderr)

            result = runner.run(step, on_tick=tick)
            result["router"] = router.stats.summary()
            result["router"]["async_stages"] = router.async_stats()
            result["camera"] = camera_stats(cam, router.stats.frames, read_ms)
            result["frames_saved"] = saver_box[0].saved if saver_box[0] else 0

            dup_warning = _duplicate_warning(result["camera"])
            if dup_warning is not None:
                sysinfo = result.setdefault("system", {})
                # Interpretation warning, not an evidence-of-bad-data warning
                # like spin/liveness below -- deliberately does not touch
                # result_valid.
                existing = sysinfo.get("warning", "")
                sysinfo["warning"] = (existing + " | " if existing else "") + dup_warning

            # Cross-check two counters that came from different places: the
            # runner counted loop iterations, the router counted frames it
            # actually processed. They should agree closely. A large gap
            # means the loop spun on a step that did no work -- the shape of
            # a fabricated FPS. The numbers already sat side by side in this
            # file; nothing was reading them against each other.
            spin = _spin_warning(result)
            if spin is not None:
                sysinfo = result.setdefault("system", {})
                sysinfo["result_valid"] = False
                existing = sysinfo.get("warning", "")
                sysinfo["warning"] = (existing + " | " if existing else "") + spin

            liveness = router.stats.liveness_warning()
            if liveness is not None:
                sysinfo = result.setdefault("system", {})
                sysinfo["result_valid"] = False
                # Append, never overwrite: a throttle warning from sysmon
                # and a dead-model warning from the router are two
                # independent reasons to distrust this run, and both must
                # survive into metrics.json.
                existing = sysinfo.get("warning", "")
                sysinfo["warning"] = (existing + " | " if existing else "") + liveness

            (runner.out / "metrics.json").write_text(json.dumps(result, indent=2))
            (runner.out / "events.json").write_text(json.dumps(router.stats.events, indent=2))
            printed = {k: result[k] for k in ("exp_id", "latency_ms", "fps_sustained")}
            printed["camera"] = result.get("camera")
            printed["stage_a_breakdown_ms"] = result.get("router", {}).get("stage_a_breakdown_ms")
            print(json.dumps(printed, indent=2))
            sysinfo = result.get("system", {})
            if sysinfo.get("result_valid") is False:
                print("\n!! " + sysinfo.get("warning", ""), file=sys.stderr)
        else:
            live_out = run_dir / "frames"
            if args.save_every > 0:
                saver_box[0] = FrameSaver(live_out, args.save_every, args.save_max)
            try:
                while not state["stop"]:
                    step()
            except KeyboardInterrupt:
                # docs/board-setup.md §8 promises a JSON summary after
                # Ctrl+C. Before this, KeyboardInterrupt propagated straight
                # past the summary print into `finally`, so it never
                # happened -- see plan §2.5.
                log.info("interrupted -- printing summary")
            summary = router.stats.summary()
            summary["async_stages"] = router.async_stats()
            summary["camera"] = camera_stats(cam, router.stats.frames, read_ms)
            summary["run_dir"] = str(run_dir)
            if recorder is not None:
                summary["video"] = recorder.stats()
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary, indent=2))
    finally:
        cam.release()
        if recorder is not None:
            video_stats = recorder.close()
            (run_dir / "video.json").write_text(json.dumps(video_stats, indent=2))
            # Keep the primary JSON self-contained as well as the small
            # dedicated video.json. Bench and live runs use different names.
            primary = run_dir / ("metrics.json" if args.bench else "summary.json")
            if primary.exists():
                payload = json.loads(primary.read_text())
                payload["video"] = video_stats
                primary.write_text(json.dumps(payload, indent=2))
            if video_stats["error"]:
                log.error("video recorder: %s", video_stats["error"])
            else:
                log.info("video: %s (%d frames, %d dropped)",
                         video_stats["path"], video_stats["frames_written"],
                         video_stats["frames_dropped"])
        alerts.close()
        if panel is not None:
            panel.close()
        if notifier is not None:
            notifier.close()
            log.info("notify: %d sent, %d failed, %d dropped; photos %d sent, %d failed", notifier.sent, notifier.failed,
                     notifier.dropped, notifier.photos_sent, notifier.photos_failed)
        router.close()
        if dash is not None:
            dash.close()
        if display:
            cv2.destroyAllWindows()
        log.info("artifacts ready: %s", run_dir.resolve())
        logging.shutdown()
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
