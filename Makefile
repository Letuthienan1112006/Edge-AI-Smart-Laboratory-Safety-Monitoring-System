PY ?= python3
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: demo-video kiosk-setup kiosk-off demo-day demo-open paper-pdf install-ramlog-service install-live-service dashboard-tunnel report-pdf energy collect-day2 dashboard dashboard-demo fetch-violations import-roboflow deploy-board start-capture capture-status stop-capture camera collect-guided peek fetch-sessions build-lab-dataset help bootstrap demo demo-parallel board-live fetch-board dash bench report verify zones test clean probe roofline sweep affinity parity sweep-workers export-rknn

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

bootstrap:       ## ON THE BOARD ONLY: apt+pip install + hardware checks. See docs/board-setup.md
	bash scripts/bootstrap_board.sh
demo:            ## live webcam, adaptive mode, overlay window
	$(PY) -m labsafe.app --source 0 --mode adaptive
demo-parallel:   ## same, but the E1 baseline (every stage every frame)
	$(PY) -m labsafe.app --source 0 --mode parallel
board-live:      ## ON BOARD: RKNN live run; always save video, log, summary and frames
	bash scripts/run_board_live.sh
fetch-board:     ## ON LAPTOP: pull the latest complete board-live bundle
	bash scripts/fetch_board_run.sh $(BOARD)
dash:            ## headless + web dashboard on :8000
	$(PY) -m labsafe.app --source 0 --no-display --dashboard
bench:           ## short smoke benchmark (60s). Real runs: scripts/bench_matrix.sh
	$(PY) -m labsafe.app --bench --exp SMOKE --duration 60 --warmup 5 --no-display
report:          ## build the results table from runs/
	$(PY) -m labsafe.bench.report runs
probe:           ## what can this CPU actually do? run on laptop AND Pi
	$(PY) -m labsafe.optimize.arm_probe
affinity:        ## big.LITTLE topology + measure big vs LITTLE vs unpinned
	$(PY) -m labsafe.optimize.affinity $(ARGS)
roofline:        ## compute-bound or memory-bound? picks the optimisation
	$(PY) -m labsafe.optimize.roofline $(ARGS)
sweep:           ## free speedup: thread/backend/imgsz sweep
	$(PY) scripts/sweep_runtime.py --backend $(or $(BACKEND),stub) --threads 1,2,3,4 $(ARGS)
parity:          ## conversion parity gate: make parity ARGS="--reference-backend onnx ..."
	$(PY) scripts/validate_conversion.py $(ARGS)
sweep-workers:   ## throughput-vs-worker-count curve + saturation point (N4)
	$(PY) scripts/sweep_workers.py --backend $(or $(BACKEND),stub) --workers 1,2,3,4 $(ARGS)
export-rknn:     ## convert .pt -> .rknn for RK3588S. x86 host ONLY, see docs/rknn-pipeline.md
	$(PY) scripts/export_rknn.py $(ARGS)
verify:          ## verify a dataset: make verify DATA=path/to/data.yaml
	$(PY) scripts/verify_dataset.py $(DATA) --crops 8
zones:           ## draw restricted zones on a live frame
	$(PY) tools/zone_editor.py --source 0
test:            ## unit tests (no camera, no models needed)
	$(PY) -m pytest tests -q
clean:
	rm -rf runs/*/ data/interim/crops

collect-guided:      ## ON BOARD: guided RAW recording of a collection plan (resumes). PLAN=configs/collection_plan_L2.yaml
	python3 scripts/collect_guided.py --plan $(or $(PLAN),configs/collection_plan.yaml) $(ARGS)
collect-day2:        ## ON LAPTOP: guided day-2 recording on the board (frees the camera first), then start all-day capture
	bash scripts/collect_day2.sh
peek:                ## ON LAPTOP: grab one frame from the board camera to check framing
	bash scripts/peek_camera.sh
fetch-sessions:      ## ON LAPTOP: pull sessions, cut frames, draft labels (see scripts/fetch_and_prepare.sh)
	bash scripts/fetch_and_prepare.sh
build-lab-dataset:   ## ON LAPTOP: corrected labels (data/labeled) -> data/processed/lab_v1, split by session
	.venv/bin/python scripts/build_lab_dataset.py

dashboard:           ## ON LAPTOP: run the lab safety system on the board, open its web dashboard (localhost:8000)
	bash scripts/dashboard.sh
dashboard-demo:      ## ON LAPTOP: same dashboard on a recorded video, lab model in PyTorch on CPU (no board, no GPU needed)
	CUDA_VISIBLE_DEVICES= PYTHONPATH=src $(or $(wildcard .venv/bin/python),$(PY)) -m labsafe.app --config configs/live/dashboard_lab_pc.yaml --classes configs/classes_lab_v3.yaml \
	  --source $(or $(VIDEO),data/interim/demo/boot014_demo.mp4) --demo-video --no-display --run-dir runs/dashboard-demo & \
	  sleep 6; xdg-open http://127.0.0.1:8000/#present >/dev/null 2>&1 || echo "Open http://127.0.0.1:8000/#present"; wait
fetch-violations:    ## ON LAPTOP: copy the violation photos of every dashboard run from the board to runs/board/
	mkdir -p runs/board && rsync -a --include='live-*/' --include='live-*/violations/***' --exclude='*' \
	  $(or $(BOARD),orangepi-lan):AI_Challenge/runs/ runs/board/ && ls -d runs/board/live-*/violations 2>/dev/null || true
poster-pdf:           ## build poster/poster.pdf (A2, tiếng Việt) from LaTeX (tectonic)
	cd poster && tectonic -X compile poster.tex 2>&1 | grep -vE 'accessing absolute path' | tail -8

paper-pdf:           ## build report/bai_bao_ngan.pdf (short 2-column paper, 6-8 pages, for the preliminary submission)
	cd report && $(or $(wildcard $(HOME)/.local/bin/tectonic),tectonic) bai_bao_ngan.tex

report-pdf:          ## build report/bao_cao.pdf from LaTeX (tectonic, installed in ~/.local/bin; no root needed)
	cd report && $(or $(wildcard $(HOME)/.local/bin/tectonic),tectonic) bao_cao.tex
meter-check:         ## ON LAPTOP: check the USB-C meter is trustworthy (timed baseline + CPU load on the board) BEFORE make energy
	python3 scripts/meter_check.py

energy:              ## ON LAPTOP: measure board power with the USB-C meter (idle, adaptive, parallel; you type the Wh readings)
	python3 scripts/energy_session.py $(ARGS)
camera:              ## ON LAPTOP: live board-camera view in the browser (framing check)
	bash scripts/camera_view.sh
start-capture:       ## ON LAPTOP: start unattended capture on the board (keeps running after you unplug the cable)
	bash scripts/capture_ctl.sh start
capture-status:      ## ON LAPTOP: is the board capturing? frames per session, age of the last frame
	bash scripts/capture_ctl.sh status
stop-capture:        ## ON LAPTOP: stop the board's unattended capture
	bash scripts/capture_ctl.sh stop

deploy-board:        ## ON LAPTOP: copy scripts/src/configs to the board (the board has no git)
	rsync -a scripts/ orangepi-lan:AI_Challenge/scripts/ --exclude __pycache__
	rsync -a src/ orangepi-lan:AI_Challenge/src/ --exclude __pycache__
	rsync -a configs/ orangepi-lan:AI_Challenge/configs/
import-roboflow:     ## ON LAPTOP: Roboflow YOLOv8 export (ZIP=path, MERGE=1) -> data/labeled/<session>
	.venv/bin/python scripts/import_roboflow.py $(ZIP) $(if $(MERGE),--merge-coats,)

install-live-service: ## ON LAPTOP: install the live system as a systemd service on the board (asks the board's sudo password)
	bash scripts/install_live_service.sh

dashboard-tunnel:     ## ON LAPTOP: open the dashboard of the running service (SSH tunnel to the board, http://localhost:8000)
	@echo "Dashboard at http://localhost:8000 (Ctrl+C closes the tunnel; the service keeps running)"
	@( sleep 2; command -v xdg-open >/dev/null && xdg-open http://localhost:8000 >/dev/null 2>&1 ) & \
	ssh -N -L 8000:127.0.0.1:8000 orangepi-lan

lab-preflight:        ## ON LAPTOP: one-command check before testing/filming at the lab (OPEN=1 also opens the dashboard)
	bash scripts/lab_preflight.sh $(if $(OPEN),--open,)

live-config:          ## ON LAPTOP: show / switch the live config: make live-config [CONFIG=demo|standard] (switching asks sudo)
	bash scripts/live_config.sh $(or $(CONFIG),show)

record-demo:          ## ON LAPTOP: record the backup demo video (annotated stream + dashboard window): make record-demo [MODE=stream|screen|both] [NAME=...]
	bash scripts/record_demo.sh $(or $(MODE),both) $(or $(NAME),demo)

panel-setup:          ## ON LAPTOP, ONCE: board permissions + I2C5 overlay for the LEDs/OLED (asks sudo), then reboot the board
	bash scripts/setup_panel_board.sh

panel-test:           ## ON LAPTOP: test the LEDs + OLED on the board (stops the live service, runs the test, starts it again; asks sudo)
	ssh -t $(or $(HOST),orangepi-lan) 'trap "sudo systemctl start labsafe-live" EXIT; sudo systemctl stop labsafe-live; cd ~/AI_Challenge && PYTHONPATH=src python3 scripts/panel_test.py'

board-internet:       ## ON LAPTOP: give the board internet through the LAN cable (laptop's Wi-Fi is the way out): make board-internet MODE=on|off|status
	bash scripts/board_internet.sh $(or $(MODE),status)

zalo-setup:           ## ON LAPTOP: connect alarms to your Zalo: make zalo-setup TOKEN=~/zalo_token.txt (then message your bot)
	python3 scripts/zalo_setup.py --token-file $(TOKEN)

zalo-push:            ## ON LAPTOP: copy the Zalo secret to the board + send a test message FROM the board (needs internet on the board)
	python3 scripts/zalo_setup.py --push

install-ramlog-service: ## ON LAPTOP: install the RAM logger as a boot-time service on the board (asks the sudo password)
	bash scripts/install_ramlog_service.sh

demo-day:             ## ON LAPTOP: ONE command for the demo: wait for the board, sync clock, deploy, restart, check, open the web page. Options: ARGS="--net" (Zalo) / "--no-deploy"
	bash scripts/demo_day.sh $(ARGS)

demo-open:            ## ON LAPTOP: board already running (plain power-on): just open the tunnel + the page in presentation mode
	bash scripts/demo_day.sh --no-deploy

kiosk-setup:          ## ON LAPTOP, ONCE: board shows the monitoring page full screen on an HDMI monitor by itself after power-on
	bash scripts/setup_kiosk.sh

kiosk-off:            ## ON LAPTOP: stop the board from opening the page on HDMI at boot
	bash scripts/setup_kiosk.sh off

demo-video:           ## ON LAPTOP: the REAL system (NPU, LEDs, OLED, web) fed with the lab VIDEO instead of the camera: make demo-video [VIDEO=path.mp4]; back to the camera: make demo-day
	DEMO_VIDEO=$(or $(VIDEO),data/interim/demo/boot014_demo.mp4) bash scripts/demo_day.sh $(ARGS)
