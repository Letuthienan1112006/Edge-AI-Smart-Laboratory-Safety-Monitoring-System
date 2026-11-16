"""Dashboard system gauges, read from fake /proc and /sys files."""
from labsafe.io.sysstats import SysStats, parse_meminfo, parse_npu_load, parse_proc_stat

STAT1 = "cpu  100 0 100 800 0 0 0 0 0 0\ncpu0 1 1 1 1\n"
STAT2 = "cpu  200 0 200 1000 0 0 0 0 0 0\n"          # +200 busy, +200 idle -> 50 %
MEM = "MemTotal:        8000000 kB\nMemFree: 100 kB\nMemAvailable:   6000000 kB\n"
STATUS = "Name:\tpython3\nVmRSS:\t  215040 kB\n"


def make(tmp_path, npu_text=None, clock=None, **kw):
    f = {"stat": tmp_path / "stat", "mem": tmp_path / "mem", "status": tmp_path / "status", "thermal": tmp_path / "temp"}
    f["stat"].write_text(STAT1)
    f["mem"].write_text(MEM)
    f["status"].write_text(STATUS)
    f["thermal"].write_text("48076\n")
    npu = tmp_path / "npu"
    if npu_text is not None:
        npu.write_text(npu_text)
    t = clock or [0.0]
    s = SysStats(interval_s=1.0, npu_load_path=str(npu), proc_stat=str(f["stat"]), meminfo=str(f["mem"]),
                 proc_status=str(f["status"]), thermal=str(f["thermal"]), npu_freq_glob=str(tmp_path / "nofreq*"),
                 clock=lambda: t[0], **kw)
    return s, f, t


def test_parsers():
    assert parse_proc_stat(STAT1) == (200, 1000)
    assert parse_proc_stat("nothing") is None
    assert parse_meminfo(MEM) == (25.0, 2000000 / 1024.0)
    assert parse_npu_load("NPU load:  Core0:  12%, Core1: 40%, Core2:  0%,") == 40.0
    assert parse_npu_load("garbage") is None


def test_cpu_is_the_share_between_two_reads_and_first_read_is_none(tmp_path):
    s, f, t = make(tmp_path)
    first = s.update(30, 25)
    assert first["cpu_pct"] is None and first["temp_c"] == 48.1 and first["ram_pct"] == 25.0 and first["rss_mb"] == 210
    f["stat"].write_text(STAT2)
    t[0] = 2.0
    assert s.update(30, 25)["cpu_pct"] == 50.0


def test_files_are_reread_once_per_interval_only(tmp_path):
    s, f, t = make(tmp_path)
    s.update(30, 25)
    f["thermal"].write_text("70000\n")
    t[0] = 0.5
    assert s.update(30, 25)["temp_c"] == 48.1          # cached
    t[0] = 1.5
    assert s.update(30, 25)["temp_c"] == 70.0


def test_npu_is_estimated_from_inference_time_when_the_kernel_counter_is_unreadable(tmp_path):
    s, _, _ = make(tmp_path)                            # no npu file at all (root-only on the board)
    j = s.update(fps=30.0, infer_ms=25.0)
    assert j["npu_src"] == "estimate" and j["npu_pct"] == 75.0 and j["infer_ms"] == 25.0   # 25 ms x 30 /s = 75 % busy
    s2, _, _ = make(tmp_path)
    assert s2.update(fps=100.0, infer_ms=25.0)["npu_pct"] == 100.0                           # capped


def test_kernel_counter_wins_when_readable(tmp_path):
    s, _, _ = make(tmp_path, npu_text="NPU load:  Core0:  62%, Core1: 0%, Core2: 0%,")
    j = s.update(fps=30.0, infer_ms=25.0)
    assert j["npu_src"] == "kernel" and j["npu_pct"] == 62.0


def test_missing_files_give_none_not_errors(tmp_path):
    s = SysStats(proc_stat=str(tmp_path / "a"), meminfo=str(tmp_path / "b"), proc_status=str(tmp_path / "c"),
                 thermal=str(tmp_path / "d"), npu_load_path=str(tmp_path / "e"), npu_freq_glob=str(tmp_path / "f*"))
    j = s.update(0.0, 0.0)
    assert j["cpu_pct"] is None and j["temp_c"] is None and j["ram_pct"] is None and j["npu_pct"] is None
