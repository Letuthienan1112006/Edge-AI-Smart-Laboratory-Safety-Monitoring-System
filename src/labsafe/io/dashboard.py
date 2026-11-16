"""Web dashboard on the stdlib http server: live annotated video, head counts, lab-coat status, alarm history.

No Flask: one fewer dependency to install on the Pi, and nothing here needs more. Streaming costs real CPU, so it is
off during benchmarks (`dashboard.enabled: false`). Bind it to 127.0.0.1 on the board and reach it through an SSH
tunnel (`make dashboard`): the page shows people, it must not be open to the whole network.

Endpoints
  /            the page (Vietnamese, self-contained: no CDN, the board has no internet)
  /stream      MJPEG of the annotated frames
  /status      current counts + active events (JSON)
  /events      alarm history, newest first; ?since=<id> returns only newer ones
  /snap/<id>   JPEG of the frame at the moment alarm <id> fired
  /crop/<id>   JPEG of the person behind alarm <id> (only for person alarms, when evidence recording is on)
  /timeline    one sample per second for the last 15 min: persons, without coat, violating
  GET /zones   the zones in force; POST /zones replaces them (drawn in the browser), POST /zones/reset restores the
               shipped ones. POST is accepted only from a page served by this dashboard: JSON content type +
               X-Requested-With header (a foreign web page cannot send either without a CORS preflight, which
               is refused) and a localhost Host header (DNS rebinding).

Times: the board clock is not reliable (no RTC battery, no internet). Every JSON carries `server_now`; the page
shows ages ("12 s trước") and converts to the viewer's clock, never the board's wall time.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

KIND_VI = {
    "zone_ppe_violation": "Không mặc áo lab trong vùng nguy hiểm",
    "zone_intrusion": "Vào vùng cấm",
    "ppe_violation": "Thiếu đồ bảo hộ",
    "fall": "Có người té ngã",
    "fire": "Phát hiện lửa",
    "smoke": "Phát hiện khói",
}

PAGE = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>Giám sát an toàn phòng lab</title>
<link rel="icon" id="fav" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ccircle cx='16' cy='16' r='13' fill='%237d90a4'/%3E%3C/svg%3E">
<style>
:root{color-scheme:dark;
--bg:#0b0f15;--s1:#121923;--s2:#1a2431;--line:#263243;--line2:#3a4a60;
--tx:#eaf0f6;--mute:#a3b3c4;--dim:#8da0b4;--acc:#6db3ff;--pri:#2563d9;
--safe-bg:#0b3b36;--safe-fg:#d8fff3;--safe:#35d6b0;
--watch-bg:#4a3703;--watch-fg:#fff1c4;--watch:#f5b82e;
--bad-bg:#b71f1c;--bad-fg:#fff;--bad:#ff7a70;
--ack-bg:#471616;--ack-fg:#ffdcd8;
--lost-bg:#2a3441;--lost-fg:#dbe4ee;--r:10px}
*{box-sizing:border-box}
html{background:var(--bg)}
body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;accent-color:var(--acc);caret-color:var(--acc);scrollbar-color:var(--line2) var(--s1)}
::selection{background:#2b5da8;color:#fff}
button,input{font:inherit;color:inherit}
h1,h2,p,ul,ol,dl,dd,figure{margin:0}
ul,ol{padding:0;list-style:none}
:focus-visible{outline:3px solid var(--acc);outline-offset:2px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.ic{width:18px;height:18px;flex:none;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
body::after{content:"";position:fixed;inset:0;border:6px solid transparent;pointer-events:none;z-index:5}
body[data-state=alarm]::after{border-color:var(--bad-bg)}

.bar{display:flex;align-items:center;flex-wrap:wrap;gap:8px 16px;padding:10px 24px;min-height:52px;background:var(--bg);border-bottom:1px solid var(--line)}
.bar h1{font-size:18px;font-weight:650;margin-right:auto}
.live{display:inline-flex;align-items:center;gap:8px;color:var(--mute);font-size:14px}
.dot{width:10px;height:10px;border-radius:50%;background:var(--dim)}
.live.on .dot{background:var(--safe)}
@media(prefers-reduced-motion:no-preference){.live.on .dot{animation:beat 2s ease-out infinite}}
@keyframes beat{0%{opacity:1}50%{opacity:.35}100%{opacity:1}}
@media(prefers-reduced-motion:no-preference){body[data-state=alarm] .band{animation:arrive .7s cubic-bezier(.16,1,.3,1)}}
@keyframes arrive{0%{filter:brightness(1.9)}100%{filter:brightness(1)}}
.fps{color:var(--mute);font-size:14px;font-variant-numeric:tabular-nums}
.ib{display:inline-flex;align-items:center;gap:8px;min-height:36px;padding:0 12px;border:1px solid var(--line2);border-radius:8px;background:var(--s1);cursor:pointer;font-size:14px}
.ib:hover{border-color:var(--acc)}
.ib[aria-pressed=true]{background:#17314f;border-color:var(--acc)}
kbd{font:inherit;font-size:12px;padding:0 6px;border:1px solid var(--line2);border-radius:5px;color:var(--mute)}

.band{display:grid;grid-template-columns:minmax(0,1fr) auto auto;align-items:center;gap:12px 32px;min-height:136px;padding:20px 24px;background:var(--lost-bg);color:var(--lost-fg);transition:background-color .25s}
body[data-state=safe] .band{background:var(--safe-bg);color:var(--safe-fg)}
body[data-state=watch] .band{background:var(--watch-bg);color:var(--watch-fg)}
body[data-state=alarm] .band{background:var(--bad-bg);color:var(--bad-fg)}
body[data-state=ackd] .band{background:var(--ack-bg);color:var(--ack-fg)}
.word{font-size:56px;line-height:1;font-weight:800;letter-spacing:.01em}
.sent{margin-top:10px;font-size:24px;line-height:1.3;height:2.6em;overflow:hidden;display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:2}
.sent b{font-weight:700}
.more{margin-left:18px;padding:2px 14px;border:2px solid currentColor;border-radius:999px;font-size:20px;font-weight:700;letter-spacing:0;vertical-align:middle;white-space:nowrap}
.more[hidden]{display:none}
.nums{display:flex;gap:28px}
.nums div{display:flex;flex-direction:column-reverse;justify-content:flex-start;gap:6px}
.nums dd{font-size:40px;line-height:1;font-weight:700;font-variant-numeric:tabular-nums}
.nums dt{font-size:14px;line-height:1.3;width:7.5em;min-height:2.6em}
.ack{display:inline-flex;align-items:center;gap:10px;min-height:56px;padding:0 24px;border-radius:10px;border:2px solid currentColor;background:rgba(255,255,255,.14);color:inherit;font-size:18px;font-weight:700;cursor:pointer;visibility:hidden}
.ack.on{visibility:visible}
.ack:hover{background:rgba(255,255,255,.26)}
.ack:focus-visible{outline-color:currentColor}

main{display:grid;grid-template-columns:minmax(0,1fr) 400px;grid-template-areas:"stage side" "time side" "ppl side" "help help";gap:16px;padding:16px 24px 32px;max-width:1720px;margin:0 auto}
.stage{grid-area:stage;min-width:0}
.side{grid-area:side;display:flex;flex-direction:column;gap:16px;min-height:0;min-width:0}
.time{grid-area:time}.ppl{grid-area:ppl}.help{grid-area:help}
.panel{background:var(--s1);border:1px solid var(--line);border-radius:var(--r);padding:16px;min-width:0}
.panel h2,.hd h2{font-size:19px;font-weight:650;margin-bottom:12px}

#vwrap{position:relative;background:#000;border-radius:8px;overflow:hidden;aspect-ratio:4/3;margin-inline:auto}
#video{width:100%;height:100%;display:block;object-fit:contain}
#zc{position:absolute;inset:0;width:100%;height:100%;display:none;cursor:crosshair;touch-action:none}
#vwrap.edit #zc{display:block}
body[data-state=alarm] #vwrap{outline:4px solid var(--bad-bg);outline-offset:3px}
.vlost{position:absolute;inset:0;display:none;align-items:center;justify-content:center;text-align:center;padding:24px;background:rgba(11,15,21,.82);font-size:22px;font-weight:650}
body[data-state=lost] .vlost{display:flex}

.tools{display:flex;flex-wrap:wrap;gap:8px 10px;align-items:center;margin-top:12px}
.tools[hidden],.editor[hidden]{display:none}
.zinfo{flex:1 1 240px;color:var(--mute);font-size:14px}
.b{display:inline-flex;align-items:center;justify-content:center;gap:8px;min-height:40px;padding:0 14px;border-radius:8px;border:1px solid var(--line2);background:var(--s2);font-weight:600;cursor:pointer}
.b:hover:not([disabled]){border-color:var(--acc)}
.b.pri{background:var(--pri);border-color:var(--pri);color:#fff}
.b.pri:hover:not([disabled]){background:#2f6fe6}
.b.dng{color:#ffc2bc;border-color:#7a2a2a}
.b[disabled]{opacity:.45;cursor:not-allowed}
.b[hidden]{display:none}
#zmsg{font-size:14px;min-height:1.4em}#zmsg.ok{color:var(--safe)}#zmsg.err{color:var(--bad)}

.editor{margin-top:12px;display:grid;gap:12px;padding:16px;background:var(--s1);border:1px solid var(--line2);border-radius:var(--r)}
.side>.editor{margin-top:0}
.steps{display:flex;flex-wrap:wrap;gap:8px}
.steps li{display:flex;align-items:center;gap:8px;padding:4px 12px 4px 6px;border-radius:999px;border:1px solid var(--line);color:var(--mute);font-size:14px}
.steps li i{display:grid;place-items:center;width:22px;height:22px;border-radius:50%;background:var(--s2);font-style:normal;font-size:13px;font-weight:700}
.steps li[aria-current=step]{border-color:var(--acc);color:var(--tx)}
.steps li[aria-current=step] i{background:var(--pri);color:#fff}
.hint{color:var(--mute);font-size:14px}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.opt{display:flex;flex-wrap:wrap;gap:8px 20px;align-items:center;font-size:14px}
.opt label{display:inline-flex;align-items:center;gap:8px;min-height:32px;cursor:pointer}
.opt input[type=text]{background:var(--bg);border:1px solid var(--line2);border-radius:6px;padding:6px 10px;width:170px}
fieldset{margin:0;padding:12px 14px;border:1px solid var(--line2);border-radius:8px;display:grid;gap:4px}
legend{padding:0 6px;font-weight:650;font-size:14px}
fieldset label{display:flex;align-items:flex-start;gap:10px;min-height:32px;cursor:pointer;font-size:14px}
fieldset[hidden],#zSaveBox[hidden]{display:none}
#zstat{color:var(--dim);font-size:13px}

.hd{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:4px 12px}
.hd .age{color:var(--mute);font-size:14px;font-variant-numeric:tabular-nums}
.empty{color:var(--mute);font-size:14px}
.evimg{display:block;border:0;padding:0;background:#000;border-radius:8px;overflow:hidden;cursor:zoom-in}
.evimg img{display:block;width:100%;height:100%;object-fit:cover}
.latest .pics{display:grid;grid-template-columns:minmax(0,9fr) minmax(0,16fr);align-items:start;gap:8px;margin-bottom:12px}
.latest .pics .evimg{width:100%;aspect-ratio:3/4}
.latest .pics .evimg.wide{aspect-ratio:4/3}
.latest .pics .only{grid-column:1/-1}.latest .pics .only .evimg{aspect-ratio:4/3}
.cap{display:block;margin-top:4px;font-size:12px;color:var(--mute)}
.ttl{font-size:22px;line-height:1.25;font-weight:700}
.meta{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;margin-top:6px;color:var(--mute);font-size:14px}
.sev{display:inline-block;padding:2px 10px;border-radius:999px;font-size:13px;font-weight:650;background:#1b2a3a;color:#bfdcff}
.sev.medium{background:#3a2d0b;color:#ffdb78}
.sev.high,.sev.critical{background:var(--bad-bg);color:#fff}
.history{flex:1 1 0;min-height:260px;display:flex;flex-direction:column}
.history .scroll{flex:1;min-height:0;overflow:auto;margin:0 -6px;padding:0 6px}
#events li{display:grid;grid-template-columns:auto 1fr;gap:12px;padding:10px 0;border-bottom:1px solid var(--line)}
#events li:last-child{border-bottom:0}
#events .imgs{display:flex;gap:6px}
#events .evimg{height:72px}#events .evimg.p{width:54px}#events .evimg.f{width:96px}
#events .t{font-weight:650;font-size:15px;line-height:1.3}
#events .meta{margin-top:4px;font-size:13px}

#chart{display:block;width:100%;height:140px}
#chart text{fill:var(--dim);font-size:12px;font-family:inherit}
#chart .grid{stroke:var(--line);stroke-width:1}
#chart .s{fill:none;stroke-linejoin:round;stroke-linecap:round}
#chart .s-p{stroke:var(--mute);stroke-width:2}
#chart .s-n{stroke:var(--watch);stroke-width:2;stroke-dasharray:6 4}
#chart .s-v{stroke:var(--bad);stroke-width:3}
#chart .mk{stroke:var(--bad);stroke-width:2;opacity:.75}
#chart .mkh{fill:var(--bad)}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:8px;color:var(--mute);font-size:13px}
.legend svg{width:26px;height:10px;vertical-align:middle;margin-right:6px}

.perf .hd{display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin-bottom:6px}.perf .hd h2{margin:0}
#perfList{display:block}
.pm{display:grid;grid-template-columns:6.6em minmax(0,1fr) 4.6em;align-items:center;gap:0 10px;padding:7px 0;border-top:1px solid var(--line)}
.pm .nm{font-size:14px;color:var(--mute)}
.pm .tr{height:12px;border-radius:6px;background:var(--s2);overflow:hidden}
.pm .fl{display:block;height:100%;width:0;border-radius:6px;background:var(--dim);transition:width .5s,background-color .3s}
.pm[data-lvl=ok] .fl{background:var(--safe)}.pm[data-lvl=warn] .fl{background:var(--watch)}.pm[data-lvl=bad] .fl{background:var(--bad)}
.pm .vl{font-size:17px;font-weight:700;font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.pm[data-lvl=warn] .vl{color:var(--watch)}.pm[data-lvl=bad] .vl{color:var(--bad)}
.pm .sub{grid-column:2/4;margin-top:2px;font-size:12px;line-height:1.3;color:var(--dim)}
.ppl .hd{margin-bottom:10px}.ppl .hd h2{margin:0}
#people{display:flex;flex-wrap:wrap;gap:8px}
.chip{padding:4px 12px;border-radius:999px;font-size:14px;border:1px solid var(--line2);background:var(--s2)}
.chip.ok{border-color:#2a6b57;color:#9df0d2}
.chip.attn{border-color:#8a6a14;color:#ffd978;border-style:dashed}
.chip.bad{border-color:var(--bad-bg);background:#3a1210;color:#ffc9c3;font-weight:650}
.sum{color:var(--mute);font-size:14px}

.help summary{cursor:pointer;font-weight:650;font-size:19px}
.help ul{display:grid;gap:6px;margin-top:12px;color:var(--mute);font-size:14px;max-width:80ch}
.help b{color:var(--tx)}

dialog{border:0;padding:0;background:transparent;max-width:96vw;max-height:94vh;color:inherit}
dialog::backdrop{background:rgba(3,6,10,.88)}
dialog img{display:block;max-width:96vw;max-height:90vh;border-radius:8px}
dialog .x{position:absolute;top:8px;right:8px;min-width:44px;min-height:44px;display:grid;place-items:center;border:0;border-radius:50%;background:rgba(11,15,21,.85);cursor:pointer}

@media(max-width:1100px){main{grid-template-columns:minmax(0,1fr) 340px}}
@media(min-width:961px){#vwrap{max-height:calc(100vh - 310px);min-height:360px}}
/* presentation mode (key P): for a projector -- compact status band, the video as large as the screen allows, the latest
   alarm photo kept beside it, history/chart/people list hidden. Remembered per browser. */
body.present .band{min-height:0;padding:10px 24px}body.present .word{font-size:44px}body.present .sent{margin-top:4px}
body.present .nums dd{font-size:32px}body.present .nums dt{min-height:0}
body.present .time,body.present .ppl,body.present .help,body.present .history,body.present #tools{display:none!important}
body.present .pm{padding:4px 0}body.present .pm .sub,body.present .perf .age{display:none}
body.present main{grid-template-columns:minmax(0,1fr) 340px;grid-template-areas:"stage side";padding:10px 16px;max-width:none}
@media(min-width:961px){body.present #vwrap{max-height:calc(100vh - 225px)}}
@media(max-width:960px){
.bar{padding:10px 16px}
.band{grid-template-columns:1fr;min-height:0;padding:16px}
.word{font-size:40px}.sent{font-size:18px;height:3.9em;-webkit-line-clamp:3}
.nums{gap:24px}.nums dd{font-size:32px}
.ack{width:100%;justify-content:center}
main{display:flex;flex-direction:column;padding:12px 16px 28px}
.side{display:contents}
.stage{order:1}.latest{order:2}.perf{order:3}.time{order:4}.history{order:5;flex:none;min-height:0}.ppl{order:6}.help{order:7}
.history .scroll{max-height:60vh}
.latest.none{display:none}
.b,.ib{min-height:44px}.opt label,fieldset label{min-height:44px}
}
@media(max-width:600px){.fps{display:none}#fs span,#pres span{display:none}.more .sfx{display:none}.more{margin-left:12px;font-size:18px;padding:0 12px}}
@media(pointer:coarse){.b,.ib{min-height:44px}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head><body data-state="connecting">
<svg width="0" height="0" style="position:absolute" aria-hidden="true" focusable="false"><defs>
<symbol id="i-pen" viewBox="0 0 24 24"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 013 3L7 19l-4 1 1-4z"/></symbol>
<symbol id="i-undo" viewBox="0 0 24 24"><polyline points="9 14 4 9 9 4"/><path d="M20 20v-7a4 4 0 00-4-4H4"/></symbol>
<symbol id="i-check" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></symbol>
<symbol id="i-save" viewBox="0 0 24 24"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></symbol>
<symbol id="i-reset" viewBox="0 0 24 24"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 102.13-9.36L1 10"/></symbol>
<symbol id="i-vol" viewBox="0 0 24 24"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 010 7.07"/><path d="M19.07 4.93a10 10 0 010 14.14"/></symbol>
<symbol id="i-mute" viewBox="0 0 24 24"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></symbol>
<symbol id="i-full" viewBox="0 0 24 24"><path d="M8 3H5a2 2 0 00-2 2v3m18 0V5a2 2 0 00-2-2h-3m0 18h3a2 2 0 002-2v-3M3 16v3a2 2 0 002 2h3"/></symbol>
<symbol id="i-x" viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></symbol>
<symbol id="i-eye" viewBox="0 0 24 24"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></symbol>
</defs></svg>

<header class="bar">
<h1>Giám sát an toàn phòng lab</h1>
<span class="live" id="live"><i class="dot"></i><span id="liveTxt">đang kết nối…</span></span>
<span class="fps" id="fps">-- FPS</span>
<button class="ib" id="sound" type="button" aria-pressed="false" title="Phím tắt M"><svg class="ic"><use href="#i-mute"/></svg><span id="soundTxt">Âm báo: tắt</span></button>
<button class="ib" id="pres" type="button" aria-pressed="false" title="Phím tắt P"><svg class="ic"><use href="#i-eye"/></svg><span>Trình chiếu</span></button>
<button class="ib" id="fs" type="button" title="Phím tắt F" aria-label="Toàn màn hình"><svg class="ic"><use href="#i-full"/></svg><span>Toàn màn hình</span></button>
</header>

<section class="band" id="band" aria-label="Trạng thái an toàn">
<div><div class="word"><span id="word">ĐANG KẾT NỐI</span><span class="more" id="more" hidden></span></div><p class="sent" id="sent" aria-live="polite" aria-atomic="true">Đang chờ dữ liệu từ camera.</p></div>
<dl class="nums">
<div><dt>Người trong khung</dt><dd id="c_persons">-</dd></div>
<div><dt>Không mặc áo lab</dt><dd id="c_not">-</dd></div>
<div><dt>Đang vi phạm</dt><dd id="c_viol">-</dd></div>
</dl>
<button class="ack" id="ack" type="button" title="Phím tắt A"><svg class="ic"><use href="#i-eye"/></svg>Đã xem</button>
</section>
<div class="sr" id="ann" role="alert" aria-live="assertive" aria-atomic="true"></div>

<main>
<section class="stage" aria-label="Camera">
<div id="vwrap"><img id="video" src="/stream" alt="Hình camera trực tiếp trong phòng lab, có khung quanh từng người được nhận diện"><canvas id="zc" aria-label="Vùng vẽ vùng nguy hiểm trên hình camera"></canvas>
<div class="vlost" id="vlost">Mất tín hiệu camera</div></div>
<div class="tools" id="tools" hidden>
<button class="b" id="zEdit" type="button"><svg class="ic"><use href="#i-pen"/></svg>Vẽ lại vùng nguy hiểm</button>
<button class="b" id="zReset" type="button"><svg class="ic"><use href="#i-reset"/></svg><span>Về vùng mặc định</span></button>
<button class="b" id="zRestore" type="button" hidden><svg class="ic"><use href="#i-undo"/></svg>Khôi phục vùng trước</button>
<span class="zinfo" id="zinfo"></span><span id="zmsg" role="status" aria-live="polite"></span>
</div>
<div class="editor" id="editor" hidden>
<ol class="steps" id="steps"><li data-s="1"><i>1</i>Vẽ viền vùng</li><li data-s="2"><i>2</i>Chốt vùng</li><li data-s="3"><i>3</i>Lưu và áp dụng</li></ol>
<p class="hint" id="zHint" tabindex="-1"></p>
<div class="opt"><label>Tên vùng <input type="text" id="zName" value="san_phong" maxlength="40"></label>
<label><input type="checkbox" id="zCoat" checked> Phải mặc áo lab mới được vào</label></div>
<div class="row"><button class="b" id="zUndo" type="button"><svg class="ic"><use href="#i-undo"/></svg>Bỏ điểm cuối</button>
<button class="b pri" id="zDone" type="button" hidden><svg class="ic"><use href="#i-check"/></svg>Chốt vùng này</button>
<button class="b dng" id="zClear" type="button">Làm lại từ đầu</button>
<button class="b" id="zCancel" type="button">Hủy <kbd>Esc</kbd></button></div>
<div id="zSaveBox" hidden>
<fieldset id="zOld" hidden><legend id="zOldLeg">Vùng cũ đang hiệu lực</legend>
<label><input type="radio" name="zold" value="keep"> Giữ vùng cũ và thêm vùng mới</label>
<label><input type="radio" name="zold" value="replace"> Xóa vùng cũ, chỉ dùng vùng mới vẽ</label></fieldset>
<div class="row" style="margin-top:10px"><button class="b pri" id="zSave" type="button"><svg class="ic"><use href="#i-save"/></svg>Lưu và áp dụng</button></div>
</div>
<div id="zstat"></div>
</div>
</section>

<aside class="side">
<section class="panel latest none" id="latestBox" aria-label="Cảnh báo mới nhất">
<div class="hd"><h2>Cảnh báo mới nhất</h2><span class="age" id="latestAge"></span></div>
<div id="latest"><p class="empty">Cảnh báo mới nhất sẽ hiện ở đây, kèm ảnh người vi phạm và ảnh toàn khung hình.</p></div>
</section>
<section class="panel perf" aria-label="Hiệu năng hệ thống">
<div class="hd"><h2>Hiệu năng hệ thống</h2><span class="age">cập nhật mỗi giây</span></div>
<ul id="perfList"></ul>
</section>
<section class="panel history" aria-label="Lịch sử cảnh báo">
<div class="hd"><h2>Lịch sử cảnh báo</h2><span class="age" id="evCount"></span></div>
<div class="scroll"><ul id="events"><li class="empty" style="display:block">Đang tải…</li></ul></div>
</section>
</aside>

<section class="panel time"><h2>15 phút gần nhất</h2><svg id="chart" role="img" aria-label="Biểu đồ số người, người không mặc áo lab và vi phạm trong 15 phút gần nhất"></svg>
<div class="legend"><span><svg viewBox="0 0 26 10"><line x1="1" y1="5" x2="25" y2="5" stroke="#a3b3c4" stroke-width="2"/></svg>Số người</span><span><svg viewBox="0 0 26 10"><line x1="1" y1="5" x2="25" y2="5" stroke="#f5b82e" stroke-width="2" stroke-dasharray="6 4"/></svg>Không mặc áo lab</span><span><svg viewBox="0 0 26 10"><line x1="1" y1="5" x2="25" y2="5" stroke="#ff7a70" stroke-width="3"/></svg>Vi phạm</span><span><svg viewBox="0 0 26 10"><line x1="13" y1="0" x2="13" y2="10" stroke="#ff7a70" stroke-width="2"/></svg>Lúc có cảnh báo</span></div></section>

<section class="panel ppl"><div class="hd"><h2>Từng người</h2><span class="sum" id="pplsum"></span></div>
<ul id="people"><li class="empty">Chưa có ai trong khung hình</li></ul></section>

<details class="panel help"><summary>Cách đọc trang này</summary><ul>
<li><b>AN TOÀN</b>: camera đang gửi hình và không có vi phạm. <b>CHÚ Ý</b>: có người không mặc áo lab (chưa trong vùng nguy hiểm), hoặc vừa có cảnh báo chưa được xác nhận. <b>BÁO ĐỘNG</b>: có người không mặc áo lab đang ở trong vùng nguy hiểm. <b>MẤT TÍN HIỆU</b>: không nhận được hình mới, nên trang không báo an toàn.</li>
<li><b>Đã xem</b> chỉ tắt chuông và làm dịu màu trên trang này; trạng thái vẫn hiện cho đến khi tình huống hết. Nó không lưu lại và không gửi gì về board.</li>
<li><b>Hiệu năng hệ thống</b>: thanh càng dài càng nhiều. Xanh là bình thường, vàng là cao, đỏ là quá cao (chữ ghi rõ mức). NPU là phần thời gian chip AI bận; nếu board không cho đọc bộ đếm của driver thì số này là ước tính từ thời gian suy luận và trang ghi rõ như vậy. Nhiệt độ là của chip SoC.</li>
<li>Mỗi cảnh báo có hai ảnh: người vi phạm (viền đỏ) và toàn khung hình lúc đó. Bấm vào ảnh để phóng to.</li>
<li>Phím tắt: <kbd>A</kbd> đã xem, <kbd>P</kbd> chế độ trình chiếu (video lớn), <kbd>F</kbd> toàn màn hình, <kbd>M</kbd> bật tắt âm báo, <kbd>Esc</kbd> hủy vẽ vùng hoặc đóng ảnh. Âm báo chỉ kêu sau lần bấm đầu tiên vào trang (quy định của trình duyệt).</li>
<li>Vùng nguy hiểm được vẽ bằng chuột trên hình camera. Khi camera đổi chỗ, vùng phải vẽ lại theo ảnh mới.</li></ul></details>
</main>

<dialog id="zoom" aria-label="Ảnh cảnh báo phóng to"><button class="x" id="zoomX" type="button" aria-label="Đóng ảnh"><svg class="ic"><use href="#i-x"/></svg></button><img alt="" src="data:image/gif;base64,R0lGODlhAQABAAAAACwAAAAAAQABAAA="></dialog>

<script>
(()=>{
const $=id=>document.getElementById(id);
const KIND=%KIND%;
const SEV={low:'Thấp',medium:'Trung bình',high:'Cao',critical:'Nghiêm trọng'};
const FALLS={falling:'đang ngã',down:'đã ngã'};
const PERSON_KINDS=['zone_ppe_violation','zone_intrusion','ppe_violation','fall'];
const ZNAME={san_phong:'Sàn phòng'};
const zname=n=>ZNAME[n]||String(n==null?'':n).replace(/_/g,' ');
const h=(tag,cls,txt)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(txt!=null)e.textContent=txt;return e};
const WORD={connecting:'ĐANG KẾT NỐI',safe:'AN TOÀN',watch:'CHÚ Ý',alarm:'BÁO ĐỘNG',ackd:'BÁO ĐỘNG',lost:'MẤT TÍN HIỆU'};
const FAV={connecting:'#7d90a4',safe:'#35d6b0',watch:'#f5b82e',alarm:'#ff4a3d',ackd:'#ff7a70',lost:'#7d90a4'};
let skew=0,lastId=0,firstEv=true,ctx=null,lastBeep=0,state='',lastKey='',tl=null;
let acked=new Set(),latch=null,evCache=[],curAct=[];
let sound=true;try{const v=localStorage.getItem('labsafe.sound');if(v!==null)sound=v==='1'}catch(e){}
let pres=false;try{pres=localStorage.getItem('labsafe.present')==='1'}catch(e){}
if(location.hash==='#present')pres=true;   // http://localhost:8000/#present opens straight in presentation mode (make demo-day)

function ago(s){s=Math.max(0,Math.round(s));if(s<60)return s+' giây trước';if(s<3600)return Math.floor(s/60)+' phút trước';return Math.floor(s/3600)+' giờ trước'}
function clock(t){return new Date((t+skew)*1000).toLocaleTimeString('vi-VN')}
function nowSrv(){return Date.now()/1000-skew}
const sig=e=>e.kind+'|'+(e.track_id==null?'':e.track_id)+'|'+(e.detail||'');
function describe(e){const bits=[];if(PERSON_KINDS.includes(e.kind)&&e.track_id!=null)bits.push('người #'+e.track_id);if(e.detail)bits.push(zname(e.detail));return{title:KIND[e.kind]||e.kind,where:bits.join(' · ')}}

/* ---- sound ---- */
function soundUi(){$('sound').setAttribute('aria-pressed',sound?'true':'false');$('soundTxt').textContent='Âm báo: '+(sound?'bật':'tắt');$('sound').querySelector('use').setAttribute('href',sound?'#i-vol':'#i-mute')}
function beep(){if(!sound)return;lastBeep=Date.now();try{ctx=ctx||new AudioContext();if(ctx.state==='suspended')ctx.resume();const o=ctx.createOscillator(),g=ctx.createGain();o.frequency.value=880;g.gain.value=.25;o.connect(g);g.connect(ctx.destination);o.start();o.stop(ctx.currentTime+.4)}catch(e){}}
function toggleSound(){sound=!sound;try{localStorage.setItem('labsafe.sound',sound?'1':'0')}catch(e){}soundUi();if(sound)beep()}

/* ---- the state band ---- */
function setState(id,strong,rest,more){
  const key=id+'|'+strong+'|'+rest+'|'+(more||0);
  const mo=$('more');mo.hidden=!more;mo.textContent='';if(more){mo.appendChild(h('span',null,'+'+more));mo.appendChild(h('span','sfx',' cảnh báo khác'));mo.title='Còn '+more+' cảnh báo khác';mo.setAttribute('aria-label','và '+more+' cảnh báo khác')}
  if(id!==state){state=id;document.body.dataset.state=id;$('word').textContent=WORD[id];document.title=(id==='alarm'||id==='ackd'?'⚠ BÁO ĐỘNG · ':id==='watch'?'Chú ý · ':id==='lost'?'Mất tín hiệu · ':'')+'Giám sát an toàn phòng lab';
    $('fav').setAttribute('href',"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ccircle cx='16' cy='16' r='13' fill='"+encodeURIComponent(FAV[id])+"'/%3E%3C/svg%3E")}
  if(key!==lastKey){lastKey=key;const s=$('sent');s.textContent='';s.title=strong+(rest?' · '+rest:'');if(strong)s.appendChild(h('b',null,strong));if(rest)s.appendChild(document.createTextNode((strong?' · ':'')+rest));
    $('ann').textContent=id==='alarm'?'Báo động: '+strong+(rest?' · '+rest:'')+(more?' · và '+more+' cảnh báo khác':''):id==='lost'?'Mất tín hiệu camera. '+(rest||''):''}
  const showAck=id==='alarm'||(id==='watch'&&!!latch);
  $('ack').classList.toggle('on',showAck);$('ack').tabIndex=showAck?0:-1;
}
function setNums(s){const f=v=>v==null?'–':v;$('c_persons').textContent=f(s.persons);$('c_not').textContent=f(s.not_wearing);$('c_viol').textContent=f(s.violating)}
function lost(msg){perf(null);$('live').classList.remove('on');$('liveTxt').textContent='mất tín hiệu';$('fps').textContent='-- FPS';setNums({});renderPeople(null);setState('lost','Không nhận được hình mới từ camera',msg)}
function doAck(){curAct.forEach(e=>acked.add(sig(e)));latch=null;poll()}

async function status(){
  let j;try{const r=await fetch('/status',{cache:'no-store'});if(!r.ok)throw 0;j=await r.json()}catch(e){lost('Mất kết nối với dashboard trên board.');return false}
  skew=Date.now()/1000-j.server_now;$('fps').textContent=(j.fps||0).toFixed(1)+' FPS';
  const age=j.frame_age_s;
  if(!(age<3)){lost(age>1e8?'Chưa nhận được khung hình nào.':'Khung hình cuối nhận được '+ago(age)+'.');return true}
  $('live').classList.add('on');$('liveTxt').textContent='đang chạy';perf(j);
  const s=j.summary||{};setNums(s);renderPeople(s.people||[],s);
  const act=(j.events||[]).filter(e=>e.kind in KIND);curAct=act;
  const cur=new Set(act.map(sig));acked=new Set([...acked].filter(x=>cur.has(x)));
  const open=act.filter(e=>!acked.has(sig(e)));
  if(open.length){
    const d=describe(open[0]);latch={d};
    setState('alarm',d.title,d.where,open.length-1);
    if(sound&&Date.now()-lastBeep>5000)beep();
  }else if(act.length){
    const d=describe(act[0]);setState('ackd',d.title,(d.where?d.where+' · ':'')+'đã xem, vẫn đang diễn ra',act.length-1);
  }else if(latch){
    setState('watch','Vừa có cảnh báo, đã dừng',latch.d.title+(latch.d.where?' · '+latch.d.where:''));
  }else if((s.violating||0)>0){
    setState('watch',s.violating+' người đang vi phạm','Đang xác nhận trước khi báo động.');
  }else if((s.not_wearing||0)>0){
    setState('watch',s.not_wearing+' người không mặc áo lab','Chưa ở trong vùng nguy hiểm.');
  }else setState('safe','Không có vi phạm',(s.persons||0)?'':'Phòng không có người trong khung.');
  return true
}
function poll(){status()}

/* ---- performance gauges: bar = share of the scale, colour + word = how it is doing ---- */
const PM=[
 {id:'fps',nm:'Tốc độ',min:0,max:30,get:(j)=>j.fps,fmt:v=>v.toFixed(1)+' FPS',lvl:v=>v>=24?0:v>=15?1:2,words:['Mượt','Chậm','Rất chậm'],sub:v=>Math.round(100*v/30)+'% tốc độ camera (30 FPS)'},
 {id:'lat',nm:'Độ trễ',min:0,max:66,get:(j,y)=>y.infer_ms,fmt:v=>v.toFixed(0)+' ms',lvl:v=>v<=40?0:v<=60?1:2,words:['Nhanh','Chậm','Rất chậm'],sub:()=>'tầng A, mỗi khung hình'},
 {id:'npu',nm:'NPU',min:0,max:100,get:(j,y)=>y.npu_pct,fmt:v=>Math.round(v)+'%',lvl:v=>v<90?0:v<98?1:2,words:['Còn dư sức','Gần kín','Kín tải'],sub:(v,y)=>(y.npu_src==='kernel'?'đọc từ driver NPU':'ước tính từ thời gian suy luận')+(y.npu_mhz?' · xung '+y.npu_mhz+' MHz':'')},
 {id:'cpu',nm:'CPU',min:0,max:100,get:(j,y)=>y.cpu_pct,fmt:v=>Math.round(v)+'%',lvl:v=>v<70?0:v<90?1:2,words:['Thoải mái','Cao','Rất cao'],sub:()=>'trung bình mọi nhân'},
 {id:'ram',nm:'RAM',min:0,max:100,get:(j,y)=>y.ram_pct,fmt:v=>Math.round(v)+'%',lvl:v=>v<70?0:v<85?1:2,words:['Dư nhiều','Cao','Sắp hết'],sub:(v,y)=>(y.ram_mb!=null?y.ram_mb+' MB đang dùng':'')+(y.rss_mb!=null?' · app '+y.rss_mb+' MB':'')},
 {id:'tmp',nm:'Nhiệt độ',min:30,max:90,get:(j,y)=>y.temp_c,fmt:v=>v.toFixed(1)+' °C',lvl:v=>v<65?0:v<80?1:2,words:['Mát','Ấm','Nóng'],sub:()=>'chip SoC; thang đo 30–90 °C'}];
const pmRow={};
PM.forEach(m=>{const li=h('li','pm');li.dataset.lvl='none';const tr=h('span','tr'),fl=h('i','fl');tr.appendChild(fl);
  const nm=h('span','nm',m.nm),vl=h('span','vl','–'),sub=h('span','sub','Chưa có dữ liệu');
  li.appendChild(nm);li.appendChild(tr);li.appendChild(vl);li.appendChild(sub);$('perfList').appendChild(li);pmRow[m.id]={li,fl,vl,sub}});
function perf(j){
  const y=(j&&j.sys)||{};
  PM.forEach(m=>{const r=pmRow[m.id];let v=j?m.get(j,y):null;
    if(v==null||isNaN(v)){r.li.dataset.lvl='none';r.fl.style.width='0';r.vl.textContent='–';r.sub.textContent=j?'Chưa có dữ liệu':'Mất tín hiệu';return}
    const l=m.lvl(v);r.li.dataset.lvl=['ok','warn','bad'][l];
    r.fl.style.width=Math.max(2,Math.min(100,100*(v-m.min)/(m.max-m.min)))+'%';
    r.vl.textContent=m.fmt(v);r.sub.textContent=m.words[l]+' · '+m.sub(v,y)})
}

/* ---- people ---- */
function renderPeople(list,s){
  const ul=$('people');ul.textContent='';
  $('pplsum').textContent=s?(s.wearing||0)+' mặc áo lab · '+(s.in_zone||0)+' trong vùng nguy hiểm':'';
  if(!list){ul.appendChild(h('li','empty','Không có dữ liệu vì mất tín hiệu camera'));return}
  if(!list.length){ul.appendChild(h('li','empty','Chưa có ai trong khung hình'));return}
  list.forEach(p=>{const c=h('li','chip '+(p.violating?'bad':(p.wearing?'ok':(p.unknown?'':'attn'))));
    const fall=FALLS[p.fall];
    c.textContent='#'+p.id+' · '+(p.wearing?'mặc áo lab':(p.unknown?'chưa thấy rõ áo':'Không mặc áo lab'))+((p.zones||[]).length?' · trong vùng':'')+(fall?' · '+fall:'');ul.appendChild(c)})
}

/* ---- zoom ---- */
const zoom=$('zoom');
function openZoom(src,alt){const im=zoom.querySelector('img');im.src=src;im.alt=alt||'';if(!zoom.open)zoom.showModal()}
zoom.addEventListener('click',e=>{if(e.target===zoom)zoom.close()});$('zoomX').onclick=()=>zoom.close();
function evImg(src,alt,cls){const b=h('button','evimg '+cls);b.type='button';b.title='Phóng to';const im=h('img');im.loading='lazy';im.src=src;im.alt=alt;b.appendChild(im);b.onclick=()=>openZoom(src,alt);return b}

/* ---- alarms: latest card + history ---- */
function sevChip(e){return h('span','sev '+(e.severity||''),SEV[e.severity]||e.severity||'')}
function ageNode(e){const sp=h('span','ageof');sp.dataset.t=e.t;return sp}
function tickAges(){document.querySelectorAll('.ageof').forEach(sp=>{sp.textContent=clock(+sp.dataset.t)+' · '+ago(nowSrv()-sp.dataset.t)});
  const la=$('latestAge');if(la.dataset.t)la.textContent=ago(nowSrv()-la.dataset.t)}
function showLatest(e){
  const d=describe(e),box=$('latest');box.textContent='';$('latestBox').classList.remove('none');
  const pics=h('div','pics');
  if(e.has_crop){const f=h('div');f.appendChild(evImg('/crop/'+e.id,'Ảnh cắt người liên quan đến cảnh báo '+e.id,'',1));f.appendChild(h('span','cap','Người vi phạm'));pics.appendChild(f)}
  const g=h('div');g.appendChild(evImg('/snap/'+e.id,'Ảnh toàn khung hình lúc cảnh báo '+e.id,e.has_crop?'wide':''));g.appendChild(h('span','cap','Toàn khung hình'));if(!e.has_crop)g.className='only';pics.appendChild(g);
  box.appendChild(pics);box.appendChild(h('div','ttl',d.title));
  const m=h('div','meta');m.appendChild(sevChip(e));if(d.where)m.appendChild(h('span',null,d.where));m.appendChild(ageNode(e));box.appendChild(m);
  $('latestAge').dataset.t=e.t;tickAges();
}
function addRow(e){
  const ul=$('events');ul.querySelector('.empty')?.remove();
  const d=describe(e),li=h('li');const imgs=h('div','imgs');
  if(e.has_crop)imgs.appendChild(evImg('/crop/'+e.id,'Ảnh cắt người, cảnh báo '+e.id,'p'));
  imgs.appendChild(evImg('/snap/'+e.id,'Ảnh toàn khung hình, cảnh báo '+e.id,'f'));
  const body=h('div');body.appendChild(h('div','t',d.title));
  const m=h('div','meta');m.appendChild(sevChip(e));if(d.where)m.appendChild(h('span',null,d.where));m.appendChild(ageNode(e));body.appendChild(m);
  li.appendChild(imgs);li.appendChild(body);ul.prepend(li);
  while(ul.children.length>80)ul.lastElementChild.remove();
}
async function events(){
  let j;try{const r=await fetch('/events?since='+lastId,{cache:'no-store'});if(!r.ok)throw 0;j=await r.json()}catch(e){return false}
  const wasFirst=firstEv;firstEv=false;if(wasFirst){const em=document.querySelector('#events .empty');if(em)em.textContent='Chưa có cảnh báo nào.'}
  const ev=(j.events||[]).filter(e=>e.id>lastId).sort((a,b)=>a.id-b.id);
  if(!ev.length)return true;
  ev.forEach(e=>{evCache.push({t:e.t,kind:e.kind});addRow(e)});
  evCache=evCache.slice(-300);
  const newest=ev[ev.length-1];showLatest(newest);
  $('evCount').textContent=$('events').children.length+' gần nhất';
  if(!wasFirst){const fresh=ev.filter(e=>e.kind in KIND);if(fresh.length){latch={d:describe(fresh[fresh.length-1])};if(state!=='ackd')beep();poll()}}
  lastId=newest.id;tickAges();return true;
}

/* ---- timeline ---- */
function drawChart(){
  const j=tl;const svg=$('chart');const W=Math.max(300,svg.clientWidth||600),H=140,L=30,R=8,T=10,B=24;svg.setAttribute('viewBox','0 0 '+W+' '+H);
  if(!j||j.samples.length<2){svg.innerHTML='<text x="'+L+'" y="'+(H/2)+'">Chưa có dữ liệu</text>';return}
  const d=j.samples,t0=j.server_now-900,mx=Math.max(4,...d.map(r=>r[1]));
  const X=t=>(L+(t-t0)/900*(W-L-R)).toFixed(1),Y=v=>(H-B-v/mx*(H-B-T)).toFixed(1);
  let o='<line class="grid" x1="'+L+'" x2="'+(W-R)+'" y1="'+Y(0)+'" y2="'+Y(0)+'"/><line class="grid" x1="'+L+'" x2="'+(W-R)+'" y1="'+Y(mx)+'" y2="'+Y(mx)+'"/>'
   +'<text x="'+(L-6)+'" y="'+(+Y(mx)+4)+'" text-anchor="end">'+mx+'</text><text x="'+(L-6)+'" y="'+(+Y(0)+4)+'" text-anchor="end">0</text>';
  [['−15 phút',0],['−10',300],['−5',600],['nay',900]].forEach(a=>{o+='<text x="'+X(t0+a[1])+'" y="'+(H-6)+'" text-anchor="'+(a[1]===0?'start':a[1]===900?'end':'middle')+'">'+a[0]+'</text>'});
  evCache.filter(e=>e.kind in KIND&&e.t>=t0).forEach(e=>{const x=X(e.t);o+='<line class="mk" x1="'+x+'" x2="'+x+'" y1="'+T+'" y2="'+Y(0)+'"/><circle class="mkh" cx="'+x+'" cy="'+T+'" r="3"/>'});
  const line=(i,c)=>'<polyline class="s '+c+'" points="'+d.map(r=>X(r[0])+','+Y(r[i])).join(' ')+'"/>';
  svg.innerHTML=o+line(1,'s-p')+line(2,'s-n')+line(3,'s-v');
}
async function timeline(){try{const r=await fetch('/timeline',{cache:'no-store'});if(!r.ok)throw 0;tl=await r.json();drawChart();return true}catch(e){return false}}

/* ---- zone editor: draw on the live video, save, applied at once on the running system ---- */
let zEditing=false,zDraft=[],zCur=[],zAllowed=[],zLive=[],zPrev=[],zResetTimer=0;
const zc=$('zc'),zg=zc.getContext('2d'),vid=$('video');
vid.addEventListener('load',()=>{if(vid.naturalWidth)$('vwrap').style.aspectRatio=vid.naturalWidth+'/'+vid.naturalHeight;if(zEditing)zDraw()});
function frameRect(){const r=vid.getBoundingClientRect(),nw=vid.naturalWidth||640,nh=vid.naturalHeight||480,s=Math.min(r.width/nw,r.height/nh),w=nw*s,hh=nh*s;return{W:r.width,H:r.height,ox:(r.width-w)/2,oy:(r.height-hh)/2,w,h:hh}}
function keepOld(){return document.querySelector('input[name=zold]:checked')?.value==='keep'}
function zDraw(){const f=frameRect();zc.width=f.W;zc.height=f.H;zg.clearRect(0,0,f.W,f.H);
  const poly=(pts,col,close)=>{if(!pts.length)return;zg.beginPath();pts.forEach((p,i)=>{const x=f.ox+p[0]*f.w,y=f.oy+p[1]*f.h;i?zg.lineTo(x,y):zg.moveTo(x,y)});if(close)zg.closePath();zg.lineWidth=3;zg.strokeStyle=col;zg.stroke();if(close){zg.fillStyle=col+'33';zg.fill()}
    pts.forEach(p=>{zg.beginPath();zg.arc(f.ox+p[0]*f.w,f.oy+p[1]*f.h,5,0,7);zg.fillStyle=col;zg.fill()})};
  const keep=keepOld(),col=keep?'#ffd24d':'#ff7a7a';
  zLive.forEach(z=>{zg.save();zg.setLineDash([8,6]);poly(z.polygon,col,false);zg.restore();
    const p0=z.polygon[0];zg.font='bold 14px system-ui,sans-serif';const lb=(keep?'Vùng cũ (giữ): ':'Vùng cũ: ')+zname(z.name),lx=f.ox+p0[0]*f.w+6,ly=f.oy+p0[1]*f.h-8,lw=zg.measureText(lb).width;zg.fillStyle='rgba(11,15,21,.88)';zg.fillRect(lx-5,ly-15,lw+10,22);zg.fillStyle=col;zg.fillText(lb,lx,ly)});
  zDraft.forEach(z=>poly(z.polygon,'#ff5ccf',true));poly(zCur,'#58ff9a',false);
  zStep()}
function zStep(){
  const pts=zCur.length,dr=zDraft.length,step=dr?3:(pts>=3?2:1),canSave=dr>0||pts>=3;
  document.querySelectorAll('#steps li').forEach(li=>{if(+li.dataset.s===step)li.setAttribute('aria-current','step');else li.removeAttribute('aria-current')});
  $('zDone').hidden=pts<3;$('zUndo').disabled=pts===0;$('zSaveBox').hidden=!canSave;
  const needChoice=canSave&&zLive.length>0;$('zOld').hidden=!needChoice;
  $('zOldLeg').textContent='Vùng cũ đang hiệu lực: '+(zLive.map(z=>zname(z.name)).join(', ')||'(không có)');
  $('zSave').disabled=needChoice&&!document.querySelector('input[name=zold]:checked');
  $('zHint').textContent=step===1?'Bấm lần lượt các điểm quanh vùng sàn cần giám sát, ít nhất 3 điểm. Vùng cũ (nét đứt trên video) vẫn đang hiệu lực cho đến khi bạn lưu.'
    :step===2?'Đủ điểm. Bấm "Chốt vùng này" để khép kín vùng, hoặc bấm thêm điểm.'
    :'Đã chốt '+dr+' vùng mới'+(pts?' và đang vẽ '+pts+' điểm':'')+'. '+(zLive.length?'Chọn cách xử lý vùng cũ rồi bấm "Lưu và áp dụng".':'Bấm "Lưu và áp dụng".')+' Muốn thêm vùng khác, bấm tiếp trên video.';
  $('zstat').textContent='Đã chốt: '+dr+' vùng mới'+(zCur.length?' · đang vẽ '+zCur.length+' điểm':'')}
function zMsg(t,c){const m=$('zmsg');m.textContent=t||'';m.className=c||''}
function zInfo(){$('zinfo').textContent=zLive.length?'Vùng đang áp dụng: '+zLive.map(z=>zname(z.name)+((z.requires||[]).includes('lab_coat')?' (cần áo lab)':' (cấm vào)')).join(', '):'Chưa có vùng nguy hiểm nào.'}
const wide=matchMedia('(min-width:961px)');
function zSetEdit(on){zEditing=on;$('vwrap').classList.toggle('edit',on);
  if(on&&wide.matches)document.querySelector('.side').prepend($('editor'));else $('tools').after($('editor'));
  $('editor').hidden=!on;
  ['zEdit','zReset'].forEach(i=>$(i).hidden=on);$('zRestore').hidden=on||!zPrev.length;zMsg('');
  if(on){document.querySelectorAll('input[name=zold]').forEach(r=>r.checked=false);zDraw();$('zHint').focus({preventScroll:true})}else{zg.clearRect(0,0,zc.width,zc.height);$('zEdit').focus({preventScroll:true})}}
zc.addEventListener('click',e=>{if(!zEditing)return;const f=frameRect(),r=zc.getBoundingClientRect(),x=(e.clientX-r.left-f.ox)/f.w,y=(e.clientY-r.top-f.oy)/f.h;
  if(x<-.02||x>1.02||y<-.02||y>1.02)return;zCur.push([Math.min(1,Math.max(0,+x.toFixed(4))),Math.min(1,Math.max(0,+y.toFixed(4)))]);zDraw()});
function zFinish(){if(zCur.length<3){zMsg('Cần ít nhất 3 điểm.','err');return false}
  let name=$('zName').value.trim()||('vung_'+(zDraft.length+1));if(zDraft.some(z=>z.name===name))name+='_'+(zDraft.length+1);
  const z={name,severity:'high',polygon:zCur};if($('zCoat').checked&&zAllowed.includes('lab_coat'))z.requires=['lab_coat'];zDraft.push(z);zCur=[];$('zName').value='vung_'+(zDraft.length+1);zDraw();zMsg('Đã chốt '+zDraft.length+' vùng.','ok');return true}
async function zPost(url,body){try{const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json','X-Requested-With':'labsafe'},body:JSON.stringify(body||{})});return[r.ok,await r.json().catch(()=>({error:'lỗi '+r.status}))]}catch(e){return[false,{error:'Không kết nối được tới board.'}]}}
async function zLoad(){try{const r=await fetch('/zones',{cache:'no-store'});if(!r.ok)return;const j=await r.json();zLive=j.zones;zAllowed=j.allowed_requires;zInfo();$('tools').hidden=false}catch(e){}}
$('zEdit').onclick=async()=>{await zLoad();zDraft=[];zCur=[];$('zName').value='san_phong';zSetEdit(true)};
$('zDone').onclick=zFinish;$('zUndo').onclick=()=>{zCur.pop();zDraw()};$('zClear').onclick=()=>{zDraft=[];zCur=[];zDraw();zMsg('')};
$('zCancel').onclick=()=>zSetEdit(false);document.querySelectorAll('input[name=zold]').forEach(r=>r.onchange=zDraw);
$('zSave').onclick=async()=>{if(zCur.length&&!zFinish())return;if(!zDraft.length)return;
  let all=zDraft.slice();if(keepOld())all=zLive.filter(o=>!zDraft.some(d=>d.name===o.name)).concat(zDraft);
  const prev=zLive.slice(),[ok,j]=await zPost('/zones',{zones:all});
  if(ok){zPrev=prev;zLive=j.zones;zSetEdit(false);zInfo();zMsg('Đã áp dụng '+j.zones.length+' vùng.','ok')}else zMsg(j.error||'Lưu thất bại','err')};
$('zRestore').onclick=async()=>{const back=zPrev.slice(),cur=zLive.slice(),[ok,j]=await zPost('/zones',{zones:back});
  if(ok){zPrev=cur;zLive=j.zones;zInfo();zMsg('Đã khôi phục vùng trước.','ok')}else zMsg(j.error||'Khôi phục thất bại','err')};
$('zReset').onclick=async()=>{const b=$('zReset').querySelector('span');
  if(!zResetTimer){b.textContent='Bấm lại để xác nhận';zResetTimer=setTimeout(()=>{zResetTimer=0;b.textContent='Về vùng mặc định'},4000);return}
  clearTimeout(zResetTimer);zResetTimer=0;b.textContent='Về vùng mặc định';
  const cur=zLive.slice(),[ok,j]=await zPost('/zones/reset');if(ok){zPrev=cur;zLive=j.zones;zInfo();$('zRestore').hidden=!zPrev.length;zMsg('Đã về vùng mặc định.','ok')}else zMsg(j.error||'Thất bại','err')};

/* ---- header controls and keys ---- */
$('sound').onclick=toggleSound;$('ack').onclick=doAck;
/* audio may only start after a click/key on the page: create/resume the context on the first one so the first alarm is heard */
const unlockAudio=()=>{try{ctx=ctx||new AudioContext();if(ctx.state==='suspended')ctx.resume()}catch(e){}};
document.addEventListener('pointerdown',unlockAudio,{once:true});document.addEventListener('keydown',unlockAudio,{once:true});
function presUi(){document.body.classList.toggle('present',pres);$('pres').setAttribute('aria-pressed',pres?'true':'false');drawChart()}
function togglePres(){pres=!pres;try{localStorage.setItem('labsafe.present',pres?'1':'0')}catch(e){}presUi()}
$('pres').onclick=togglePres;
function toggleFs(){if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen?.()}
$('fs').onclick=toggleFs;
document.addEventListener('keydown',e=>{
  if(e.ctrlKey||e.metaKey||e.altKey)return;const tag=(e.target.tagName||'').toLowerCase();
  if(e.key==='Escape'){if(zEditing&&!zoom.open)zSetEdit(false);return}
  if(tag==='input'||tag==='textarea')return;
  const k=e.key.toLowerCase();
  if(zEditing&&(e.key==='Backspace'||k==='z')&&zCur.length){e.preventDefault();zCur.pop();zDraw();return}
  if(k==='a'&&$('ack').classList.contains('on'))doAck();else if(k==='f')toggleFs();else if(k==='m')toggleSound();else if(k==='p')togglePres()});
window.addEventListener('resize',()=>{if(zEditing)zDraw();drawChart()});

/* One failed board connection used to trigger three fixed-rate request loops,
   producing dozens of console/network errors. Back off each endpoint separately,
   never overlap one endpoint with itself, and return to the normal cadence as
   soon as a request succeeds. */
function backoffPoll(fn,base,max){let delay=base,busy=false;const run=async()=>{if(busy)return;busy=true;let ok=false;try{ok=(await fn())!==false}catch(e){}busy=false;delay=ok?base:Math.min(max,Math.max(base*2,delay*2));setTimeout(run,delay)};run()}
soundUi();presUi();zLoad();
backoffPoll(status,500,5000);backoffPoll(events,1000,5000);backoffPoll(timeline,5000,15000);setInterval(tickAges,1000);
})();
</script></body></html>"""


class _State:
    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.jpeg: bytes | None = None
        self.seq = 0
        self.frame_t = 0.0
        self.status: dict = {"severity": "none", "fps": 0.0}
        self.events: deque = deque(maxlen=200)        # alarm history, oldest first
        self.snaps: dict[int, bytes] = {}             # alarm id -> jpeg
        self.crops: dict[int, bytes] = {}             # alarm id -> jpeg of the person
        self.next_id = 1
        self.timeline: deque = deque(maxlen=900)      # (t, persons, not_wearing, violating), 1/s
        self.last_sample = 0.0


class Dashboard:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000, jpeg_quality: int = 70,
                 stream_max_fps: float = 15.0, snapshot_width: int = 480, zones=None):
        """`zones`: optional object with current() -> list[dict], apply(list) -> list[dict], reset() -> list[dict],
        and `allowed` (classes a zone may require). Without it the editor is read-only (/zones answers 404)."""
        self.state = _State()
        zstore = zones
        self.quality = int(jpeg_quality)
        self.snapshot_width = int(snapshot_width)
        state = self.state
        min_dt = 1.0 / max(1.0, stream_max_fps)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep the console usable
                pass

            def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj: dict) -> None:
                obj["server_now"] = time.time()
                self._send(json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

            def do_GET(self):
                url = urlparse(self.path)
                path = url.path
                try:
                    if path == "/status":
                        with state.cond:
                            st = dict(state.status)
                            st["frame_age_s"] = round(time.time() - state.frame_t, 2) if state.frame_t else 1e9
                        self._json(st)
                    elif path == "/events":
                        since = int(parse_qs(url.query).get("since", ["0"])[0] or 0)
                        with state.cond:
                            evs = [e for e in state.events if e["id"] > since]
                        self._json({"events": evs[::-1]})
                    elif path.startswith("/snap/") or path.startswith("/crop/"):
                        try:
                            sid = int(path.rsplit("/", 1)[1].split(".")[0])
                        except ValueError:
                            sid = -1
                        with state.cond:
                            buf = (state.crops if path.startswith("/crop/") else state.snaps).get(sid)
                        if buf is None:
                            self._send(b"not found", "text/plain", 404)
                        else:
                            self._send(buf, "image/jpeg")
                    elif path == "/zones" and zstore is not None:
                        self._json({"zones": zstore.current(), "allowed_requires": list(zstore.allowed)})
                    elif path == "/timeline":
                        with state.cond:
                            samples = list(state.timeline)
                        self._json({"samples": samples})
                    elif path == "/stream":
                        self._stream()
                    elif path in ("/", "/index.html"):
                        page = PAGE.replace("%KIND%", json.dumps(KIND_VI, ensure_ascii=False))
                        self._send(page.encode(), "text/html; charset=utf-8")
                    else:
                        self._send(b"not found", "text/plain", 404)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _post_ok(self) -> bool:
                host = (self.headers.get("Host") or "").split(":")[0]
                ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
                return (host in ("127.0.0.1", "localhost", "[::1]") and ctype == "application/json"
                        and self.headers.get("X-Requested-With") == "labsafe")

            def do_POST(self):
                path = urlparse(self.path).path
                try:
                    if zstore is None or path not in ("/zones", "/zones/reset"):
                        return self._send(b"not found", "text/plain", 404)
                    if not self._post_ok():
                        return self._send(b"forbidden", "text/plain", 403)
                    n = int(self.headers.get("Content-Length") or 0)
                    if n > 65536:
                        return self._send(b"too large", "text/plain", 413)
                    body = self.rfile.read(n) if n else b"{}"
                    try:
                        if path == "/zones/reset":
                            zones = zstore.reset()
                        else:
                            zones = zstore.apply(json.loads(body.decode("utf-8")).get("zones"))
                    except (ValueError, AttributeError, UnicodeDecodeError) as e:   # json errors are ValueErrors
                        return self._send(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False).encode(),
                                          "application/json; charset=utf-8", 400)
                    self._json({"ok": True, "zones": zones})
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _stream(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seen = -1
                while True:
                    # Wait for a NEW frame instead of spinning: a busy loop here burned a core per open tab.
                    with state.cond:
                        state.cond.wait_for(lambda: state.seq != seen, timeout=2.0)
                        buf, seen = state.jpeg, state.seq
                    if buf is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(buf)}\r\n\r\n".encode())
                    self.wfile.write(buf + b"\r\n")
                    time.sleep(min_dt)

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://{host}:{port}/"

    def publish(self, frame: np.ndarray | None, status: dict, alarming: list | None = None,
                crops: list[bytes | None] | None = None) -> None:
        """Called once per processed frame from the main loop. `alarming` = debounced SafetyEvents that fired now;
        `crops` = JPEG of the person for each of them (aligned, None where there is no person photo)."""
        now = time.time()
        # frame=None: update the status / alarms only, keep the last video frame (the caller skips drawing to save CPU)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]) if frame is not None else (False, None)
        snap = None
        if alarming and frame is not None:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (self.snapshot_width, int(h * self.snapshot_width / w))) \
                if w > self.snapshot_width else frame
            ok2, sbuf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            snap = sbuf.tobytes() if ok2 else None
        s = status.get("summary", {})
        with self.state.cond:
            st = self.state
            if ok:
                st.jpeg, st.seq, st.frame_t = buf.tobytes(), st.seq + 1, now
            st.status = status
            for i, ev in enumerate(alarming or []):
                d = ev.as_dict() if hasattr(ev, "as_dict") else dict(ev)
                d["id"] = st.next_id
                st.next_id += 1
                crop = crops[i] if crops and i < len(crops) else None
                d["has_crop"] = crop is not None
                st.events.append(d)
                if snap is not None:
                    st.snaps[d["id"]] = snap
                if crop is not None:
                    st.crops[d["id"]] = crop
            # keep snapshots only for events still in the history
            if len(st.snaps) > st.events.maxlen or len(st.crops) > st.events.maxlen:
                keep = {e["id"] for e in st.events}
                st.snaps = {k: v for k, v in st.snaps.items() if k in keep}
                st.crops = {k: v for k, v in st.crops.items() if k in keep}
            if now - st.last_sample >= 1.0:
                st.last_sample = now
                st.timeline.append((round(now, 1), s.get("persons", 0), s.get("not_wearing", 0), s.get("violating", 0)))
            st.cond.notify_all()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
