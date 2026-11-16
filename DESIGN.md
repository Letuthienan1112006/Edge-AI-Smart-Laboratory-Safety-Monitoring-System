---
name: Giám sát an toàn phòng lab (labsafe dashboard)
description: A dark control-room console where one full-width state band tells a supervisor, from across the room, whether the lab is safe.
colors:
  ink: "#0b0f15"
  panel: "#121923"
  panel-raised: "#1a2431"
  hairline: "#263243"
  hairline-strong: "#3a4a60"
  text: "#eaf0f6"
  text-muted: "#a3b3c4"
  text-dim: "#8da0b4"
  link-blue: "#6db3ff"
  action-blue: "#2563d9"
  safe-field: "#0b3b36"
  safe-text: "#d8fff3"
  safe-signal: "#35d6b0"
  watch-field: "#4a3703"
  watch-text: "#fff1c4"
  watch-signal: "#f5b82e"
  alarm-field: "#b71f1c"
  alarm-text: "#ffffff"
  alarm-signal: "#ff7a70"
  ack-field: "#471616"
  ack-text: "#ffdcd8"
  lost-field: "#2a3441"
  lost-text: "#dbe4ee"
typography:
  state-word:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "56px"
    fontWeight: 800
    lineHeight: 1
    letterSpacing: "0.01em"
  numeral:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "40px"
    fontWeight: 700
    lineHeight: 1
    fontFeature: "tabular-nums"
  sentence:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "24px"
    fontWeight: 400
    lineHeight: 1.3
  headline:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "19px"
    fontWeight: 650
    lineHeight: 1.5
  title:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "22px"
    fontWeight: 700
    lineHeight: 1.25
  body:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
  caption:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.5
rounded:
  kbd: "5px"
  input: "6px"
  control: "8px"
  panel: "10px"
  pill: "999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
  band-gap: "32px"
components:
  state-band:
    backgroundColor: "{colors.lost-field}"
    textColor: "{colors.lost-text}"
    padding: "20px 24px"
    height: "136px"
  state-band-safe:
    backgroundColor: "{colors.safe-field}"
    textColor: "{colors.safe-text}"
  state-band-watch:
    backgroundColor: "{colors.watch-field}"
    textColor: "{colors.watch-text}"
  state-band-alarm:
    backgroundColor: "{colors.alarm-field}"
    textColor: "{colors.alarm-text}"
  state-band-acknowledged:
    backgroundColor: "{colors.ack-field}"
    textColor: "{colors.ack-text}"
  panel:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text}"
    rounded: "{rounded.panel}"
    padding: "16px"
  button:
    backgroundColor: "{colors.panel-raised}"
    textColor: "{colors.text}"
    rounded: "{rounded.control}"
    padding: "0 14px"
    height: "40px"
  button-primary:
    backgroundColor: "{colors.action-blue}"
    textColor: "#ffffff"
    rounded: "{rounded.control}"
    padding: "0 14px"
    height: "40px"
  button-header:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text}"
    rounded: "{rounded.control}"
    padding: "0 12px"
    height: "36px"
  button-acknowledge:
    textColor: "{colors.alarm-text}"
    rounded: "{rounded.panel}"
    padding: "0 24px"
    height: "56px"
  chip-ok:
    backgroundColor: "{colors.panel-raised}"
    textColor: "#9df0d2"
    rounded: "{rounded.pill}"
    padding: "4px 12px"
  chip-attention:
    backgroundColor: "{colors.panel-raised}"
    textColor: "#ffd978"
    rounded: "{rounded.pill}"
    padding: "4px 12px"
  chip-violating:
    backgroundColor: "#3a1210"
    textColor: "#ffc9c3"
    rounded: "{rounded.pill}"
    padding: "4px 12px"
  input:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.text}"
    rounded: "{rounded.input}"
    padding: "6px 10px"
---

# Design System: Giám sát an toàn phòng lab

## Overview

**Creative North Star: "The Control Room Window"**

The page is a control-room console read from three metres away, dark so it sits quietly in a lab and on a projector. One full-width band owns the top of the page and states the situation in a single word: AN TOÀN, CHÚ Ý, BÁO ĐỘNG, or MẤT TÍN HIỆU. Everything below is evidence for that word: the live video is the largest object, the latest alarm's photos sit beside it, and counts, chart and per-person chips are details. The owner pinned this direction ("Phòng điều khiển, đọc từ xa"); no concept roll was run.

Colour does jobs only. The ground is a cool ink with slightly lifted panels, separated by tone and 1px hairlines rather than nested cards. The four state colours are the only loud colours on the page and each one floods the band as a field, not a badge. Calm is a quiet deep teal; alarm is a saturated red that also frames the viewport and the video. Type is the system sans stack only (the page is self-contained, with no CDN and no downloaded fonts, because the board has no internet), so hierarchy comes from size, weight and tabular numerals, not from face choice.

The page is Vietnamese, operates over a localhost SSH tunnel, and never shows safe while the feed is dead: lost signal has its own steel-grey field.

**Key Characteristics:**
- One state band, reserved height, full width; the state is a background field plus a 56px word.
- Video is the largest object; evidence and history live in a 400px right column.
- Tone and hairline separation; panels are flat.
- Large tabular numerals for counts; muted text for everything secondary.
- One-stroke SVG icons (18px, 2px stroke, round caps), no emoji.

## Colors

A cool-ink ground with four signal fields; the palette is neutral blue-greys until a state needs the eye.

### Primary
- **Action Blue** (`action-blue`): the only filled action button (Chốt vùng này, Lưu và áp dụng) and the current step marker in the zone editor.
- **Link Blue** (`link-blue`): focus rings (3px), hover borders on controls, accent-color and caret, current-step border. It is interaction colour, never state colour.

### Secondary (state fields)
- **Safe Teal** (`safe-field` with `safe-text`): the band when the camera is live and nothing is violated. `safe-signal` is the live dot, chart-free ok messages.
- **Watch Amber** (`watch-field` with `watch-text`): someone is without a lab coat outside a danger zone, or a just-stopped alarm. `watch-signal` is the dashed "không mặc áo lab" chart line and attention chips.
- **Alarm Red** (`alarm-field` with `alarm-text`): active violation, fall, fire or smoke. `alarm-signal` is the lighter red used for text, chart violation line, alarm markers and errors on dark panels.
- **Acknowledged Maroon** (`ack-field` with `ack-text`): the same alarm after "Đã xem"; calmer, still red-family, still reads BÁO ĐỘNG.
- **Lost Steel** (`lost-field` with `lost-text`): no fresh frame, or connecting. Never green.

### Neutral
- **Ink** (`ink`): page and header ground, input fill.
- **Panel** (`panel`) and **Panel Raised** (`panel-raised`): panels and buttons one step above the ground.
- **Hairline** (`hairline`) and **Hairline Strong** (`hairline-strong`): 1px dividers and control borders.
- **Text** (`text`), **Text Muted** (`text-muted`), **Text Dim** (`text-dim`): primary, secondary and tertiary copy on the dark ground; all are chosen to keep contrast on `ink` and `panel`.

### Named Rules
**The State Owns The Loud Colours Rule.** Teal, amber and red appear only as the band field, its signal accents, severity and chip semantics, and chart series. Never use them decoratively, and never use blue to mean state.
**The Lost Is Not Safe Rule.** Absence of data gets steel grey and its own word. No green element may render while the feed is dead.

## Typography

**Display Font:** system sans stack (system-ui, Segoe UI, Roboto, Helvetica Neue, Arial). Fixed by the offline, no-download constraint of the surface.
**Body Font:** the same stack.
**Label/Mono Font:** none; numerals use `font-variant-numeric: tabular-nums`.

**Character:** Plain, heavy at the top of the hierarchy and quiet below. The state word is the only extra-heavy (800) element on the page.

### Hierarchy
- **State word** (800, 56px, 1, +0.01em): the band word; 40px under 960px width.
- **Numeral** (700, 40px, 1, tabular): the three counts in the band; 32px under 960px.
- **Sentence** (400 with a 700 lead phrase, 24px, 1.3): who and where, clamped to two lines in a reserved 2.6em; 18px and three lines under 960px.
- **Title** (700, 22px, 1.25): the latest alarm's kind.
- **Headline** (650, 19px): panel headings and the help summary.
- **Body** (400, 15px, 1.5): page default; the page name is 18px / 650 in the header bar.
- **Label** (14px, 1.5): buttons, hints, metadata, ages, legends (13px), zone status (13px).
- **Caption** (12px): picture captions, key caps, chart axis text.

### Named Rules
**The Three-Metre Rule.** Anything that must be read from across the room (state word, sentence, counts) is 24px or larger; anything under 14px is reference text only.
**The Tabular Rule.** Every number that updates live uses tabular numerals so the layout never jitters.

## Layout

A top bar (min 52px) then the band (min 136px, reserved so a state change never shifts the page), then a two-column grid: stage and evidence. The main grid is a fluid video column plus a 400px side column (340px under 1100px), capped at 1720px, with 16px gaps and 24px side padding. The video keeps its stream aspect ratio and is limited to the viewport height minus 310px (min 360px) on wide screens. Under the video: zone tools, then the 15-minute chart, then the per-person chips, then a collapsed help panel. The side column holds the latest-alarm evidence over a scrolling history. Spacing steps in use: 4, 8, 12, 16, 24, 32.

Under 960px the band collapses to one column (word 40px, acknowledge full width), the side column dissolves into the single stack in order: video, latest alarm, chart, history, people, help, and all controls reach 44px minimum height (also on coarse pointers). Under 600px FPS and the fullscreen label hide.

The band is a three-part grid: word and sentence (fluid), counts (28px gap), acknowledge (right). The viewport also receives a 6px transparent frame that turns alarm red in alarm state.

## Elevation & Depth

Flat. Depth comes from tonal steps (ink, panel, panel-raised) and 1px hairlines; there are no box shadows anywhere. Emphasis is by field colour, outline and filter: in alarm the viewport gets a 6px red frame, the video gets a 4px red outline offset 3px, and the band arrives with a 0.7s brightness settle (1.9 to 1). Dialogs sit on an `rgba(3,6,10,.88)` backdrop.

### Named Rules
**The Flat Panels Rule.** Panels never nest as cards inside cards and never carry shadows; separation is tone plus a hairline.
**The Whole-Screen Alarm Rule.** An alarm is announced by field, frame and outline together, not by a small badge.

## Shapes

Soft but restrained: panels 10px, controls and video 8px, text inputs 6px, key caps 5px, status pills and chips fully round (999px). The acknowledge button borrows the panel radius (10px) with a 2px currentColor border. Borders are 1px hairlines; the zone-editor steps are numbered round markers (22px) inside pills. Icons are inline SVG symbols, 18px, 2px stroke, round caps and joins, `currentColor`.

## Components

### State band
Full-width field that swaps colour per `data-state` with a 0.25s background transition (none under reduced motion). Left: the 56px word, an optional "+N cảnh báo khác" pill (2px currentColor outline), and a two-line sentence. Centre: three counts, each a 14px label under or over a 40px numeral. Right: the acknowledge button, hidden by visibility (so layout holds) until an alarm or an unacknowledged watch state.

### Buttons
- **Shape:** 8px radius, 40px high (44px on touch and under 960px), 1px strong hairline, 600 weight, icon plus label at 8px gap.
- **Default:** raised-panel fill; hover shifts the border to Link Blue.
- **Primary:** Action Blue fill and border, white text; hover lightens the fill.
- **Danger:** default fill, light-red text, dark-red border (for "Làm lại từ đầu").
- **Header control:** 36px, panel fill, 14px; pressed state gets a deep-blue fill and blue border.
- **Acknowledge:** 56px, 18px bold, 2px currentColor border, translucent white fill (14%, 26% on hover) so it works on every band colour.
- **Disabled:** 45% opacity, not-allowed cursor. **Focus:** 3px Link Blue outline at 2px offset (currentColor on the band).

### Panels
10px radius, panel fill, 1px hairline, 16px padding, 19px / 650 heading with 12px below. The history list uses 1px row dividers instead of cards.

### Chips and severity pills
Fully round. Person chips: ok (green border and text), attention (dashed amber border), violating (red border, dark red fill, bold). Severity pills: low steel-blue, medium amber, high and critical solid alarm red with white text.

### Inputs / Fields
Ink fill, 1px strong hairline, 6px radius, 6px 10px padding; checkboxes and radios take the accent colour; fieldsets use a 8px radius strong hairline with a 650 legend.

### Evidence images
Buttons wrapping images, black fill, 8px radius, zoom-in cursor; the latest alarm shows a 3:4 person crop beside a 4:3 full frame (9fr / 16fr), history rows show 72px-high thumbnails. Clicking opens a modal dialog with a dark backdrop and a 44px circular close button.

### Chart
A 140px SVG: hairline grid, 12px dim axis text, persons as a muted solid line (2px), without-coat as an amber dashed line (6 4), violations as a red solid line (3px) and alarm moments as vertical red markers with a dot. A legend repeats each stroke pattern so series are distinguishable without colour.

### Zone editor
A stepped, inline editor: three numbered step pills (1 Vẽ viền vùng, 2 Chốt vùng, 3 Lưu và áp dụng) with the current one in Link Blue, a hint line, name and coat-rule options, actions, and a save box that only appears when savable. On wide screens it docks at the top of the side column; on narrow screens it follows the tools. Drawing happens on a canvas over the video with a crosshair cursor. Reset needs a second press within 4 seconds.

## Do's and Don'ts

### Do:
- **Do** lead every state change through the band: field colour, word and sentence together, in the reserved height so nothing shifts.
- **Do** use the tokens (`ink`, `panel`, `hairline`, state fields) for new surfaces; separate regions by tone and a 1px hairline.
- **Do** keep touch targets at 44px under 960px and 40px or more on desktop.
- **Do** use tabular numerals for live numbers and pair every colour-coded series or chip with a non-colour cue (dash pattern, border style, text).
- **Do** keep icons as one-stroke inline SVG at 18px, 2px stroke.
- **Do** honour `prefers-reduced-motion`: the dot pulse, band arrival and band transition are the only motion and all are disabled.

### Don't:
- **Don't** use green, amber or red for decoration, or blue to signal state.
- **Don't** show any positive or safe element while the feed is lost.
- **Don't** introduce box shadows, nested cards, gradients, or a five-equal-cards stat row.
- **Don't** add web fonts, CDN assets or external requests; the page must work offline.
- **Don't** let an unacknowledged alarm disappear; acknowledgement is browser-only and quiets sound and colour, not the state.
- **Don't** use thin banners that push the layout when a state appears.
