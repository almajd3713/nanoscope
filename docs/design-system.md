# nanoscope design system

This is the repo's copy of the nanoscope design system, for people and agents working in `web/`.
The source is a private claude.ai artifact (Claude Design "Design System"):
https://claude.ai/artifact/D7a6hvoZR3zsH4zWiXYQQL

What lives where:

- `web/src/styles/tokens.json` mirrors the artifact's `tokens.json`; `tokens.css` is generated from it
  (`pnpm -C web tokens`) and is never edited by hand.
- `web/src/styles/fonts/` and `fonts.css` hold the bundled IBM Plex files (SIL OFL 1.1).
- `web/src/icons/index.ts` is the only place Phosphor is imported, and exports the 29-icon allowlist.
- The component previews and the logos stay in the artifact; the React code in `web/` is written fresh.

Mirrored from artifact version 1791460873-986d on 2026-10-08. The sections below are the artifact's
README, Layout, Voice and copy, Data, Banned patterns and Implementation files, one after another.
Where this file and the artifact differ, the artifact wins; update this file from it.


## Overview

nanoscope is a lab for small language models: a learner builds a model from primitives, trains it on a CPU in minutes, and later measures changes with seeds and confidence intervals. The interface is a **lab instrument**. It shows what the files on disk say, as precisely and calmly as it can, and gets out of the way. Every rule below serves that.

These rules bind every screen in phases P13 to P17. Where a rule and a habit disagree, the rule wins. Where a case is not covered, choose the plainer option and add the rule here (see *Changing this system* in Implementation).

## Principles

1. **Instrument, not dashboard.** A page is about one thing: a run, a lesson, a comparison, a model. Put that thing's name and state at the top, its main measurement next, and the detail below. No overview grids of equal cards, no hero banners, no illustrations.
2. **Color is information.** The neutrals carry the layout. `accent` means "you can act on this" or "this is selected". `good`, `warn` and `bad` mean a state. Series colors identify data. Nothing is colored for decoration.
3. **Values look like values.** Every number, unit, shape, run ref, seed, hash and command is set in Plex Mono (`value`, `value-strong`, `readout`, `code`). Words that describe them are set in Plex Sans. Never set labels, headings or navigation in mono.
4. **The files are the truth.** The interface shows what the library wrote (`status.json`, `results.json`, verdict text, problem+json `detail`) word for word. It never computes a statistic, rounds a CI differently, or rewords an error. Every screen has its `EquivalentCommand`, so nothing the GUI does lacks a command; it is shown when the person turns on Settings › Show command-line equivalents (off by default, since most people never use the CLI).
5. **Levels expose, they never fork.** Learn, Tinker, Research and Extend change which controls are visible (the single table in `web/src/levels.ts`). They never change colors, density, layout grids or request bodies.
6. **Calm under load.** Live data updates in place: no animation of numbers, no pulsing, no spinners where a step count exists. A long operation always shows a determinate progress line and where its status file lives.

## Color

- Ground: the page is `surface`. Content sits on `surface-raised` panels with a 1px `line` border. Inputs, code, command blocks and plot areas are `surface-sunken`. Use at most these three levels; a panel inside a panel uses no new fill (separate with `line` or spacing instead).
- Text: `ink` for primary text, `ink-muted` for help, units, metadata and axis labels. Do not invent a lighter grey: `ink-muted` is the lightest text that passes 4.5:1 on every ground in both themes. `ink-disabled` is for disabled controls only.
- Accent: `accent` (cobalt) marks primary buttons, links, the selected tab, the selected nav item, the selected graph node and the `focus` ring. Use one primary button per view region. Hover and pressed use `accent-strong`; text on a fill uses `on-accent`. Selected rows and active drop slots use `accent-soft`.
- State: `good`, `warn`, `bad`, each with a `*-soft` ground. They appear only through `StateTag`, `CheckResult`, `Verdict`, `Note` and `Problem`, always as **glyph + word**. A state is never shown by color alone, and good and bad are never told apart by hue alone (their lightness is close by design, so the glyph and the word carry the meaning).
- Data: `series-1` to `series-6` (Okabe-Ito, darkened where needed for 3:1 on light grounds) identify runs and variants on plots, in the same order everywhere a set appears. `baseline` (dashed, neutral) and `band` mark the shipped reference. `heat-1` to `heat-5` is the one sequential scale (block stats, attention maps). Heat fills never carry text.
- Diffs: one line per row, removed lines in `diff-del` on `diff-del-soft` with a leading −, added lines in `diff-add` on `diff-add-soft` with a leading +. This is the one place red and green appear outside a state; the signs carry the meaning for readers who cannot tell the colors apart.
- Code: the editor and code blocks use `syn-*` tokens on `surface-raised`, with `selection` for selected text. They are distinct from `accent` and the state hues on purpose.
- Themes: light and dark are designed separately with the same token names. The default follows the OS; a toggle in the top bar overrides it and is kept in localStorage inside try/catch. Every third-party surface (Monaco, React Flow, uPlot, elk layouts) takes its colors from these tokens at runtime and re-reads them when the theme changes.

## Type

- Two families from one superfamily: Plex Sans (`--font-sans`) for words, Plex Mono (`--font-mono`) for values. Both are bundled as woff2 files and served by the app; nothing loads from Google Fonts or a CDN (compose runs offline). `Δ` comes from the bundled Plex Sans Greek subset, because Plex Mono has no Greek.
- Interface text: `title` once per page, `heading` for panels, `body` as default, `small` in dense tables, `label` for field labels and table headers, `caption` for axis labels and footnotes.
- Lesson text: `prose`, `prose-heading` and `prose-subheading`, at a measure of `measure` (68ch). Only lesson pages and long help use the Reading styles.
- Values: `readout` for the one live number a page is about (at most two per page), `value` for every other number, `value-strong` when the number is the subject of its row, `code` and `code-small` for code and commands.
- All figures are tabular (`font-variant-numeric: tabular-nums`). Use the true minus `−` (U+2212), `±`, `×` and `Δ`. Never put a space inside a number set in mono (a mono space is a full character wide and splits the number in two): large counts take an attached SI suffix with up to two decimals (`18.4k tokens/s`, `13.2M params`, `1.8G FLOP/token`), matching the library's output where it already formats a number.
- Case: sentence case everywhere: titles, buttons, tabs, table headers, labels. No uppercase text, no letter-spacing on labels, no italic except in prose and comments.
- Weights: 400, 500 and 600 only. 600 is for `title` and headings; 500 for button labels and emphasis.

## Space, density and layout

- Spacing comes from `space-1` to `space-7` (4px base). Lay out siblings with flex or grid and `gap`; never add margins per element.
- One density. Data surfaces (runs list, compare rows, queue, components table, palette) use `small` text at `row` (28px) rows. Reading surfaces (lesson text, dialogs with explanation) use `body` or `prose`. There is no compact/comfortable toggle and no level changes density.
- Controls are `control` (28px) tall. `control-lg` (36px) is reserved for the single primary action of a lesson page (Start, Train, Check). Toolbar icon buttons are `control-sm`.
- Desktop-first: every screen is designed at `bp-wide` (1280px) and must work at `bp-narrow` (1024px). Below `bp-phone`, only the watch screens (run page, runs list, queue, lesson reading) adapt; the editor, graph and study builder show "Open on a larger screen" with a link to the run or file they would have opened. Layout specifics are in the Layout section.

## Borders, radii, elevation

- Borders separate; shadows lift. Panels on the page have a `line` border and no shadow. Controls that must be found have a `line-strong` border.
- Radii are small and assigned by role: `radius-0` for tables and plot areas, `radius-1` for tags and slots, `radius-2` for buttons, inputs, panels and graph nodes, `radius-3` for popovers and dialogs. Nothing is pill-shaped or fully round except the radio dot and the mark's center point.
- `shadow-pop` for popovers, menus, tooltips and dialogs; `shadow-drag` only for a block being dragged. No other shadows, no colored shadows, no glows, no blur or glass.
- Never put a colored stripe on one side of a card or row. State goes in a `StateTag`; selection uses `accent-soft` plus the `accent` focus or selected outline on all four sides.

## Iconography

- Text first. Navigation, tabs and buttons are text labels. An icon appears only where it carries meaning a word would make noisy: a state glyph, a lock, a certified seal, the drag handle, and tight toolbars (stop, resume, duplicate, copy) where each icon button has a visible tooltip and an `aria-label`.
- Icons are Phosphor Regular, from the allowlist in the Icons asset group (29 icons, each with one meaning). Adding one means adding it to that group and its README first. Do not use Lucide, Heroicons, emoji or sparkle icons, and never put an icon in a tinted rounded square.
- Size 16px in text and tables, 20px in toolbars. Icons take `currentColor` and inherit the text color of their context.

## Motion

- Motion only shows a change of state: a panel opening (120ms, ease-out), a tag changing state (no animation; it simply changes), a dragged block settling into a slot (160ms). Nothing animates on page load.
- Live values replace in place. No count-up, no pulsing dots, no shimmer skeletons for content that has a known shape (draw the frame and write "Loading run…" in `ink-muted`).
- Under `prefers-reduced-motion: reduce`, all transitions become instant.

## States, in one place

| State | Where it comes from | Glyph | Word | Tokens |
|---|---|---|---|---|
| queued | status.json, queue | `circle` | queued | `ink-muted` on `surface-sunken` |
| preparing | status.json | `circle-half` | preparing | `ink` on `surface-sunken` |
| running | status.json, queue | `play` | running | `ink` on `surface-sunken`, with the step count beside it |
| done | status.json, queue | `check` | done | `good` on `good-soft` |
| stopped | status.json (Ctrl-C) | `pause` | stopped | `warn` on `warn-soft` |
| cancelling | queue | `prohibit` | cancelling | `warn` on `warn-soft` |
| cancelled | status.json, queue | `prohibit` | cancelled | `ink-muted` on `surface-sunken` |
| failed | status.json, queue | `x` | failed | `bad` on `bad-soft` |

Running is deliberately not colored: it is the one state that already has live numbers and a progress line next to it. Verdicts (`better`, `worse`, `within noise`, `no CI`, `baseline`) and check results follow the same glyph-plus-word rule; see the Data section.

## Focus and accessibility

- Every control is reachable by keyboard in reading order. The focus ring is 2px solid `focus` with a 2px offset in the ground color; it is never removed, only restyled.
- Text meets 4.5:1 and meaningful marks (control borders, plot lines, icons, focus) meet 3:1 on their grounds in both themes. The usage note on each color token names the grounds it was checked on.
- Drag and drop always has a keyboard and click path: the inspector's swap menu does what dragging does, with the same patch.
- Every chart has a text equivalent next to it (the table it was drawn from, or the verdict sentence).

## The mark

- The mark is a reticle: a ring with four inner ticks and a cobalt center point, the view through an eyepiece graticule. The lockup is the mark plus the lowercase wordmark "nanoscope" in Plex Sans SemiBold, outlined.
- Use `nanoscope-lockup-light.svg` on light grounds and `-dark.svg` on dark grounds; the mark alone is the favicon and appears in the top bar at 20px. Do not recolor it, add effects, or set "nanoscope" in another face. Always lowercase.

## Third-party surfaces

- Radix UI primitives provide behavior (dialog, popover, menu, tabs, tooltip, toggle group, scroll area). Their styling comes only from these tokens through CSS Modules.
- Monaco, uPlot, React Flow and elkjs are themed from tokens at runtime. Their default themes and default node styles must never show.
- The detailed mapping, file layout and lint rules that enforce all of this are in the Implementation section.

## Layout

### The shell

- **Top bar** (`topbar`, 44px, `surface`, `line` bottom border): the 20px mark and wordmark at the left, then the text nav: **Learn, Models, Runs, Compare, Studies, Hardware, Components** (Models arrives with the model page in P14). At the right: the `LevelSwitch`, the Settings link (text, like the nav), then the theme toggle (`sun` / `moon`). The current nav item is `ink` with a 2px `accent` underline; the others are `ink-muted`. No icons in the nav.
- **Queue footer** (`surface`, `line` top border, collapsed to one 28px line): "2 running · 1 queued" in `small`, expanding upward into the `JobRow` list with cancel. It is the shell's only persistent panel.
- **Content** sits between them, on `surface`, with a `space-5` side gutter (`space-4` below `bp-phone`). Pages are left-aligned, never centered in a narrow column, except lesson prose, which is held to `measure`.
- There is no left sidebar in the shell. Pages that need a side column (palette, lesson outline, inspector, run side column) own it.

### Page anatomy

Every page follows the same order, top to bottom:

1. **Header row**: `title` (the thing's own name: a run ref, a lesson title, a study name), its `StateTag` if it has a state, and the page's actions at the right (one primary button at most).
2. **Measurement row** (pages that have one): one or two `Readout`s and the `Progress` line, inline. Never a row of cards.
3. **Body**: panels on `surface-raised`, `space-5` apart.
4. **Equivalent command**: the `EquivalentCommand` block at the end of the header area or the bottom of the main column, present on every screen (Login is the only exception) and visible only when Settings › Show command-line equivalents is on (off by default). The mockups draw it on.

### Grid and widths

- Designed at `bp-wide` (1280px). A 12-column grid with 24px gutters inside the content area; side columns use fixed widths (`sidebar` 240px, `inspector` 320px) and the main column takes the rest.
- At `bp-narrow` (1024px) the side columns stay and the main column shrinks; panels in a two-column arrangement stack.
- Below `bp-phone` (600px), only these screens adapt, to one column: run page (header, readout, curve, samples), runs list (state and ref columns only), queue, lesson reading. The model page, study builder and graph show an `EmptyState`: "The model editor needs a screen at least 1024px wide" plus a link to the file's raw view and the equivalent command.

### Screens (P13 and P14)

| Screen | Layout | Main content | Side column |
|---|---|---|---|
| Onboarding | Single column, `measure` wide, left-aligned | Two choices as plain radio rows ("I'm learning: lessons unlock blocks as I go" / "I know this: everything open"), one Continue button | none |
| Lessons | Main column only | Paths as sections (`heading`), each a list of `LessonRow`s in order | none |
| Lesson | Main + `sidebar` outline at the left | `Tabs` Surface / Deep / Reading over `prose`; the primary action (`control-lg`); `CheckResult`; `Predict` | Lesson outline of the path, compute estimates, "Passing unlocks" |
| Run | Main + `inspector` column at the right | Header (ref, `StateTag`, actions), `Readout` + `Progress`, `Curve` with baseline band, samples timeline | Config summary (`DataTable`), device, `EquivalentCommand` |
| Runs | Main only | `DataTable`: state, ref, preset, step, bpb, started; filters above as a text field and state toggles | none |
| Run form (Tinker) | Main + `inspector` | Fields generated from ModelSpec/PresetSpec, live problems inline | The request preview as JSON, the equivalent command |
| Compare | Main only | `Verdict` sentence(s) at the top, the verdict table, `ForestPlot` beside the table at `bp-wide` (below at `bp-narrow`), notes as `Note`s, per-seed curves, the precision plan line | none |
| Components | Main only | `DataTable` of blocks and features: lock state, how unlocked, evidence link; "Unlock all" with `ConfirmDialog` | none |
| Models (P14) | Main only | `DataTable` of the workspace's model files: class, kind, params and FLOPs from the last trace, last edit, trace state; New model | none |
| Settings | Main only, `measure` wide sections | This browser (theme, level, show command-line equivalents), Learning (gating policy), Data (each preset's token cache, Prepare), This server (read-only: version, folders, offline mode, which secrets the workers have; never their values) | none |
| Model (P14) | Three columns: palette (`sidebar`) / editor + graph split / inspector | Monaco on the left half, the graph on the right half (the split is draggable, minimum 360px each) | Inspector: arguments of the selected node, swap menu, shapes, params, FLOPs |

### Levels

The level changes visibility only. Each control is listed once in `web/src/levels.ts` with the lowest level that shows it; hidden controls keep their level-0 defaults and are never sent changed.

| Level | Adds |
|---|---|
| Learn | lessons, model graph (surface view), palette with swaps and locks, Train, live run page, baseline verdict, samples |
| Tinker | code editor beside the graph, the run form, seeds, duplicate and change one thing, compare, predictions |
| Research | study builder, budgets, param matching, record mode, queue and devices, forest plot, ablation cards, bench |
| Extend | workspace tree, block registration and certification, authoring preview, schemas and `/docs` |

A level never moves a control that exists at a lower level. Switching level only adds or removes; it never re-flows what remains into a different arrangement.

### The depth dial

Surface, Detailed and Research views (plan 6.3) are a `Tabs`-like segmented control placed above the thing they change (the graph, a result). They use the same components with more columns or more detail, not a different screen.

### Loading, empty and error states

- **Loading**: draw the frame (header, panel outlines) at once and write "Loading run…" in `ink-muted` in the main panel. No skeleton shimmer. If loading takes more than 3 seconds, say what is being waited on ("Waiting for the worker to start the describe job").
- **Empty**: `EmptyState` with a sentence naming what will appear here, the one action that creates it, and the command.
- **Error**: `Problem` showing the problem+json `title` and `detail` word for word, the status code, and the next step if the library names one. A failed run's page shows the error and the traceback tail from `status.json` in a `surface-sunken` block in `code-small`.
- **Disconnected**: when the SSE stream drops, the page keeps the last values, and a line under the header says "Live updates paused. Reconnecting…" in `warn`. Values never silently go stale.

## Voice and copy

nanoscope talks like a patient lab partner who knows the field: plain, specific, never excited. It addresses the person as "you" and never speaks as "we" or "I".

### Rules

- **Sentence case** for everything: page titles, buttons, tabs, headers, labels, menu items.
- **Name nanoscope's own things** with the words the library and CLI use: run, study, seed, preset, baseline, verdict, lesson, check, unlock, block, slot, layer, worker, job, queue, record mode, preregistration. Use the same word everywhere for the same thing. The glossary below is the list.
- **Buttons say what happens to what**: "Train", "Train 3 seeds", "Run the check", "Stop run", "Resume", "Duplicate and change one thing", "Unlock all", "Commit preregistration". Never "Submit", "OK", "Continue" when a specific verb exists.
- **Results are plain sentences.** "Check passed. Attention is unlocked." "Stopped at step 312. Resume continues from the last checkpoint." No exclamation marks, no congratulation, no emoji.
- **Library text is shown word for word.** Verdicts, notes, problem `detail`, check reasons and status errors come from the API and are never paraphrased, shortened or translated. Add context around them, not inside them.
- **Errors say what happened and what to do.** When the library gives the next step (a lesson to pass, a command to run), show it as a link or an `EquivalentCommand`.
- **Numbers carry units**, in `ink-muted`, after the value: `2.734 bpb`, `0:41 left`, `13.2M params`, `1.8G FLOP/token`. The SI suffix attaches to the number with no space.
- **Time**: remaining time as `m:ss` or `h:mm:ss` ("0:41 left"); timestamps as `2026-10-07 14:03` (24-hour, ISO date). Relative time ("3 minutes ago") only in lists, with the absolute time in a tooltip.
- **No marketing words** (see Banned patterns, Copy). No "simply", "just" or "easy": they shame a learner who finds it hard.

### Learner and researcher

- The same sentence serves both. Depth comes from the Surface, Deep and Reading tabs and the depth dial, not from a different tone.
- Explain a term the first time a lesson uses it, in the lesson's own text. The interface itself does not explain; it links to the lesson ("What is a seed?" links to the lesson that teaches it).

### Glossary

| Term | Meaning in the interface | Not |
|---|---|---|
| run | One training of one model on one preset with one seed; a folder under runs/ | job, experiment, training |
| ref | The run's address, e.g. `tinystories-5min/gpt2/seed-0` | ID, name |
| study | A set of runs (variants × seeds) planned together | experiment, sweep |
| variant | One model or setting inside a study | arm, condition |
| seed | The random seed of a run | repeat, trial |
| preset | A named training budget and dataset (`tinystories-5min`) | config, profile |
| baseline | The reference a comparison measures against; shipped baselines are 5-seed reference runs | control, benchmark |
| verdict | The library's word for a row: `better`, `worse`, `within noise`, `no CI`, `baseline` | result, score |
| check | A lesson's automated test (`defines`, `equivalent`, `trains`, `verdict`, `predicted`, `reproduces`) | test, quiz |
| unlock | Passing a lesson's check makes a block available to import | enable, earn |
| block | A model component (`Attention`, `RMSNorm`) | layer, module, widget |
| slot | A named place in a block that takes another block (`norm`, `attn`, `mlp`, `pos`) | socket, port |
| worker | The process that runs jobs (training, checks, describe) | runner, agent |
| job | One queued unit of work for a worker | task |
| record mode | A study whose plan is committed before it runs, so its result is evidence | strict mode |

## Data, plots and live state

### Numbers

- Show the number the API returned, formatted the way the library formats it. The frontend never computes a mean, delta, CI, p-value or verdict (lint rule and test in P13.30).
- Every number is set in `value` (tabular, Plex Mono). Align numeric table columns right, on the decimal point where the precision is shared.
- Deltas are signed with a true minus: `Δ −0.012 ± 0.008`. A CI is shown as `± half-width` inline and as `[low, high]` in the Research view.
- Lower is better for both metrics (loss and bpb). Every axis and column that shows them says so once: "bpb (lower is better)".

### Verdicts

- The verdict word comes from `rows[].verdict` verbatim. `Verdict` pairs it with a glyph and a state token: `better` `check` on `good`, `worse` `x` on `bad`, `within noise` `minus` on `ink-muted`, `no CI` `info` on `ink-muted`, `baseline` no glyph on `ink-muted`.
- `within noise` is a result, not a failure: it is never shown in `bad` or `warn`.
- Notes from the library (`notes[]`) are shown as `Note` with `tone="warn"`, word for word, under the table they qualify.
- The precision plan line ("with 5 seeds the CI would be about ±0.008") sits directly under the verdict table in `small` `ink-muted`.

### Curves (uPlot)

- Plot area `surface-raised` with `grid` lines at the y ticks only, no vertical grid. Axis labels and ticks in `caption` `ink-muted`; the y-axis label names the metric and "lower is better".
- One series per run or variant in the fixed `series-1` to `series-6` order. The line is 1.5px; the latest point gets a 3px dot and its value as a label at the right edge.
- The baseline is a dashed `baseline` line with a `band` fill for the shipped seeds' range. Eval points are 3px dots on the line; training loss between evals is drawn thinner (1px) and lighter (60% opacity), so eval points read as the measurement.
- No smoothing by default. A smoothing control may exist at Tinker and above, off by default, and the raw line stays visible behind the smoothed one.
- Legends are direct labels at the line ends when there are four series or fewer; otherwise a legend above the plot, in series order. Never a legend that relies on color alone: each entry has the variant's name.
- Live: new points append; the x-range grows; nothing animates.

### Forest plot (plain SVG)

- One row per variant, in the verdict table's order, aligned to the table's rows when they sit side by side.
- A vertical zero line in `ink-muted`, labeled "no difference". The point estimate is a 6px square in `ink`; the CI is a 2px horizontal line in `ink`; the row's verdict glyph sits at the right end.
- The x-axis is the delta in the metric's units, with "← better" and "worse →" under the axis (lower is better). Every value comes from the API; the plot is drawn from `rows[].delta` only.

### Sequential scale

- `heat-1` to `heat-5` for block stats on graph nodes (activation RMS, gradient norm, update ratio, attention entropy) and for attention maps. Five steps, no interpolation beyond them for node fills (attention maps may interpolate).
- A legend with the five steps and their value ranges sits next to any heat-colored view. Heat fills never carry text.

### Live state and staleness

- Live pages (run, study, queue) subscribe with `useEvents`. When the stream drops, values stay, and the header shows "Live updates paused. Reconnecting…" in `warn` until it is back.
- A value older than its expected update interval (for example a running run whose step has not changed in 2 minutes) is marked with "last update 2:14 ago" in `ink-muted`. Values never silently go stale.

## Banned patterns

AI coding tools return the median of their training data, and that median has a recognizable look. These patterns were collected from published audits of AI-generated interfaces (sources at the end), not from memory, and checked against nanoscope's own screens. Each one is banned unless a row in this table says otherwise. A UI pull request that contains one is not ready to merge.

### Color and surface

| Banned | Why it reads as generated | Do this instead |
|---|---|---|
| Purple, indigo or violet as a primary or accent; any `#6366f1`-like hue | Tailwind's `indigo-500` default became the median of AI output | `accent` cobalt, interactive only |
| Gradients on backgrounds, buttons, text (`background-clip: text`) or state pills | Decorative gradients are the most cited tell | Solid token fills. No gradient anywhere |
| Glows, colored box-shadows, aurora or blob backgrounds | "Premium" shorthand from crypto and AI launch pages | `shadow-pop` for floating things only |
| Glassmorphism, backdrop blur | Templated, hurts legibility | Solid `surface-raised` |
| Permanent dark mode with medium-grey body text | Dark-only themes with low-contrast text | Both themes designed; body text `ink`, secondary `ink-muted` (4.5:1+) |
| Several saturated hues competing with no hierarchy | No priority | Neutrals carry layout; one accent; state hues only for state |
| Second-order "tasteful" defaults: cream + terracotta, near-black + one acid-green or vermillion pop, broadsheet hairline layouts | What models fall back to once purple is banned | Our palette: cool neutrals, cobalt, Okabe-Ito data colors |

### Layout and components

| Banned | Why | Do this instead |
|---|---|---|
| Cards inside cards (more than one level of framed container) | Hierarchy by nesting instead of by type and space | One `surface-raised` panel level; group inside it with spacing and `line` |
| A colored stripe on one side of a card, row or callout | One of the most cited AI tells | `StateTag`, `Note` with a glyph, or `accent-soft` with a full outline |
| A row of equal stat cards (label, big number, % delta) | The "equal-3 KPI grid" of every generated dashboard | `Readout` inline in the page header; numbers in tables |
| Grids of identical feature or icon cards | Generic marketing layout | Lists and tables with real content |
| Icons in tinted rounded squares | Decoration, not meaning | Text first; a bare 16px Phosphor glyph only where it means something |
| Badges everywhere; an eyebrow badge or uppercase kicker above headings | Badge spam reads auto-generated | A tag only when it shows a real state; headings stand alone |
| Decorative numbering (01 / 02 / 03) | Structure that encodes nothing | Numbers only where order is real: lesson numbers (`foundations/03`), steps of a run, layers |
| Status dots that map to no defined state | "Meaningless status dots" | `StateTag` with a state from `status.STATES` or the queue |
| Fake window chrome (three traffic-light dots), fake terminals | Imitation of a product instead of the product | The real `EquivalentCommand` block |
| `rounded-2xl shadow-lg` on everything; one radius and one padding for all things | Uniform spacing and radius flatten hierarchy | Radii by role (`radius-0` to `radius-3`), spacing by relationship |
| Centered page layouts and centered hero headings | Landing-page reflex in a tool | Left-aligned pages, title at top left |
| Single-column, mobile-first layouts on desktop | Wastes the width a researcher needs | Multi-column at `bp-wide`; side columns of `sidebar` and `inspector` width |
| Empty states with an illustration and "Nothing here yet" | Teaches nothing | `EmptyState`: what will appear, the one action that creates it, and the command |
| Untouched component-library defaults (shadcn, Mantine, MUI looks) | Monoculture | Radix primitives styled only by our tokens |

### Type

| Banned | Why | Do this instead |
|---|---|---|
| Inter, Roboto or system UI as the only face; the "tasteful free" set (Space Grotesk, Geist, Instrument Serif, Fraunces) | The median fonts of generated UIs | Plex Sans + Plex Mono, bundled |
| A serif italic accent word inside a sans headline | Recycled hero trick | No accent words. Headings are plain |
| Uppercase headings or labels; uppercase mono "template chrome" labels with letter-spacing | A second-order tell, and the one an instrument look drifts into | Sentence-case `label` in Plex Sans |
| Monospace for prose, labels or navigation | Costume, not information | Mono only for values and code |
| Flat hierarchy where everything is the same size and weight | No hierarchy | The type scale in tokens |

### Motion

| Banned | Do this instead |
|---|---|
| Fade-up or stagger animations on page load or scroll | Nothing animates on load |
| Count-up numbers | Values replace in place |
| Pulsing dots, breathing glows, infinite spinners where progress is known | A determinate progress line with the step count |
| Bounce or spring easing | 120 to 160ms ease-out, only for a change of state |
| Motion that ignores `prefers-reduced-motion` | All transitions instant under reduce |

### Icons and imagery

| Banned | Do this instead |
|---|---|
| Lucide as the icon set | Phosphor Regular from the allowlist |
| Emoji anywhere in the UI (nav, headings, empty states, toasts) | Words; a Phosphor glyph where needed |
| Sparkle or magic-wand icons for anything | Say what happens ("Generate sample") |
| Arrows on every call-to-action button | Arrow only for navigation that leaves the page (`arrow-square-out`) |
| Stock illustrations, placeholder avatars, decorative charts | Real data from the user's runs, or nothing |

### Copy

| Banned | Do this instead |
|---|---|
| Marketing words: seamless, powerful, effortless, elevate, supercharge, unleash, "AI-powered" | Product verbs: train, check, compare, stop, resume, duplicate |
| Sentence-shape repetition ("Everything you need to…", "Built for…") | Headings that name the thing: "Validation loss", "Passing unlocks" |
| Vague buttons ("Submit", "Continue", "Get started") | The action and its object: "Train 3 seeds", "Run the check" |
| Errors that apologize or paraphrase ("Oops! Something went wrong") | The library's problem `detail` word for word, and what to do |
| Exclamation marks and celebration copy | A plain result: "Check passed. Attention is unlocked." |
| Fake numbers or invented example results shown as real | Real runs, or examples marked as examples |

### Review checklist (run before merging any UI change)

1. Squint at the screen: does color appear anywhere that is not interactive, a state, or data? Remove it.
2. Search the diff for hex values, `rgb(`, `px` font sizes, `box-shadow`, `gradient`, `uppercase`, `letter-spacing`, `@keyframes` and icon imports. Each must come from tokens or an allowlist.
3. Turn on the other theme and look again. Then emulate reduced motion and a 1024px window.
4. Read every string aloud. Does it name nanoscope's own things, in sentence case, with no marketing word?
5. Does the screen show its `EquivalentCommand`, and does every state show a glyph and a word?

### Sources

- Developers Digest, "AI design slop: 16 patterns that out your app as vibe-coded": https://www.developersdigest.tech/blog/ai-design-slop-and-how-to-spot-it
- funboy322/avoid-ai-design, audit catalog including the second-order "tasteful" defaults: https://github.com/funboy322/avoid-ai-design
- The Fountain Institute, "7 signs a UI has been vibe coded": https://www.thefountaininstitute.com/blog/signs-vibe-coded-ui
- Vibemole, "How to avoid building apps that look vibe coded": https://vibemole.com/resources/avoid-vibecoded-app-design
- Mania Design, "Spot the slop: a UI designer's guide to fixing AI defaults": https://www.mania.design/blog/spot-the-slop-a-ui-designers-guide-to-fixing-ai-defaults/
- Codercops, "Designing for developers" (density, hierarchy, empty states in tool UIs): https://blog.codercops.com/blog/designing-for-developers
- 925 Studios, "AI slop fonts and gradients": https://www.925studios.co/blog/ai-slop-design-tells
- Design Systems Collective, "AI chose your UI (did it choose wrong?)": https://www.designsystemscollective.com/ai-chose-your-ui-did-it-choose-wrong-b0d30611c724
- Laith Junaidy, "How to design a dashboard with Claude Code" (equal KPI grids, gradient state pills): https://github.com/Laith0003/ux-skill/wiki/How-to-design-a-dashboard-with-Claude-Code

Collected 2026-10-07. Re-run the research when this list is more than six months old; the median moves.

## Implementation

How the system becomes code in `web/` (P13 onward), and how it is kept from drifting.

### Source of truth and the repo mirror

- This design system is the source. The repo holds a mirror that code and agents follow:
  - `web/src/styles/tokens.json`: a copy of this system's `tokens.json`.
  - `web/src/styles/tokens.css`: generated from it by `web/scripts/tokens.ts` (`pnpm tokens`). Never edited by hand.
  - `web/src/styles/fonts/`: the woff2 files listed in `tokens.json`, with `@font-face` rules that add `unicode-range` (Latin, Greek) and `font-display: swap`.
  - `web/src/icons/`: the allowlisted Phosphor icons, imported from `@phosphor-icons/react` through one re-export module.
  - `docs/design-system.md`: the README and sections of this system, as one file, for agents working in the repo.
- A CI step (`pnpm tokens --check`) fails when `tokens.css` is stale against `tokens.json`.

### Changing this system

1. Change it here first (tokens, a rule, a component's guideline), with the reason in the change note.
2. Mirror it: copy `tokens.json`, regenerate `tokens.css`, update `docs/design-system.md`, in one commit titled `design: …`.
3. Log the decision in `docs/checklist.md` (Decisions log) when it changes a rule rather than a value.

An agent that wants a value or pattern this system does not have stops and asks; it never adds a local one-off.

### Styling

- **Radix UI primitives** for behavior: Dialog, AlertDialog, Popover, DropdownMenu, Tabs, Tooltip, ToggleGroup, ScrollArea, Separator, VisuallyHidden. No Radix Themes.
- **CSS Modules** per component (`Button.module.css`). Every color, size, radius, shadow and font comes from a `var(--token)`. Allowed literals: `0`, `1px` hairline widths, `100%`, and percentages in layout.
- **No** Tailwind, shadcn/ui, Mantine, MUI, Chakra, styled-components or CSS-in-JS runtime. No global utility classes beyond the type-style classes generated from `tokens.json` (`.title`, `.body`, `.value`, …).
- Themes: `tokens.css` defines light on `:root, [data-theme="light"]` and dark on `[data-theme="dark"]`, plus a `@media (prefers-color-scheme: dark)` block for `:root:not([data-theme="light"])`. The theme toggle sets `data-theme` on `<html>` and stores the choice in localStorage inside try/catch; "system" removes the attribute.

### Lint rules that enforce the system

- **stylelint**: `color-no-hex`, `function-disallowed-list: [rgb, rgba, hsl, hsla, linear-gradient, radial-gradient, conic-gradient]`, `declaration-property-value-disallowed-list` for `text-transform: uppercase`, `letter-spacing` other than 0, `backdrop-filter`, and `box-shadow` values other than `var(--shadow-pop)`, `var(--shadow-drag)` or `none`; `unit-disallowed-list` for `px` in `font-size`; `@keyframes` allowed only in `motion.module.css`.
- **ESLint**: `no-restricted-imports` for `lucide-react`, `@heroicons/*`, `react-icons`, any statistics library (`simple-statistics`, `jstat`, `d3-array`'s `deviation`/`mean`, `mathjs`), and for `@phosphor-icons/react` outside `web/src/icons/` (so the allowlist is the only door). A custom rule rejects emoji characters in JSX text.
- **Tests**: an axe pass on every page (P13.29); a test that each page renders `EquivalentCommand` (P13.10); a test that verdict text equals the API fixture (P13.30).

### Third-party surfaces

| Surface | How it takes the tokens |
|---|---|
| Monaco | A theme built at runtime from computed `--surface-raised`, `--ink`, `--ink-muted`, `--syn-*`, `--selection`, `--line`, `--accent`, `--bad`, `--warn`; rebuilt on theme change. Font `--font-mono` at the `code` style. Markers: errors `bad`, warnings `warn`. |
| uPlot | Series strokes from `--series-N`, `--baseline`, `--band`; axes and grid from `--ink-muted`, `--grid`; font from `--font-mono` at `caption` size. Redraw on theme change. |
| React Flow | Custom node types only (`BlockNode`); default node and edge styles turned off. Edges 1px `--line-strong`, selected edge `--accent`. Background: none (no dot grid). Controls and minimap themed or hidden. |
| elkjs | Layered layout, top to bottom, node spacing `space-5`, layer spacing `space-6`. No layout is stored. |

### Fonts

- Served from the app's own static files; the wheel and the image carry them. No request leaves the machine for a font.
- Preload `ibm-plex-sans-latin-400-normal.woff2` and `ibm-plex-mono-latin-400-normal.woff2` in `index.html`; the rest load on use.

### Component inventory and P13/P14 files

The components in this system map to files in `web/src/components/` (P13) and `web/src/graph/` (P14). The previews here are the visual specification; the React code in `web/` is written fresh in TypeScript with Radix and CSS Modules, matching them.

| Component | Repo file | Checklist |
|---|---|---|
| Button, Icon | components/Button.tsx, icons/index.ts | P13.08 onward |
| StateTag | components/StateTag.tsx | P13.17, P13.23 |
| CheckResult | components/CheckResult.tsx | P13.15 |
| Verdict, Note | components/Verdict.tsx, components/Note.tsx | P13.24 |
| Readout, Progress | components/Readout.tsx, components/Progress.tsx | P13.17 |
| Curve | components/Curve.tsx (uPlot) | P13.18 |
| ForestPlot | components/ForestPlot.tsx | P13.25 |
| Field | components/Field.tsx | P13.21 |
| LevelSwitch, TopBar | app/Shell.tsx, app/level.ts | P13.08 |
| Tabs | components/Tabs.tsx | P13.13 |
| DataTable | components/DataTable.tsx | P13.23, P13.27 |
| EquivalentCommand | components/EquivalentCommand.tsx | P13.10 |
| Problem | components/ProblemView.tsx | P13.04 |
| EmptyState | components/EmptyState.tsx | all pages |
| ConfirmDialog | components/ConfirmDialog.tsx | P13.27 |
| JobRow | components/QueuePanel.tsx | P13.28 |
| LessonRow, LockNote | pages/Lessons.tsx, components/LockNote.tsx | P13.12, P13.27 |
| BlockNode, PaletteItem | graph/nodes.tsx, graph/Palette.tsx | P14.12 to P14.18 |
