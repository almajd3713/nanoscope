## What and why

<!-- One or two sentences. Link the checklist item (for example P13.12). -->

## Checks

- [ ] `make check` passes (Python changes), and `pnpm -C web lint && pnpm -C web lint:css && pnpm -C web typecheck && pnpm -C web test` pass (web changes)
- [ ] The checklist item is ticked in `docs/checklist.md` with the date and commit

## UI changes: design system review

Skip this section when the PR touches no UI. The rules are in `docs/design-system.md`.

- [ ] Banned patterns: none of the table in `docs/design-system.md` ("Banned patterns") appears. Color is only interactive, a state, or data; no gradients, glows, glass, nested cards, side stripes, KPI card rows, tinted icon squares, or emoji
- [ ] The diff has no hex, `rgb(`, `px` font sizes, `box-shadow`, `gradient`, `uppercase`, `letter-spacing`, `@keyframes` or icon imports that do not come from tokens or the allowlist
- [ ] Both themes: looked at light and dark
- [ ] Reduced motion: transitions are instant under `prefers-reduced-motion: reduce`
- [ ] 1024px: the screen works at 1024px wide (and, for watch screens, below 600px)
- [ ] Copy read aloud: sentence case, nanoscope's own words, no marketing words, library text shown word for word
- [ ] `EquivalentCommand` is present on the screen (it shows only when Settings turns it on)
- [ ] Every state shows a glyph and a word, never color alone
