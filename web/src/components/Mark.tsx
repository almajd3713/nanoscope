// The reticle: a ring, four inner ticks and a cobalt center point. Follows the theme through
// currentColor and --accent, so it needs only one file for both themes.
export function Mark({ size = 20 }: { size?: number }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} role="img" aria-label="nanoscope">
      <g fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round">
        <circle cx="12" cy="12" r="9.5" />
        <path d="M12 2.5V7M12 17V21.5M2.5 12H7M17 12H21.5" />
      </g>
      <circle cx="12" cy="12" r="1.75" fill="var(--accent)" />
    </svg>
  );
}
