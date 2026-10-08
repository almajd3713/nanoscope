// The design system's CSS rules (5-implementation.md). tokens.css is generated and exempt.
export default {
  ignoreFiles: ["src/styles/tokens.css"],
  rules: {
    "color-no-hex": true,
    "function-disallowed-list": [
      "rgb", "rgba", "hsl", "hsla", "hwb", "lab", "lch", "oklab", "oklch", "color-mix",
      "linear-gradient", "radial-gradient", "conic-gradient",
      "repeating-linear-gradient", "repeating-radial-gradient", "repeating-conic-gradient",
    ],
    "declaration-property-value-disallowed-list": {
      "text-transform": ["/uppercase/"],
      "letter-spacing": ["/^(?!0$).+/"],
      "backdrop-filter": ["/.+/"],
      "box-shadow": ["/^(?!var\\(--shadow-(pop|drag)\\)$|none$).+/"],
    },
    "declaration-property-unit-disallowed-list": { "font-size": ["px"] },
  },
  overrides: [
    {
      files: ["**/*.css"],
      excludeFiles: ["**/motion.module.css"],
      rules: { "at-rule-disallowed-list": ["keyframes"] },
    },
  ],
};
