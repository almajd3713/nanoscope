import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";
import tseslint from "typescript-eslint";
import noBannedCopy from "./eslint-rules/no-banned-copy.js";

export default tseslint.config(
  { ignores: ["dist", "src/api/schema.d.ts"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    rules: {
      "no-restricted-imports": [
        "error",
        {
          paths: [
            { name: "@phosphor-icons/react", message: "Import icons from src/icons (the design system's allowlist)." },
            { name: "lucide-react", message: "Banned by the design system." },
            { name: "react-icons", message: "Banned by the design system." },
            { name: "simple-statistics", message: "The page shows the library's numbers; it computes none." },
            { name: "jstat", message: "The page shows the library's numbers; it computes none." },
            { name: "mathjs", message: "The page shows the library's numbers; it computes none." },
            { name: "d3-array", importNames: ["mean", "deviation", "variance", "median", "quantile"], message: "The page shows the library's numbers; it computes none." },
          ],
          patterns: [
            { group: ["@heroicons/*", "react-icons/*", "lucide-react/*"], message: "Banned by the design system." },
            { group: ["@phosphor-icons/react/*"], message: "Import icons from src/icons (the design system's allowlist)." },
          ],
        },
      ],
    },
  },
  {
    files: ["src/**/*.tsx"],
    plugins: { nanoscope: { rules: { "no-banned-copy": noBannedCopy } } },
    rules: { "nanoscope/no-banned-copy": "error" },
  },
  {
    files: ["src/icons/**"],
    rules: { "no-restricted-imports": "off" },
  },
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: { globals: globals.browser },
    plugins: { "react-hooks": reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // react-markdown passes `node`; it is dropped from the props by destructuring
      "@typescript-eslint/no-unused-vars": ["error", { ignoreRestSiblings: true }],
    },
  },
);
