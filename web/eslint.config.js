import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist"] },
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
    files: ["src/icons/**"],
    rules: { "no-restricted-imports": "off" },
  },
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: { globals: globals.browser },
    plugins: { "react-hooks": reactHooks },
    rules: reactHooks.configs.recommended.rules,
  },
);
