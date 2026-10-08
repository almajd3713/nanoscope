import { RuleTester } from "eslint";
import tseslint from "typescript-eslint";
import { describe, it } from "vitest";
import rule from "./no-banned-copy.js";

RuleTester.describe = describe;
RuleTester.it = it;

const tester = new RuleTester({
  languageOptions: {
    parser: tseslint.parser,
    parserOptions: { ecmaFeatures: { jsx: true } },
  },
});

tester.run("no-banned-copy", rule, {
  valid: [
    "const a = <p>Passing unlocks Attention. Δ −0.012 ± 0.008 · 3 seeds</p>;",
    'const a = <input aria-label="Learning rate" placeholder="3e-4" />;',
    "const a = <p>{'Loading run…'}</p>;",
    "const a = <p>© nanoscope</p>;",
    "const powerful = 1; const b = <p>{powerful}</p>;",
    "const a = <p>power-user mode</p>;",
  ],
  invalid: [
    { code: "const a = <p>A seamless start</p>;", errors: [{ messageId: "banned" }] },
    { code: "const a = <p>Powerful comparisons</p>;", errors: [{ messageId: "banned" }] },
    { code: "const a = <p>Effortless</p>;", errors: [{ messageId: "banned" }] },
    { code: "const a = <p>Elevate your model</p>;", errors: [{ messageId: "banned" }] },
    { code: "const a = <p>Supercharge training</p>;", errors: [{ messageId: "banned" }] },
    { code: "const a = <p>Unleash the block</p>;", errors: [{ messageId: "banned" }] },
    { code: "const a = <p>AI-powered hints</p>;", errors: [{ messageId: "banned" }] },
    { code: 'const a = <button title="Seamless">Go</button>;', errors: [{ messageId: "banned" }] },
    { code: "const a = <p>{'unleash'}</p>;", errors: [{ messageId: "banned" }] },
    { code: "const a = <p>{`so powerful`}</p>;", errors: [{ messageId: "banned" }] },
    { code: "const a = <p>Done 🎉</p>;", errors: [{ messageId: "emoji" }] },
    { code: 'const a = <button aria-label="Run 🚀">Go</button>;', errors: [{ messageId: "emoji" }] },
    { code: "const a = <p>{'✨'}</p>;", errors: [{ messageId: "emoji" }] },
  ],
});
