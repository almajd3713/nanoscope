// Design system, anti-patterns: no emoji and none of the banned marketing words in
// interface text (JSX text, string props, and string literals inside JSX expressions).
const BANNED = ["seamless", "seamlessly", "powerful", "effortless", "effortlessly", "elevate", "supercharge", "unleash", "AI-powered"];
const BANNED_RE = new RegExp(`(?<![\\w-])(${BANNED.join("|")})(?![\\w-])`, "i");
const EMOJI_RE = /(?![©®™])\p{Extended_Pictographic}|\p{Emoji_Presentation}/u;

export default {
  meta: {
    type: "problem",
    schema: [],
    messages: {
      emoji: "No emoji in the interface; use text or an allowlisted icon.",
      banned: 'Banned copy word "{{word}}" (design system, voice).',
    },
  },
  create(context) {
    function check(node, text) {
      if (EMOJI_RE.test(text)) context.report({ node, messageId: "emoji" });
      const m = BANNED_RE.exec(text);
      if (m) context.report({ node, messageId: "banned", data: { word: m[1] } });
    }
    return {
      JSXText(node) {
        check(node, node.value);
      },
      JSXAttribute(node) {
        const v = node.value;
        if (v && v.type === "Literal" && typeof v.value === "string") check(v, v.value);
      },
      JSXExpressionContainer(node) {
        const e = node.expression;
        if (e.type === "Literal" && typeof e.value === "string") check(e, e.value);
        if (e.type === "TemplateLiteral") for (const q of e.quasis) check(q, q.value.cooked ?? "");
      },
    };
  },
};
