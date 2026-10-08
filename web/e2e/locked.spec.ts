import { expect, test } from "@playwright/test";
import { checkLesson, readFile, saveFile, solution } from "./helpers";

const FILE = "lessons/modern-block/07-assemble/starter.py";

// A model that places the position encoding in an attention block, so there is a slot to drop on.
const MODEL = `from nanoscope.blocks import Attention, Block, Decoder, GELUMLP, LayerNorm, NoPE


class MyModern(Decoder):
    def __init__(self, vocab_size: int, context_length: int = 256, d_model: int = 128,
                 n_layers: int = 4):
        super().__init__(
            vocab_size, context_length, d_model=d_model, n_layers=n_layers,
            block=Block(
                norm=LayerNorm(),
                attn=Attention(n_heads=4, pos=NoPE()),
                mlp=GELUMLP(),
            ),
            final_norm=LayerNorm(),
        )
`;

// Plan done-when: a locked block shows its unlock lesson and cannot be dropped; once the lesson's
// check passes (with the solution file), it can be dragged without a restart.
// Needs a stack where RoPE has not been earned yet: e2e/stack.sh reset.
test("a locked block names its lesson, cannot be dropped, and drags after the lesson passes", async ({ page, request }) => {
  const unlocks = (await (await request.get("/api/learn/unlocks")).json()) as { lockable: Record<string, { state: string }> };
  test.skip(unlocks.lockable["block:RoPE"]?.state === "earned", "RoPE was earned by an earlier run: run e2e/stack.sh reset");

  await request.post("/api/learn/policy", { data: { policy: "guided" } });
  const started = await request.post("/api/curricula/modern-block/07-assemble/start", { data: {} });
  expect(started.ok(), await started.text()).toBe(true);
  await saveFile(request, FILE, MODEL);

  await page.goto(`/model/${FILE}`);
  const palette = page.getByRole("navigation", { name: "Block palette" });
  const rope = palette.getByRole("listitem").filter({ hasText: "RoPE" });
  await expect(rope).toHaveAttribute("aria-disabled", "true");
  await expect(rope.getByRole("link", { name: "modern-block/02-rope" })).toBeVisible();
  const target = page.getByLabel("NoPE in pos", { exact: true });

  // dragging a locked item does not start, and nothing reaches the file
  await rope.dragTo(target);
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(await readFile(request, FILE)).toBe(MODEL);

  // pass the RoPE lesson with the solution
  await request.post("/api/curricula/modern-block/02-rope/start", { data: {} });
  await saveFile(request, "lessons/modern-block/02-rope/starter.py", solution("modern-block/02-rope.py"));
  expect((await checkLesson(request, "modern-block/02-rope")).passed).toBe(true);

  // the palette follows without a reload: RoPE is a normal item now
  await expect(rope).not.toHaveAttribute("aria-disabled", "true");
  await rope.dragTo(target);
  await expect.poll(() => readFile(request, FILE)).toContain("pos=RoPE()");
  expect(await readFile(request, FILE)).toBe(MODEL.replace("pos=NoPE()", "pos=RoPE()").replace("GELUMLP, LayerNorm, NoPE", "GELUMLP, LayerNorm, NoPE, RoPE"));
});
