import { expect, test } from "@playwright/test";
import { saveFile } from "./helpers";

const MODEL = "models/for_user_block.py";
const BLOCKS = "blocks/gate_e2e.py";

const MODEL_SOURCE = `from nanoscope.blocks import Attention, Block, Decoder, GELUMLP, LayerNorm


class ForUserBlock(Decoder):
    def __init__(self, vocab_size: int, context_length: int = 256):
        super().__init__(
            vocab_size, context_length, d_model=64, n_layers=2,
            block=Block(norm=LayerNorm(), attn=Attention(n_heads=4), mlp=GELUMLP()),
        )
`;

const GATE = `import torch
import torch.nn as nn

from nanoscope.blocks import register_block


def naive_gate_e2e(x, weight, scale):
    return x * torch.sigmoid(weight) * scale


@register_block(reference=naive_gate_e2e, family="mlp")
class GateE2E(nn.Module):
    """x times sigmoid of a learned vector."""

    def __init__(self, d_model, context_length, scale=2.0):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model))
        self.scale = scale

    def forward(self, x):
        return x * torch.sigmoid(self.weight) * self.scale
`;

// Plan done-when: a user block that matches its reference passes its check job and shows as
// certified in the palette, and it got there without restarting anything.
test("a user block appears in the palette, passes its check job and shows as certified", async ({ page, request }) => {
  await page.addInitScript(() => localStorage.setItem("nanoscope.level", "Extend")); // Extend has certification
  await saveFile(request, MODEL, MODEL_SOURCE);
  await page.goto(`/model/${MODEL}`);
  const palette = page.getByRole("navigation", { name: "Block palette" });
  await expect(palette.getByText("LayerNorm", { exact: true })).toBeVisible();
  await expect(palette.getByText("GateE2E")).toHaveCount(0);

  // registering a block in a workspace file puts it in the palette, with no restart
  await saveFile(request, BLOCKS, GATE);
  const item = palette.getByRole("listitem").filter({ hasText: "GateE2E" });
  await expect(item).toBeVisible();
  await expect(item).toContainText("not certified");

  // certify it from the Models page: a job a worker runs
  await page.getByRole("link", { name: "Models" }).click();
  const row = page.getByRole("region", { name: "Your blocks" }).getByRole("row").filter({ hasText: "GateE2E" });
  await row.getByRole("button", { name: "Certify" }).click();
  await expect(row.getByText("certified", { exact: true })).toBeVisible({ timeout: 90_000 });

  // and the palette shows the seal
  await page.goto(`/model/${MODEL}`);
  const sealed = page.getByRole("navigation", { name: "Block palette" }).getByRole("listitem").filter({ hasText: "GateE2E" });
  await expect(sealed.getByLabel("certified")).toBeVisible();
});
