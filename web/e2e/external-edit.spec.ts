import { expect, test } from "@playwright/test";
import { saveFile } from "./helpers";

const FILE = "models/external_edit.py";
const MODEL = `from nanoscope.blocks import Attention, Block, Decoder, GELUMLP, LayerNorm


class ExternalEdit(Decoder):
    def __init__(self, vocab_size: int, context_length: int = 256):
        super().__init__(
            vocab_size, context_length, d_model=64, n_layers=4,
            block=Block(norm=LayerNorm(), attn=Attention(n_heads=4), mlp=GELUMLP()),
        )
`;

// Plan done-when: an edit made to the file by something else (another editor, git) updates the
// open graph and the open editor.
test("an external edit to the file updates the open graph and editor", async ({ page, request }) => {
  await page.addInitScript(() => localStorage.setItem("nanoscope.level", "Tinker")); // Tinker has the editor
  await saveFile(request, FILE, MODEL);
  await page.goto(`/model/${FILE}`);
  await expect(page.getByText("4 × layer")).toBeVisible();
  await expect(page.getByLabel("GELUMLP in mlp", { exact: true })).toBeVisible();
  const code = page.getByRole("region", { name: `Editor for ${FILE}` });
  await expect(code.locator(".view-lines")).toContainText("n_layers=4");

  // someone else saves the file: more layers, a different mlp
  await saveFile(request, FILE, MODEL.replace("n_layers=4", "n_layers=6").replace("GELUMLP()", "SwiGLU()").replace("LayerNorm\n", "LayerNorm, SwiGLU\n"));
  await expect(page.getByText("6 × layer")).toBeVisible();
  await expect(page.getByLabel("SwiGLU in mlp", { exact: true })).toBeVisible();
  await expect(code.locator(".view-lines")).toContainText("n_layers=6");
  await expect(code.getByText("saved")).toBeVisible(); // a clean buffer follows the disk without asking
});
