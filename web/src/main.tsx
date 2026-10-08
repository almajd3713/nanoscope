import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles/tokens.css";
import "./styles/fonts.css";
import { Providers } from "./app/providers";
import { initTheme } from "./styles/theme";

initTheme();

const root = document.getElementById("root");
if (!root) throw new Error("missing #root");

createRoot(root).render(
  <StrictMode>
    <Providers>
      <p>nanoscope</p>
    </Providers>
  </StrictMode>,
);
