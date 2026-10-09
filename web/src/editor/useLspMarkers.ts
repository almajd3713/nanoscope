import { useEffect, useState } from "react";
import { documentUri, toMarker } from "./lsp";
import type { LspState } from "./lspService";
import type { Marker } from "./types";

// The language server's findings for one open file, as markers: it publishes them after every
// change, so these follow the buffer, where ruff's follow the saved file.
export function useLspMarkers(path: string, service: LspState): Marker[] {
  const [found, setFound] = useState<{ uri: string; markers: Marker[] } | null>(null);
  const client = service.status === "connected" ? service.client : null;
  const uri = service.status === "connected" ? documentUri(service.root, path) : null;
  useEffect(() => {
    if (!client || !uri) return;
    return client.onDiagnostics((at, diagnostics) => {
      if (at === uri) setFound({ uri, markers: diagnostics.map(toMarker) });
    });
  }, [client, uri]);
  return found !== null && found.uri === uri ? found.markers : [];
}
