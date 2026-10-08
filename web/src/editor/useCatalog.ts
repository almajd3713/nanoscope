import { useQuery } from "@tanstack/react-query";
import { useCallback } from "react";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { type Catalog, type CatalogBlock, type Suggestion, suggestionsAt } from "./completions";

// The block and preset catalogs the editor completes from (the same ones the palette and the
// run form read). Returns the completion function for the editor's surface.
export function useCatalog(): (textBefore: string) => Suggestion[] {
  const blocks = useQuery({
    queryKey: ["blocks"],
    queryFn: () => unwrap(api.GET("/api/blocks")) as unknown as Promise<{ blocks: CatalogBlock[] }>,
  });
  const presets = useQuery({
    queryKey: ["presets"],
    queryFn: () => unwrap(api.GET("/api/presets")) as unknown as Promise<{ name: string }[]>,
  });
  const catalog: Catalog = {
    blocks: blocks.data?.blocks ?? [],
    presets: (presets.data ?? []).map((p) => p.name),
  };
  const key = JSON.stringify([catalog.blocks.length, catalog.presets.length, blocks.dataUpdatedAt, presets.dataUpdatedAt]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useCallback((text: string) => suggestionsAt(text, catalog), [key]);
}
