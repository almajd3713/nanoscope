import * as RadixTabs from "@radix-ui/react-tabs";
import type { ReactNode } from "react";
import styles from "./Tabs.module.css";

type Item = { id: string; label: string; content: ReactNode };

export function Tabs({ items, defaultId }: { items: Item[]; defaultId?: string }) {
  return (
    <RadixTabs.Root defaultValue={defaultId ?? items[0]?.id}>
      <RadixTabs.List className={styles.list}>
        {items.map((item) => (
          <RadixTabs.Trigger key={item.id} value={item.id} className={`${styles.tab} body`}>
            {item.label}
          </RadixTabs.Trigger>
        ))}
      </RadixTabs.List>
      {items.map((item) => (
        <RadixTabs.Content key={item.id} value={item.id} className={styles.panel}>
          {item.content}
        </RadixTabs.Content>
      ))}
    </RadixTabs.Root>
  );
}
