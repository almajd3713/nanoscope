import * as RadixTabs from "@radix-ui/react-tabs";
import type { ReactNode } from "react";
import styles from "./Tabs.module.css";

type Item = { id: string; label: string; content: ReactNode };

type Props = {
  items: Item[];
  defaultId?: string;
  // controlled use: the selected id and its change handler
  value?: string;
  onValueChange?: (id: string) => void;
};

export function Tabs({ items, defaultId, value, onValueChange }: Props) {
  return (
    <RadixTabs.Root
      {...(value !== undefined ? { value, onValueChange } : { defaultValue: defaultId ?? items[0]?.id })}
    >
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
