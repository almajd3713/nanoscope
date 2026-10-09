import * as AlertDialog from "@radix-ui/react-alert-dialog";
import type { ReactNode } from "react";
import { Button } from "./Button";
import styles from "./Dialog.module.css";

type Props = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  children: ReactNode;
  confirm: string;
  onConfirm: () => void;
  busy?: boolean;
  // the label of the safe choice (default "Cancel")
  cancel?: string;
};

// A confirm step for something that changes a lot. Cancel has the focus, so Enter does not
// confirm by accident; the confirm button says what it does.
export function ConfirmDialog({ open, onOpenChange, title, children, confirm, onConfirm, busy, cancel = "Cancel" }: Props) {
  return (
    <AlertDialog.Root open={open} onOpenChange={onOpenChange}>
      <AlertDialog.Portal>
        <AlertDialog.Overlay className={styles.scrim} />
        <AlertDialog.Content className={styles.dialog}>
          <AlertDialog.Title className="heading">{title}</AlertDialog.Title>
          <AlertDialog.Description asChild>
            <div className={`${styles.body} body`}>{children}</div>
          </AlertDialog.Description>
          <div className={styles.foot}>
            <AlertDialog.Cancel asChild>
              <Button>{cancel}</Button>
            </AlertDialog.Cancel>
            <Button variant="danger" disabled={busy} onClick={onConfirm}>
              {confirm}
            </Button>
          </div>
        </AlertDialog.Content>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  );
}
