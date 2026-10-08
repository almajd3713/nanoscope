import * as AlertDialog from "@radix-ui/react-alert-dialog";
import { Button } from "./Button";
import styles from "./ConflictDialog.module.css";
import dialog from "./Dialog.module.css";

type Props = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  path: string;
  // the server's unified diff: what is on disk against what you sent
  diff: string;
  busy?: boolean;
  onKeepMine: () => void;
  onTakeTheirs: () => void;
};

// One diff line: a − or + sign always accompanies the red or green, so colour is never alone.
function DiffLine({ text }: { text: string }) {
  const kind = text.startsWith("+++") || text.startsWith("---") || text.startsWith("@@")
    ? "hunk"
    : text.startsWith("+")
      ? "add"
      : text.startsWith("-")
        ? "del"
        : "";
  const shown = kind === "del" || kind === "add" ? (kind === "del" ? "−" : "+") + text.slice(1) : text;
  return <span className={`${styles.line} ${kind ? styles[kind] : ""}`}>{shown || " "}</span>;
}

// The server refused a save because the file changed since you read it. Nothing was written;
// you decide whose text stays.
export function ConflictDialog({ open, onOpenChange, path, diff, busy, onKeepMine, onTakeTheirs }: Props) {
  const lines = diff.split("\n");
  if (lines[lines.length - 1] === "") lines.pop();
  return (
    <AlertDialog.Root open={open} onOpenChange={onOpenChange}>
      <AlertDialog.Portal>
        <AlertDialog.Overlay className={dialog.scrim} />
        <AlertDialog.Content className={dialog.dialog} style={{ maxWidth: 720 }}>
          <AlertDialog.Title className="heading">{path} changed on disk</AlertDialog.Title>
          <AlertDialog.Description asChild>
            <div className={`${dialog.body} body`}>
              <p>
                Someone or something saved this file after you opened it, so nothing was written. Below, − is what is on disk now and +
                is your text.
              </p>
              <pre className={`${styles.diff} code`} aria-label="Differences">
                {lines.map((line, i) => (
                  <DiffLine key={i} text={line} />
                ))}
              </pre>
            </div>
          </AlertDialog.Description>
          <div className={dialog.foot}>
            <AlertDialog.Cancel asChild>
              <Button variant="quiet">Decide later</Button>
            </AlertDialog.Cancel>
            <Button disabled={busy} onClick={onTakeTheirs}>
              Take theirs
            </Button>
            <Button variant="danger" disabled={busy} onClick={onKeepMine}>
              Keep mine
            </Button>
          </div>
        </AlertDialog.Content>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  );
}
