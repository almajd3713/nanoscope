import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { neighbours } from "../lessonOrder";
import { ButtonLink } from "./Button";
import styles from "./LessonNav.module.css";

// Previous and next lesson in catalog order. Shown on the lesson page and beside the lesson's file.
export function LessonNav({ id, size = "md" }: { id: string; size?: "sm" | "md" }) {
  const all = useQuery({ queryKey: ["curricula"], queryFn: () => unwrap(api.GET("/api/curricula")) });
  const { previous, next } = neighbours(all.data, id);
  if (!previous && !next) return null;
  return (
    <nav className={styles.nav} aria-label="Lesson order">
      {previous ? (
        <ButtonLink size={size} to={`/learn/${previous.id}`} title={previous.title}>
          Previous: {previous.title}
        </ButtonLink>
      ) : (
        <span />
      )}
      {next && (
        <ButtonLink size={size} to={`/learn/${next.id}`} title={next.title}>
          Next: {next.title}
        </ButtonLink>
      )}
    </nav>
  );
}
