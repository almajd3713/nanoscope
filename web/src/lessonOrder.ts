// The catalog is a flat reading order: the API sends courses sorted by their `order`, and
// each course's lessons by number, so the lesson before or after one may be in another course.
type Entry = { id: string; title: string };
type Course = { id: string; title: string; lessons: Entry[] };

export type Neighbours = { previous: Entry | null; next: Entry | null };

export function neighbours(courses: Course[] | undefined, id: string): Neighbours {
  const flat = (courses ?? []).flatMap((c) => c.lessons.map((l) => ({ id: l.id, title: l.title })));
  const at = flat.findIndex((l) => l.id === id);
  if (at < 0) return { previous: null, next: null };
  return { previous: flat[at - 1] ?? null, next: flat[at + 1] ?? null };
}
