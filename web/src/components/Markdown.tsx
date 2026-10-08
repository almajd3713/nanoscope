import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import styles from "./Markdown.module.css";

// Lesson text. Raw HTML in the source is not rendered (react-markdown escapes it) and unsafe
// link schemes are dropped, so lesson.md cannot inject markup.
export function Markdown({ children }: { children: string }) {
  return (
    <div className={`${styles.prose} prose`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h2: ({ node: _node, ...p }) => <h2 className="prose-heading" {...p} />,
          h3: ({ node: _node, ...p }) => <h3 className="prose-subheading" {...p} />,
          code: ({ node: _node, className, ...p }) => (
            <code className={className ?? "value"} {...p} />
          ),
          a: ({ node: _node, ...p }) => <a rel="noopener noreferrer" {...p} />,
          table: ({ node: _node, ...p }) => (
            <div className={styles.tableWrap}>
              <table {...p} />
            </div>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
