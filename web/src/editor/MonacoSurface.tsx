import { useEffect, useRef } from "react";
import { monaco, followTheme, THEME } from "./monaco";
import { resolvedTheme } from "../styles/theme";
import type { Marker, SurfaceProps } from "./types";
import { readToken } from "../styles/theme";

const SEVERITY = {
  error: monaco.MarkerSeverity.Error,
  warning: monaco.MarkerSeverity.Warning,
  info: monaco.MarkerSeverity.Info,
} as const;

function toMarkers(markers: Marker[]): monaco.editor.IMarkerData[] {
  return markers.map((m) => ({
    severity: SEVERITY[m.severity],
    message: m.message,
    source: m.source,
    code: m.code ?? undefined,
    startLineNumber: m.line,
    startColumn: m.column,
    endLineNumber: m.endLine,
    endColumn: m.endColumn,
  }));
}

// The real editor. One model per path; the buffer's text belongs to the page, which passes it in.
export default function MonacoSurface({ path, value, onChange, onSave, markers, readOnly, revealLine, complete }: SurfaceProps) {
  const host = useRef<HTMLDivElement>(null);
  const editor = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);
  const latest = useRef({ onChange, onSave, complete });
  useEffect(() => {
    latest.current = { onChange, onSave, complete };
  });

  useEffect(() => {
    if (!host.current) return;
    followTheme(() => resolvedTheme() === "dark");
    const uri = monaco.Uri.parse(`inmemory://nanoscope/${path}`);
    const model = monaco.editor.getModel(uri) ?? monaco.editor.createModel(value, "python", uri);
    const instance = monaco.editor.create(host.current, {
      model,
      theme: THEME,
      automaticLayout: true,
      minimap: { enabled: false },
      fontFamily: readToken("font-mono") || "monospace",
      fontSize: 13,
      scrollBeyondLastLine: false,
      readOnly,
      tabSize: 4,
    });
    editor.current = instance;
    instance.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => latest.current.onSave());
    const sub = instance.onDidChangeModelContent(() => latest.current.onChange(instance.getValue()));
    const KIND = {
      block: monaco.languages.CompletionItemKind.Class,
      argument: monaco.languages.CompletionItemKind.Property,
      preset: monaco.languages.CompletionItemKind.Constant,
    } as const;
    const provider = monaco.languages.registerCompletionItemProvider("python", {
      triggerCharacters: ["(", ",", " ", '"', "'"],
      provideCompletionItems: (m, position) => {
        if (m.uri.toString() !== model.uri.toString() || !latest.current.complete) return { suggestions: [] };
        const before = m.getValueInRange({ startLineNumber: 1, startColumn: 1, endLineNumber: position.lineNumber, endColumn: position.column });
        const word = m.getWordUntilPosition(position);
        const range = { startLineNumber: position.lineNumber, endLineNumber: position.lineNumber, startColumn: word.startColumn, endColumn: word.endColumn };
        return {
          suggestions: latest.current.complete(before).map((s) => ({
            label: s.label,
            kind: KIND[s.kind],
            insertText: s.insertText,
            detail: s.detail,
            documentation: s.documentation,
            range,
          })),
        };
      },
    });
    return () => {
      provider.dispose();
      sub.dispose();
      instance.dispose();
      model.dispose();
      editor.current = null;
    };
    // the model and editor are made once per path; later values arrive through the effects below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path]);

  useEffect(() => {
    const instance = editor.current;
    if (instance && instance.getValue() !== value) instance.setValue(value);
  }, [value]);

  useEffect(() => {
    const model = editor.current?.getModel();
    if (model) monaco.editor.setModelMarkers(model, "nanoscope", toMarkers(markers));
  }, [markers]);

  useEffect(() => {
    if (revealLine) {
      editor.current?.revealLineInCenter(revealLine);
      editor.current?.setPosition({ lineNumber: revealLine, column: 1 });
      editor.current?.focus();
    }
  }, [revealLine]);

  return <div ref={host} style={{ height: "100%", minHeight: 240 }} />;
}
