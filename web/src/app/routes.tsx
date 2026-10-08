import { Navigate, Route, Routes } from "react-router-dom";
import { Shell } from "./Shell";

// Each screen replaces its title here as its checklist item lands.
function Pending({ title }: { title: string }) {
  return <h1 className="title">{title}</h1>;
}

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route path="/" element={<Navigate to="/learn" replace />} />
        <Route path="/learn" element={<Pending title="Lessons" />} />
        <Route path="/runs" element={<Pending title="Runs" />} />
        <Route path="/compare" element={<Pending title="Compare" />} />
        <Route path="/studies" element={<Pending title="Studies" />} />
        <Route path="/hardware" element={<Pending title="Hardware" />} />
        <Route path="/components" element={<Pending title="Components" />} />
        <Route path="/settings" element={<Pending title="Settings" />} />
      </Route>
    </Routes>
  );
}
