import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { Lesson } from "../pages/Lesson";
import { Lessons } from "../pages/Lessons";
import { Components } from "../pages/Components";
import { Compare } from "../pages/Compare";
import { Settings } from "../pages/Settings";
import { Hardware } from "../pages/Hardware";
import { Studies } from "../pages/Studies";
import { Study } from "../pages/Study";
import { StudyBuilder } from "../pages/StudyBuilder";
import { Runs } from "../pages/Runs";
import { RunForm } from "../pages/RunForm";
import { Inspect } from "../pages/Inspect";
import { Run } from "../pages/Run";
import { Models } from "../pages/Models";
import { Onboarding } from "../pages/Onboarding";
import { BareLayout } from "./BareLayout";
import { Shell } from "./Shell";

// The model page carries the graph layout engine and the code editor: loaded when it is opened.
const Model = lazy(() => import("../pages/Model").then((m) => ({ default: m.Model })));

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<BareLayout />}>
        <Route path="/welcome" element={<Onboarding />} />
      </Route>
      <Route element={<Shell />}>
        <Route path="/" element={<Navigate to="/learn" replace />} />
        <Route path="/learn" element={<Lessons />} />
        <Route path="/learn/:path/:lesson" element={<Lesson />} />
        <Route path="/models" element={<Models />} />
        <Route path="/model/*" element={<Suspense fallback={<p className="small">Loading the model page</p>}><Model /></Suspense>} />
        <Route path="/runs" element={<Runs />} />
        <Route path="/runs/new" element={<RunForm />} />
        <Route path="/runs/*" element={<Run />} />
        <Route path="/inspect/*" element={<Inspect />} />
        <Route path="/compare" element={<Compare />} />
        <Route path="/studies" element={<Studies />} />
        <Route path="/studies/:name" element={<Study />} />
        <Route path="/studies/new" element={<StudyBuilder />} />
        <Route path="/studies/:name/edit" element={<StudyBuilder />} />
        <Route path="/hardware" element={<Hardware />} />
        <Route path="/components" element={<Components />} />
        <Route path="/settings" element={<Settings />} />
      </Route>
    </Routes>
  );
}
