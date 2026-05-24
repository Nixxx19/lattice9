import { useState } from "react";
import { Routes, Route, NavLink } from "react-router-dom";
import Overview from "./pages/Overview";
import Inference from "./pages/Inference";
import { InferenceCtx, InferenceSession, emptySession } from "./state";

function App() {
  const [session, setSession] = useState<InferenceSession>(emptySession);
  const update = (patch: Partial<InferenceSession>) =>
    setSession((prev) => ({ ...prev, ...patch }));

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-1.5 rounded-md text-base transition-colors ${
      isActive
        ? "bg-zinc-800 text-zinc-100"
        : "text-zinc-400 hover:text-zinc-200"
    }`;

  return (
    <InferenceCtx.Provider value={{ session, update }}>
      <div className="min-h-screen bg-zinc-950 text-zinc-100">
        <header className="border-b border-zinc-900 bg-zinc-950/80 backdrop-blur sticky top-0 z-50">
          <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded bg-zinc-100 text-zinc-950 flex items-center justify-center font-semibold text-base">
                p
              </div>
              <h1 className="text-xl font-semibold tracking-tight">plasma-mesh</h1>
              <span className="text-xs text-zinc-500 border border-zinc-800 px-2 py-0.5 rounded">
                distributed inference
              </span>
            </div>
            <nav className="flex gap-1">
              <NavLink to="/" className={linkClass} end>
                overview
              </NavLink>
              <NavLink to="/inference" className={linkClass}>
                inference
              </NavLink>
            </nav>
          </div>
        </header>

        <main className="max-w-6xl mx-auto px-6 py-10">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/inference" element={<Inference />} />
          </Routes>
        </main>
      </div>
    </InferenceCtx.Provider>
  );
}

export default App;
