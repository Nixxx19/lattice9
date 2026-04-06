import { Routes, Route, NavLink } from "react-router-dom";
import Overview from "./pages/Overview";
import Inference from "./pages/Inference";

function App() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
      isActive
        ? "bg-indigo-600 text-white"
        : "text-gray-400 hover:text-white hover:bg-gray-800"
    }`;

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-sm">
              H
            </div>
            <h1 className="text-xl font-bold text-white">Hivemind</h1>
            <span className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded-full">
              Distributed Inference
            </span>
          </div>
          <nav className="flex gap-2">
            <NavLink to="/" className={linkClass} end>
              Overview
            </NavLink>
            <NavLink to="/inference" className={linkClass}>
              Inference
            </NavLink>
          </nav>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/inference" element={<Inference />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
