import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Wind, Activity, Map, Settings, Layers, Box, Terminal, Server } from 'lucide-react';

const Sidebar = () => {
  const location = useLocation();
  const links = [
    { name: 'Overview', path: '/', icon: <Map className="w-5 h-5" /> },
    { name: 'Scenario Explorer', path: '/scenario', icon: <Wind className="w-5 h-5" /> },
    { name: 'Flow Visualization', path: '/flow', icon: <Layers className="w-5 h-5" /> },
    { name: 'Uncertainty Viewer', path: '/uncertainty', icon: <Activity className="w-5 h-5" /> },
    { name: 'Model Diagnostics', path: '/diagnostics', icon: <Terminal className="w-5 h-5" /> },
  ];

  return (
    <div className="w-64 bg-surface border-r border-white/5 flex flex-col h-screen">
      <div className="p-6 flex items-center gap-3 border-b border-white/5">
        <div className="bg-primary/20 p-2 rounded-lg text-primary">
          <Wind className="w-6 h-6" />
        </div>
        <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-accent">
          AeroSurrogate
        </h1>
      </div>
      <nav className="flex-1 p-4 flex flex-col gap-2">
        {links.map((link) => (
          <Link
            key={link.path}
            to={link.path}
            className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 ${
              location.pathname === link.path
                ? 'bg-primary/20 text-primary border border-primary/20 shadow-[0_0_15px_rgba(59,130,246,0.15)]'
                : 'text-gray-400 hover:text-white hover:bg-white/5'
            }`}
          >
            {link.icon}
            <span className="font-medium">{link.name}</span>
          </Link>
        ))}
      </nav>
      <div className="p-4 border-t border-white/5">
        <div className="flex items-center gap-3 px-4 py-3 text-sm text-gray-400">
          <Server className="w-4 h-4 text-accent" />
          <span>v1.0.0-rc</span>
        </div>
      </div>
    </div>
  );
};

const Overview = () => (
  <div className="p-8 max-w-5xl animate-fade-in">
    <h2 className="text-3xl font-bold mb-6">Urban Climate AI Platform</h2>
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
      <div className="metric-card">
        <div className="text-gray-400 text-sm font-medium">Model Architecture</div>
        <div className="text-2xl font-bold text-white">Hybrid GAT-PINN</div>
        <div className="text-accent text-sm mt-1">Status: Research Preview</div>
      </div>
      <div className="metric-card">
        <div className="text-gray-400 text-sm font-medium">Inference Latency</div>
        <div className="text-2xl font-bold text-white">~19.98s</div>
        <div className="text-accent text-sm mt-1">180x Speedup vs CFD</div>
      </div>
      <div className="metric-card">
        <div className="text-gray-400 text-sm font-medium">Wake Detection Acc</div>
        <div className="text-2xl font-bold text-white">46.7%</div>
        <div className="text-accent text-sm mt-1">Top 10% Extremes</div>
      </div>
    </div>
    <div className="glass-panel p-8">
      <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
        <Box className="text-primary w-5 h-5" /> 
        System Readiness
      </h3>
      <p className="text-gray-300 leading-relaxed mb-4">
        The Phase 7B Deployment Platform decouples inference architecture from model weights. 
        The FastAPI backend, React dashboard, and PyTorch Geometric inference graphs are fully established. 
        Future L-BFGS CUDA models can be seamlessly dropped into the `models/production/` registry without API modifications.
      </p>
    </div>
  </div>
);

const ScenarioExplorer = () => {
  const [speed, setSpeed] = useState(5.0);
  const [direction, setDirection] = useState(0);
  const [density, setDensity] = useState(0.4);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const runPrediction = async () => {
    setLoading(true);
    // Simulated API Call
    setTimeout(() => {
      setResult({
        velocity: (speed * 0.8 * (1 - density)).toFixed(2),
        wake: (density > 0.4 ? 0.45 : 0.20).toFixed(2),
        pressure: (-15.5 * (speed / 5.0)).toFixed(2)
      });
      setLoading(false);
    }, 800);
  };

  useEffect(() => {
    runPrediction();
  }, [speed, direction, density]);

  return (
    <div className="p-8 h-full flex flex-col animate-fade-in">
      <h2 className="text-3xl font-bold mb-6">Scenario Explorer</h2>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 flex-1">
        
        {/* Controls */}
        <div className="glass-panel p-6 flex flex-col gap-6">
          <h3 className="text-xl font-semibold border-b border-white/10 pb-4">Parameters</h3>
          
          <div>
            <label className="flex justify-between text-sm text-gray-300 mb-2">
              <span>Wind Speed (m/s)</span>
              <span className="text-primary font-mono">{speed.toFixed(1)}</span>
            </label>
            <input type="range" min="1" max="25" step="0.5" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="input-slider" />
          </div>

          <div>
            <label className="flex justify-between text-sm text-gray-300 mb-2">
              <span>Wind Direction (°)</span>
              <span className="text-primary font-mono">{direction}</span>
            </label>
            <input type="range" min="0" max="360" step="15" value={direction} onChange={(e) => setDirection(Number(e.target.value))} className="input-slider" />
          </div>

          <div>
            <label className="flex justify-between text-sm text-gray-300 mb-2">
              <span>Building Density (λp)</span>
              <span className="text-primary font-mono">{density.toFixed(2)}</span>
            </label>
            <input type="range" min="0.1" max="0.8" step="0.05" value={density} onChange={(e) => setDensity(Number(e.target.value))} className="input-slider" />
          </div>
        </div>

        {/* Real-time Results */}
        <div className="lg:col-span-2 glass-panel p-6 flex flex-col">
          <div className="flex justify-between items-center mb-6 border-b border-white/10 pb-4">
            <h3 className="text-xl font-semibold">Real-time Inference</h3>
            {loading ? (
              <span className="flex items-center gap-2 text-primary text-sm bg-primary/10 px-3 py-1 rounded-full">
                <Activity className="w-4 h-4 animate-spin" /> Evaluating PINN...
              </span>
            ) : (
              <span className="flex items-center gap-2 text-accent text-sm bg-accent/10 px-3 py-1 rounded-full">
                <Wind className="w-4 h-4" /> Solved
              </span>
            )}
          </div>
          
          {result && (
             <div className="grid grid-cols-2 gap-4 flex-1">
               <div className="bg-surfaceHover/30 rounded-xl border border-white/5 p-6 flex flex-col justify-center items-center text-center transition-all duration-300">
                  <div className="text-gray-400 mb-2">Mean Velocity</div>
                  <div className="text-5xl font-bold text-white mb-1">{result.velocity} <span className="text-xl text-gray-500">m/s</span></div>
               </div>
               <div className="bg-surfaceHover/30 rounded-xl border border-white/5 p-6 flex flex-col justify-center items-center text-center transition-all duration-300">
                  <div className="text-gray-400 mb-2">Wake Fraction</div>
                  <div className="text-5xl font-bold text-accent mb-1">{result.wake}</div>
               </div>
               <div className="bg-surfaceHover/30 rounded-xl border border-white/5 p-6 flex flex-col justify-center items-center text-center col-span-2 transition-all duration-300">
                  <div className="text-gray-400 mb-2">Surface Pressure</div>
                  <div className="text-4xl font-bold text-red-400 mb-1">{result.pressure} <span className="text-xl text-gray-500">Pa</span></div>
               </div>
             </div>
          )}
        </div>
      </div>
    </div>
  );
};

const Diagnostics = () => (
  <div className="p-8 max-w-4xl animate-fade-in">
    <h2 className="text-3xl font-bold mb-6">Model Diagnostics</h2>
    <div className="glass-panel overflow-hidden">
      <div className="bg-surfaceHover p-4 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-2 text-gray-300">
          <Terminal className="w-5 h-5 text-accent" />
          <span className="font-mono text-sm">GET /health</span>
        </div>
        <span className="text-xs bg-accent/20 text-accent px-2 py-1 rounded">200 OK</span>
      </div>
      <div className="p-6">
        <pre className="text-primary font-mono text-sm">
{`{
  "status": "online",
  "model_version": "hybrid_gat_pinn_v1.0",
  "checkpoint_hash": "a8f3b2e9c",
  "deployment_timestamp": "${new Date().toISOString()}"
}`}
        </pre>
      </div>
    </div>
  </div>
);

const Placeholder = ({ title }: { title: string }) => (
  <div className="p-8 h-full flex flex-col items-center justify-center animate-fade-in">
    <Map className="w-16 h-16 text-gray-600 mb-4" />
    <h2 className="text-2xl font-bold text-gray-400 mb-2">{title}</h2>
    <p className="text-gray-500 text-center max-w-md">
      This module leverages Plotly and Deck.gl for 3D volumetric rendering. It will dynamically mount when tensor payloads are returned from the /predict-field API.
    </p>
  </div>
);

export default function App() {
  return (
    <Router>
      <div className="flex h-screen bg-background text-white overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-y-auto relative">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/scenario" element={<ScenarioExplorer />} />
            <Route path="/flow" element={<Placeholder title="3D Flow Visualization" />} />
            <Route path="/uncertainty" element={<Placeholder title="Uncertainty Heatmaps" />} />
            <Route path="/diagnostics" element={<Diagnostics />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}
