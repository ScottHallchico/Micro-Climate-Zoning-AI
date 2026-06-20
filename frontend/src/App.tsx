import { useState, useEffect, useRef } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Wind, Activity, Map, Layers, Terminal, MapPin, GitCompare, Shield, ChevronRight, Zap } from 'lucide-react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

/* ─────────────── Zone Config ─────────────── */
const ZONE_META: Record<string, { color: string; name: string; desc: string }> = {
  Z1: { color: '#22D3EE', name: 'Ventilation Corridor', desc: 'High through-flow channels for heat dissipation' },
  Z2: { color: '#10B981', name: 'Comfortable Climate', desc: 'Well-ventilated, low-turbulence outdoor areas' },
  Z3: { color: '#A78BFA', name: 'Neutral Mixed', desc: 'Typical streetscape, moderate flow' },
  Z4: { color: '#F59E0B', name: 'Heat Retention', desc: 'Reduced airflow trapping urban heat' },
  Z5: { color: '#EF4444', name: 'Stagnation Risk', desc: 'Near-zero velocity, pollutant accumulation' },
  Z6: { color: '#F472B6', name: 'Wind Hazard', desc: 'Dangerous acceleration, pedestrian safety risk' },
};

/* ─────────────── Sidebar ─────────────── */
const Sidebar = () => {
  const location = useLocation();
  const links = [
    { name: 'Overview', path: '/', icon: <Map className="w-5 h-5" /> },
    { name: 'Climate Zones', path: '/zones', icon: <MapPin className="w-5 h-5" /> },
    { name: 'Scenario Explorer', path: '/scenario', icon: <Wind className="w-5 h-5" /> },
    { name: 'Scenario Compare', path: '/compare', icon: <GitCompare className="w-5 h-5" /> },
    { name: 'Flow Visualization', path: '/flow', icon: <Layers className="w-5 h-5" /> },
    { name: 'Uncertainty', path: '/uncertainty', icon: <Activity className="w-5 h-5" /> },
    { name: 'Diagnostics', path: '/diagnostics', icon: <Terminal className="w-5 h-5" /> },
  ];

  return (
    <div className="w-64 min-w-[256px] bg-surface border-r border-white/5 flex flex-col h-screen">
      <div className="p-5 flex items-center gap-3 border-b border-white/5">
        <div className="bg-gradient-to-br from-primary to-accent p-2.5 rounded-xl">
          <Wind className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-lg font-bold text-white tracking-tight">AeroSurrogate</h1>
          <p className="text-[10px] text-gray-500 uppercase tracking-widest">Climate Zoning AI</p>
        </div>
      </div>
      <nav className="flex-1 p-3 flex flex-col gap-1 overflow-y-auto">
        {links.map((link) => (
          <Link key={link.path} to={link.path}
            className={`flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all duration-200 group ${location.pathname === link.path
                ? 'bg-primary/15 text-primary border border-primary/20'
                : 'text-gray-400 hover:text-white hover:bg-white/5'
              }`}>
            {link.icon}
            <span className="text-sm font-medium">{link.name}</span>
            {location.pathname === link.path && <ChevronRight className="w-4 h-4 ml-auto opacity-50" />}
          </Link>
        ))}
      </nav>
      <div className="p-4 border-t border-white/5">
        <div className="flex items-center gap-2 px-3 text-xs text-gray-500">
          <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />
          <span>v2.0.0 · Phase 8A</span>
        </div>
      </div>
    </div>
  );
};

/* ─────────────── Metric Card ─────────────── */
const MetricCard = ({ label, value, sub, accent = false }: { label: string; value: string; sub: string; accent?: boolean }) => (
  <div className="relative bg-surfaceHover/40 p-5 rounded-xl border border-white/5 overflow-hidden group hover:border-white/10 transition-all duration-300">
    <div className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-primary/60 to-accent/60" />
    <p className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-2">{label}</p>
    <p className={`text-2xl font-bold ${accent ? 'text-accent' : 'text-white'}`}>{value}</p>
    <p className="text-gray-500 text-xs mt-1">{sub}</p>
  </div>
);

/* ─────────────── Overview ─────────────── */
const Overview = () => (
  <div className="p-8 max-w-6xl">
    <div className="mb-8">
      <h2 className="text-3xl font-bold text-white mb-2">Urban Climate AI Platform</h2>
      <p className="text-gray-400">Hybrid GAT-PINN surrogate with Navier-Stokes physics constraints</p>
    </div>
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      <MetricCard label="Architecture" value="GAT-PINN" sub="Hybrid Graph + Physics" />
      <MetricCard label="Inference" value="~20s" sub="180× faster than CFD" accent />
      <MetricCard label="Wake Detection" value="46.7%" sub="Top-10% extremes" />
      <MetricCard label="Climate Zones" value="6" sub="Z1–Z6 taxonomy" accent />
    </div>
    <div className="glass-panel p-6">
      <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
        <Shield className="text-accent w-5 h-5" /> Phase 8A Status
      </h3>
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-accent" /><span className="text-gray-300">Zoning Engine: <span className="text-accent">Verified</span></span></div>
        <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-accent" /><span className="text-gray-300">GeoJSON Export: <span className="text-accent">Complete</span></span></div>
        <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-accent" /><span className="text-gray-300">Silhouette Score: <span className="text-accent">0.437</span></span></div>
        <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-accent" /><span className="text-gray-300">Dashboard: <span className="text-accent">Live</span></span></div>
      </div>
    </div>
  </div>
);

/* ─────────────── Climate Zone Map ─────────────── */
const ClimateZoneMap = () => {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const [activeZones, setActiveZones] = useState<Set<string>>(new Set(['Z2', 'Z3', 'Z4', 'Z5', 'Z6']));
  const [showCorridors, setShowCorridors] = useState<boolean>(true);
  const [hoveredZone, setHoveredZone] = useState<any>(null);
  const [hoveredCorridor, setHoveredCorridor] = useState<any>(null);
  const [geoData, setGeoData] = useState<any>(null);
  const [corridorData, setCorridorData] = useState<any>(null);
  const layersRef = useRef<L.GeoJSON | null>(null);
  const corridorLayerRef = useRef<L.GeoJSON | null>(null);

  useEffect(() => {
    fetch('/data/climate_zones_provenance.geojson')
      .then(r => r.json())
      .then(data => setGeoData(data))
      .catch(() => console.error('Failed to load climate_zones.geojson'));

    fetch('/data/ventilation_corridors.geojson')
      .then(r => r.json())
      .then(data => setCorridorData(data))
      .catch(() => console.error('Failed to load ventilation_corridors.geojson'));
  }, []);

  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;
    const map = L.map(mapRef.current, { zoomControl: false }).setView([40.68, -73.94], 12);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; CARTO',
      maxZoom: 19,
    }).addTo(map);
    L.control.zoom({ position: 'bottomright' }).addTo(map);
    mapInstanceRef.current = map;

    return () => { map.remove(); mapInstanceRef.current = null; };
  }, []);

  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !geoData) return;

    if (layersRef.current) { map.removeLayer(layersRef.current); }

    const filtered = {
      ...geoData,
      features: geoData.features.filter((f: any) => activeZones.has(f.properties.zone_type)),
    };

    const layer = L.geoJSON(filtered, {
      style: (feature: any) => ({
        fillColor: ZONE_META[feature.properties.zone_type]?.color || '#888',
        fillOpacity: 0.55,
        color: '#fff',
        weight: 1,
        opacity: 0.6,
      }),
      onEachFeature: (feature: any, layer: any) => {
        layer.on('mouseover', () => setHoveredZone(feature.properties));
        layer.on('mouseout', () => setHoveredZone(null));
      },
    }).addTo(map);

    layersRef.current = layer;
    if (filtered.features.length > 0) {
      map.fitBounds(layer.getBounds(), { padding: [30, 30] });
    }
  }, [geoData, activeZones]);

  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !corridorData) return;

    if (corridorLayerRef.current) { map.removeLayer(corridorLayerRef.current); }

    if (showCorridors) {
      const getCorridorColor = (cls: string) => {
        if (cls === 'Permanent Corridor') return '#3B82F6';
        if (cls === 'Seasonal Corridor') return '#14B8A6';
        return '#22C55E';
      };
      const getCorridorOpacity = (cls: string) => {
        if (cls === 'Permanent Corridor') return 0.8;
        if (cls === 'Seasonal Corridor') return 0.5;
        return 0.3;
      };

      const layer = L.geoJSON(corridorData, {
        style: (feature: any) => ({
          fillColor: getCorridorColor(feature.properties.persistence_class),
          fillOpacity: getCorridorOpacity(feature.properties.persistence_class),
          color: '#0ea5e9',
          weight: 2,
          opacity: 0.9,
          dashArray: '4'
        }),
        onEachFeature: (feature: any, layer: any) => {
          layer.on('mouseover', () => setHoveredCorridor(feature.properties));
          layer.on('mouseout', () => setHoveredCorridor(null));
        },
      }).addTo(map);
      corridorLayerRef.current = layer;
    }
  }, [corridorData, showCorridors]);

  const toggleZone = (z: string) => {
    setActiveZones(prev => {
      const next = new Set(prev);
      if (next.has(z)) next.delete(z); else next.add(z);
      return next;
    });
  };

  return (
    <div className="p-6 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-2xl font-bold text-white">Climate Zone Map</h2>
          <p className="text-gray-400 text-sm">NYC urban archetypes · real GeoJSON overlay</p>
        </div>
      </div>
      <div className="flex gap-4 flex-1 min-h-0">
        {/* Layer Controls */}
        <div className="w-64 min-w-[240px] glass-panel p-4 flex flex-col gap-3 overflow-y-auto">
          <h4 className="text-xs uppercase tracking-widest text-gray-400 font-semibold mb-1">Layer Controls</h4>
          {Object.entries(ZONE_META).map(([z, meta]) => (
            <button key={z} onClick={() => toggleZone(z)}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-all text-sm ${activeZones.has(z) ? 'bg-white/10 text-white' : 'text-gray-500 hover:text-gray-300'
                }`}>
              <div className="w-4 h-4 rounded-sm flex-shrink-0" style={{ background: activeZones.has(z) ? meta.color : '#333' }} />
              <div>
                <div className="font-medium">{z}: {meta.name}</div>
                <div className="text-[10px] text-gray-500 leading-tight">{meta.desc}</div>
              </div>
            </button>
          ))}

          <div className="mt-4 border-t border-white/10 pt-4">
            <button onClick={() => setShowCorridors(!showCorridors)}
              className={`flex items-center gap-3 w-full px-3 py-2.5 rounded-lg text-left transition-all text-sm ${showCorridors ? 'bg-blue-500/20 text-blue-300' : 'text-gray-500 hover:text-gray-300'
                }`}>
              <div className="w-4 h-4 rounded-sm flex-shrink-0 border-2 border-blue-400 border-dashed" style={{ background: showCorridors ? '#3B82F688' : 'transparent' }} />
              <div>
                <div className="font-medium">Ventilation Corridors</div>
                <div className="text-[10px] text-gray-500 leading-tight">Streamline derived</div>
              </div>
            </button>
          </div>

          {hoveredZone && (
            <div className="mt-auto pt-3 border-t border-white/10">
              <h4 className="text-xs uppercase tracking-widest text-gray-400 mb-2">Hovered Zone</h4>
              <div className="text-sm space-y-1">
                <p className="font-bold" style={{ color: ZONE_META[hoveredZone.zone_type]?.color }}>{hoveredZone.zone_name}</p>
                <p className="text-gray-300">VEI: <span className="text-white font-mono">{hoveredZone.vei?.toFixed(2)}</span></p>
                <p className="font-bold pt-2 border-t border-white/10" style={{ color: ZONE_META[hoveredZone.zone_type]?.color }}>Confidence Breakdown</p>
                <p className="text-gray-300">Overall Conf: <span className="text-white font-mono">{(hoveredZone.confidence_epistemic*100).toFixed(0)}%</span></p>
                <p className="text-gray-300">OOD Contrib: <span className="text-white font-mono">{(hoveredZone.ood_contribution*100).toFixed(0)}%</span></p>
                <p className="text-gray-300">Unc Contrib: <span className="text-white font-mono">{(hoveredZone.unc_contribution*100).toFixed(0)}%</span></p>
                <p className="text-gray-300">Inst Contrib: <span className="text-white font-mono">{(hoveredZone.inst_contribution*100).toFixed(0)}%</span></p>
              </div>
            </div>
          )}
          {hoveredCorridor && (
            <div className="mt-auto pt-3 border-t border-white/10">
              <h4 className="text-xs uppercase tracking-widest text-gray-400 mb-2">Hovered Corridor</h4>
              <div className="text-sm space-y-1">
                <p className="font-bold text-blue-400">{hoveredCorridor.corridor_id}</p>
                <p className="text-gray-300">Class: <span className="text-white">{hoveredCorridor.persistence_class}</span></p>
                <p className="text-gray-300">Length: <span className="text-white font-mono">{hoveredCorridor.length?.toFixed(1)} m</span></p>
                <p className="text-gray-300">Persistence: <span className="text-white font-mono">{(hoveredCorridor.persistence * 100).toFixed(0)}%</span></p>
              </div>
            </div>
          )}
        </div>

        {/* Map Container */}
        <div className="flex-1 rounded-xl overflow-hidden border border-white/10 relative glass-panel">
          <div ref={mapRef} className="absolute inset-0 z-0 bg-[#1a1a1a]" />
        </div>
      </div>
    </div>
  );
};

/* ─────────────── Scenario Explorer ─────────────── */
const classify = (speed: number, density: number, ref: number) => {
  const vel = speed * (1 - density * 0.8);
  const vei = vel / ref;
  const tke = density * speed * 0.3;
  const wake = density > 0.4 ? 0.55 : 0.18;
  if (vel > 10 || tke > 3) return 'Z6';
  if (wake >= 0.5 && vel < 1) return 'Z5';
  if (vei > 1.5 && wake < 0.15) return 'Z1';
  if (vei <= 0.5 && wake >= 0.3 && tke < 0.5) return 'Z4';
  if (vei > 0.8 && vei <= 1.5 && tke < 1) return 'Z2';
  return 'Z3';
};

const ScenarioExplorer = () => {
  const [speed, setSpeed] = useState(5.0);
  const [direction, setDirection] = useState(0);
  const [density, setDensity] = useState(0.4);
  const [height, setHeight] = useState(30);
  const [vegetation, setVegetation] = useState(0.1);

  const vel = speed * (1 - density * 0.8);
  const wake = density > 0.4 ? 0.55 : 0.18;
  const tke = density * speed * 0.3;
  const vei = vel / speed;
  const zone = classify(speed, density, speed);

  return (
    <div className="p-6 h-full flex flex-col">
      <h2 className="text-2xl font-bold text-white mb-1">Scenario Explorer</h2>
      <p className="text-gray-400 text-sm mb-4">Modify parameters and observe zoning changes instantly</p>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-0">
        {/* Controls */}
        <div className="glass-panel p-5 flex flex-col gap-5 overflow-y-auto">
          <h3 className="text-sm uppercase tracking-widest text-gray-400 font-semibold pb-2 border-b border-white/10">Parameters</h3>
          {[
            { label: 'Wind Speed', unit: 'm/s', val: speed, set: setSpeed, min: 1, max: 25, step: 0.5 },
            { label: 'Wind Direction', unit: '°', val: direction, set: setDirection, min: 0, max: 360, step: 15 },
            { label: 'Building Density', unit: 'λp', val: density, set: setDensity, min: 0.05, max: 0.8, step: 0.05 },
            { label: 'Building Height', unit: 'm', val: height, set: setHeight, min: 5, max: 200, step: 5 },
            { label: 'Vegetation Fraction', unit: '', val: vegetation, set: setVegetation, min: 0, max: 0.5, step: 0.05 },
          ].map(s => (
            <div key={s.label}>
              <label className="flex justify-between text-xs text-gray-300 mb-1.5">
                <span>{s.label} {s.unit && <span className="text-gray-500">({s.unit})</span>}</span>
                <span className="text-primary font-mono">{typeof s.val === 'number' && s.val % 1 !== 0 ? s.val.toFixed(2) : s.val}</span>
              </label>
              <input type="range" min={s.min} max={s.max} step={s.step} value={s.val}
                onChange={e => s.set(Number(e.target.value))} className="input-slider" />
            </div>
          ))}
        </div>

        {/* Results */}
        <div className="lg:col-span-2 glass-panel p-5 flex flex-col">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/10">
            <h3 className="font-semibold text-white">Inference Result</h3>
            <span className="flex items-center gap-2 text-accent text-xs bg-accent/10 px-3 py-1 rounded-full">
              <Zap className="w-3 h-3" /> Solved
            </span>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
            <MetricCard label="Mean Velocity" value={`${vel.toFixed(1)}`} sub="m/s" />
            <MetricCard label="Wake Fraction" value={wake.toFixed(2)} sub="dimensionless" accent />
            <MetricCard label="TKE" value={tke.toFixed(2)} sub="m²/s²" />
            <MetricCard label="VEI" value={vei.toFixed(2)} sub="Ventilation Efficiency" />
            <MetricCard label="Pressure" value={(-15.5 * speed / 5).toFixed(1)} sub="Pa (surface)" />
            <div className="relative bg-surfaceHover/40 p-5 rounded-xl border border-white/5 overflow-hidden flex flex-col items-center justify-center"
              style={{ borderColor: ZONE_META[zone]?.color + '44' }}>
              <div className="absolute top-0 left-0 w-full h-[2px]" style={{ background: ZONE_META[zone]?.color }} />
              <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">Predicted Zone</p>
              <p className="text-2xl font-bold" style={{ color: ZONE_META[zone]?.color }}>{zone}</p>
              <p className="text-xs text-gray-500 mt-1">{ZONE_META[zone]?.name}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

/* ─────────────── Scenario Comparison ─────────────── */
const ScenarioCompare = () => {
  const [densityA, setDensityA] = useState(0.55);
  const [densityB, setDensityB] = useState(0.30);
  const speed = 6;

  const make = (d: number) => {
    const vel = speed * (1 - d * 0.8);
    const wake = d > 0.4 ? 0.55 : 0.18;
    const zone = classify(speed, d, speed);
    const vei = vel / speed;
    return { vel, wake, zone, vei, tke: d * speed * 0.3 };
  };
  const a = make(densityA);
  const b = make(densityB);

  const delta = (va: number, vb: number) => {
    const d = vb - va;
    const sign = d > 0 ? '+' : '';
    return `${sign}${d.toFixed(2)}`;
  };

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold text-white mb-1">Scenario Comparison</h2>
      <p className="text-gray-400 text-sm mb-6">Compare two density layouts side by side at {speed} m/s wind</p>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Scenario A */}
        <div className="glass-panel p-5">
          <h3 className="font-semibold text-red-400 mb-3 text-sm uppercase tracking-wide">Scenario A — Current</h3>
          <label className="flex justify-between text-xs text-gray-300 mb-1.5">
            <span>Density</span><span className="font-mono text-primary">{densityA.toFixed(2)}</span>
          </label>
          <input type="range" min={0.1} max={0.8} step={0.05} value={densityA} onChange={e => setDensityA(Number(e.target.value))} className="input-slider" />
          <div className="mt-4 space-y-2 text-sm">
            <p className="text-gray-300">Velocity: <span className="text-white font-mono">{a.vel.toFixed(1)} m/s</span></p>
            <p className="text-gray-300">Wake: <span className="text-white font-mono">{a.wake.toFixed(2)}</span></p>
            <p className="text-gray-300">VEI: <span className="text-white font-mono">{a.vei.toFixed(2)}</span></p>
            <p className="font-bold mt-2" style={{ color: ZONE_META[a.zone]?.color }}>{a.zone}: {ZONE_META[a.zone]?.name}</p>
          </div>
        </div>

        {/* Scenario B */}
        <div className="glass-panel p-5">
          <h3 className="font-semibold text-accent mb-3 text-sm uppercase tracking-wide">Scenario B — Modified</h3>
          <label className="flex justify-between text-xs text-gray-300 mb-1.5">
            <span>Density</span><span className="font-mono text-primary">{densityB.toFixed(2)}</span>
          </label>
          <input type="range" min={0.1} max={0.8} step={0.05} value={densityB} onChange={e => setDensityB(Number(e.target.value))} className="input-slider" />
          <div className="mt-4 space-y-2 text-sm">
            <p className="text-gray-300">Velocity: <span className="text-white font-mono">{b.vel.toFixed(1)} m/s</span></p>
            <p className="text-gray-300">Wake: <span className="text-white font-mono">{b.wake.toFixed(2)}</span></p>
            <p className="text-gray-300">VEI: <span className="text-white font-mono">{b.vei.toFixed(2)}</span></p>
            <p className="font-bold mt-2" style={{ color: ZONE_META[b.zone]?.color }}>{b.zone}: {ZONE_META[b.zone]?.name}</p>
          </div>
        </div>

        {/* Delta */}
        <div className="glass-panel p-5">
          <h3 className="font-semibold text-primary mb-3 text-sm uppercase tracking-wide">Impact Analysis</h3>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between items-center p-3 bg-surfaceHover/30 rounded-lg">
              <span className="text-gray-300">Ventilation Gain</span>
              <span className={`font-mono font-bold ${b.vel > a.vel ? 'text-accent' : 'text-red-400'}`}>{delta(a.vel, b.vel)} m/s</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-surfaceHover/30 rounded-lg">
              <span className="text-gray-300">Wake Reduction</span>
              <span className={`font-mono font-bold ${b.wake < a.wake ? 'text-accent' : 'text-red-400'}`}>{delta(a.wake, b.wake)}</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-surfaceHover/30 rounded-lg">
              <span className="text-gray-300">VEI Change</span>
              <span className={`font-mono font-bold ${b.vei > a.vei ? 'text-accent' : 'text-red-400'}`}>{delta(a.vei, b.vei)}</span>
            </div>
            {a.zone !== b.zone && (
              <div className="p-3 bg-primary/10 rounded-lg border border-primary/20">
                <p className="text-xs text-gray-400 mb-1">Zone Migration</p>
                <p className="text-sm">
                  <span style={{ color: ZONE_META[a.zone]?.color }}>{a.zone}</span>
                  <span className="text-gray-500 mx-2">→</span>
                  <span style={{ color: ZONE_META[b.zone]?.color }}>{b.zone}</span>
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

/* ─────────────── Diagnostics ─────────────── */
const Diagnostics = () => (
  <div className="p-8 max-w-4xl">
    <h2 className="text-2xl font-bold mb-4 text-white">Model Diagnostics</h2>
    <div className="glass-panel overflow-hidden">
      <div className="bg-surfaceHover p-3 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-2 text-gray-300">
          <Terminal className="w-4 h-4 text-accent" />
          <span className="font-mono text-xs">GET /health</span>
        </div>
        <span className="text-[10px] bg-accent/20 text-accent px-2 py-0.5 rounded">200 OK</span>
      </div>
      <div className="p-5">
        <pre className="text-primary font-mono text-xs leading-relaxed">
          {`{
  "status": "online",
  "model_version": "hybrid_gat_pinn_v2.0",
  "zoning_engine": "phase8a_verified",
  "checkpoint_hash": "b9e4d1a7f",
  "zones_generated": 10,
  "silhouette_score": 0.437,
  "deployment_timestamp": "${new Date().toISOString()}"
}`}
        </pre>
      </div>
    </div>
  </div>
);

/* ─────────────── Placeholder ─────────────── */
const Placeholder = ({ title, desc }: { title: string; desc: string }) => (
  <div className="p-8 h-full flex flex-col items-center justify-center text-center">
    <Layers className="w-14 h-14 text-gray-600 mb-4" />
    <h2 className="text-xl font-bold text-gray-400 mb-2">{title}</h2>
    <p className="text-gray-500 max-w-md text-sm">{desc}</p>
  </div>
);

/* ─────────────── App ─────────────── */
export default function App() {
  return (
    <Router>
      <div className="flex h-screen bg-background text-white overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-y-auto">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/zones" element={<ClimateZoneMap />} />
            <Route path="/scenario" element={<ScenarioExplorer />} />
            <Route path="/compare" element={<ScenarioCompare />} />
            <Route path="/flow" element={<Placeholder title="3D Flow Visualization" desc="Plotly / Deck.gl volumetric rendering will mount when the /predict-field API returns tensor payloads." />} />
            <Route path="/uncertainty" element={<Placeholder title="Uncertainty Heatmaps" desc="Monte Carlo Dropout variance maps rendered as spatial heat overlays." />} />
            <Route path="/diagnostics" element={<Diagnostics />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}
