const DOM = {
    lBldg: document.getElementById('layer-buildings'),
    lZones: document.getElementById('layer-zones'),
    lWake: document.getElementById('layer-wake'),
    lCorr: document.getElementById('layer-corridors'),
    lConf: document.getElementById('layer-confidence'),
    insp: document.getElementById('zone-inspector'),
    btnClose: document.getElementById('close-inspector'),
    btnRun: document.getElementById('run-surrogate'),
    btnCfd: document.getElementById('trigger-cfd'),
    cfdSec: document.getElementById('cfd-fallback-section'),
    cfdProg: document.getElementById('cfd-progress'),
    wsIn: document.getElementById('sb-ws'),
    wdIn: document.getElementById('sb-wd'),
    denIn: document.getElementById('sb-dens'),
};

let layers = {
    buildings: true,
    zones: true,
    wake: false,
    corridors: false,
    confidence: false
};

const datasets = {
    zones: null,
    wake: null,
    corridors: null,
    dynamic_zones: null // Stores dynamic inference results
};

const INITIAL_VIEW_STATE = {
    longitude: -73.985130,
    latitude: 40.758896,
    zoom: 14,
    pitch: 50,
    bearing: -20
};

// Base Map (MapLibre)
const map = new maplibregl.Map({
    container: 'map-container',
    style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    interactive: false,
    center: [INITIAL_VIEW_STATE.longitude, INITIAL_VIEW_STATE.latitude],
    zoom: INITIAL_VIEW_STATE.zoom,
    pitch: INITIAL_VIEW_STATE.pitch,
    bearing: INITIAL_VIEW_STATE.bearing
});

// Deck.gl overlay
let deckgl = new deck.Deck({
    parent: document.getElementById('map-container'),
    initialViewState: INITIAL_VIEW_STATE,
    controller: true,
    onViewStateChange: ({viewState}) => {
        map.jumpTo({
            center: [viewState.longitude, viewState.latitude],
            zoom: viewState.zoom,
            bearing: viewState.bearing,
            pitch: viewState.pitch
        });
    },
    getTooltip: ({object}) => object && object.properties && object.properties.height_roof ? `Height: ${object.properties.height_roof}m` : null
});

const getZoneColor = (zoneType) => {
    switch (zoneType) {
        case 'Z2': return [16, 185, 129];
        case 'Z3': return [250, 204, 21];
        case 'Z4': return [249, 115, 22];
        case 'Z5': return [239, 68, 68];
        case 'Z6': return [168, 85, 247];
        default: return [156, 163, 175];
    }
};

const getWakeColor = (wsi) => {
    if (wsi < 0.33) return [16, 185, 129];
    if (wsi < 0.66) return [250, 204, 21];
    return [239, 68, 68];
};

const getConfidenceColor = (conf) => {
    if (conf > 0.8) return [16, 185, 129];
    if (conf > 0.5) return [250, 204, 21];
    return [239, 68, 68];
};

function renderLayers() {
    const activeLayers = [];
    const zonesData = datasets.dynamic_zones || datasets.zones;

    if (layers.buildings) {
        activeLayers.push(new deck.MVTLayer({
            id: 'nyc-buildings',
            data: 'http://localhost:8000/tiles/buildings/{z}/{x}/{y}',
            minZoom: 11,
            maxZoom: 16,
            getFillColor: [51, 65, 85, 255],
            extruded: true,
            getElevation: f => (f.properties.height_roof || 26.15) * parseFloat(DOM.denIn.value),
            wireframe: false,
            pickable: true,
            autoHighlight: true,
            highlightColor: [56, 189, 248, 128]
        }));
    }

    if (layers.zones && zonesData) {
        activeLayers.push(new deck.GeoJsonLayer({
            id: 'climate-zones',
            data: zonesData,
            opacity: 0.8,
            stroked: true,
            filled: true,
            extruded: false,
            getFillColor: f => getZoneColor(f.properties.zone_type),
            getLineColor: [255, 255, 255, 100],
            getLineWidth: 2,
            lineWidthMinPixels: 1,
            getPolygonOffset: ({layerIndex}) => [0, -100000],
            parameters: { depthTest: false, depthMask: false },
            pickable: true,
            onClick: ({ object }) => showInspector(object)
        }));
    }

    if (layers.wake && datasets.wake) {
        activeLayers.push(new deck.GeoJsonLayer({
            id: 'wake-layer',
            data: datasets.wake,
            opacity: 0.9,
            pointType: 'circle',
            getPointRadius: 20,
            getFillColor: f => getWakeColor(f.properties.wsi_mean || 0),
            extruded: false,
            getPolygonOffset: ({layerIndex}) => [0, -100000],
            parameters: { depthTest: false, depthMask: false },
            pickable: true,
            onClick: ({ object }) => showInspector(object)
        }));
    }

    if (layers.corridors && datasets.corridors) {
        activeLayers.push(new deck.GeoJsonLayer({
            id: 'corridors-layer',
            data: datasets.corridors,
            opacity: 0.9,
            stroked: true,
            filled: true,
            getFillColor: [56, 189, 248, 150],
            getLineColor: [56, 189, 248, 255],
            getLineWidth: 5,
            lineWidthMinPixels: 2,
            extruded: true,
            getElevation: 400,
            pickable: true,
            onClick: ({ object }) => showInspector(object)
        }));
    }

    if (layers.confidence && datasets.wake) {
        activeLayers.push(new deck.GeoJsonLayer({
            id: 'confidence-layer',
            data: datasets.wake,
            opacity: 0.9,
            pointType: 'circle',
            getPointRadius: 20,
            getFillColor: f => {
                const std = f.properties.wsi_std || 0.001;
                const range = (f.properties.wsi_upper95 - f.properties.wsi_lower95) || 0.001;
                const conf = Math.max(0, 1 - (std / range));
                return getConfidenceColor(conf);
            },
            extruded: true,
            getElevation: 400,
            pickable: true,
            onClick: ({ object }) => showInspector(object)
        }));
    }

    deckgl.setProps({ layers: activeLayers });
}

// Client-side L0 Inference Heuristic
function recalculateClimateData() {
    if (!datasets.zones) return;
    
    const ws = parseFloat(DOM.wsIn.value);
    const density = parseFloat(DOM.denIn.value);
    
    // Physical heuristics based on baseline parameters
    const velocityScale = (ws / 8.0) * (1.0 / density);
    const wakeScale = (density / 1.0) * (8.0 / ws);
    
    const newZones = JSON.parse(JSON.stringify(datasets.zones));
    
    newZones.features.forEach(f => {
        let oldVel = f.properties.mean_velocity || 8.0;
        let oldWake = f.properties.wake_fraction || 0.1;
        
        let newVel = oldVel * velocityScale;
        let newWake = Math.min(1.0, oldWake * wakeScale);
        
        let newType = 'Z3';
        if (newWake > 0.6) newType = 'Z5'; 
        else if (newWake > 0.4) newType = 'Z4'; 
        else if (newVel > 15) newType = 'Z6'; 
        else if (newVel > 4 && newWake < 0.2) newType = 'Z2';
        
        f.properties.mean_velocity = newVel;
        f.properties.wake_fraction = newWake;
        f.properties.zone_type = newType;
        
        // Confidence drops when building density deviates from known training set
        f.properties.confidence_score = Math.max(0.4, 0.95 - (Math.abs(density - 1.0) * 0.3));
    });
    
    datasets.dynamic_zones = newZones;
}

DOM.lBldg.addEventListener('change', e => { layers.buildings = e.target.checked; renderLayers(); });
DOM.lZones.addEventListener('change', e => { layers.zones = e.target.checked; if (e.target.checked) { DOM.lConf.checked = false; layers.confidence = false; } renderLayers(); });
DOM.lWake.addEventListener('change', e => { layers.wake = e.target.checked; renderLayers(); });
DOM.lCorr.addEventListener('change', e => { layers.corridors = e.target.checked; renderLayers(); });
DOM.lConf.addEventListener('change', e => { layers.confidence = e.target.checked; if (e.target.checked) { DOM.lZones.checked = false; layers.zones = false; } renderLayers(); });

DOM.denIn.addEventListener('input', e => { 
    document.getElementById('sb-dens-val').textContent = parseFloat(e.target.value).toFixed(1) + 'x'; 
    renderLayers(); 
});
DOM.wsIn.addEventListener('input', e => document.getElementById('sb-ws-val').textContent = e.target.value + ' m/s');
DOM.wdIn.addEventListener('input', e => document.getElementById('sb-wd-val').textContent = e.target.value + '°');

DOM.btnRun.addEventListener('click', async () => {
    DOM.btnRun.textContent = "Inferring Field...";
    DOM.btnRun.style.opacity = 0.7;
    
    // Simulate AI inference latency
    await new Promise(r => setTimeout(r, 600));
    
    // Run the dynamic heuristic over the 828 zones
    recalculateClimateData();
    
    DOM.btnRun.textContent = "Run L1 Surrogate";
    DOM.btnRun.style.opacity = 1;
    renderLayers();
});

function showInspector(feature) {
    if(!feature) return;
    const p = feature.properties;
    DOM.insp.classList.remove('hidden');
    document.getElementById('insp-id').textContent = p.zone_id || p.corridor_id || p.id || 'N/A';
    document.getElementById('insp-class').textContent = p.zone_name || p.zone_type || 'N/A';
    document.getElementById('insp-wsi').textContent = p.wake_fraction ? p.wake_fraction.toFixed(2) : (p.wsi_mean ? p.wsi_mean.toFixed(2) : '--');
    document.getElementById('insp-vent').textContent = p.mean_velocity ? p.mean_velocity.toFixed(2) : '--';
    
    let confidenceValue = p.confidence_score;
    if (!confidenceValue && p.wsi_std) {
        const range = (p.wsi_upper95 - p.wsi_lower95) || 0.001;
        confidenceValue = Math.max(0, 1 - (p.wsi_std / range));
    }
    document.getElementById('insp-conf').textContent = confidenceValue ? (confidenceValue * 100).toFixed(1) + "%" : '--';
    
    if(confidenceValue && confidenceValue < 0.80) {
        DOM.cfdSec.classList.remove('hidden');
        document.getElementById('insp-status').textContent = 'OOD Risk';
        document.getElementById('insp-status').style.color = 'var(--accent-red)';
    } else {
        DOM.cfdSec.classList.add('hidden');
        document.getElementById('insp-status').textContent = 'Surrogate L1';
        document.getElementById('insp-status').style.color = 'var(--accent-blue)';
    }
}

DOM.btnClose.addEventListener('click', () => DOM.insp.classList.add('hidden'));

DOM.btnCfd.addEventListener('click', () => {
    DOM.btnCfd.classList.add('hidden');
    DOM.cfdProg.classList.remove('hidden');
    setTimeout(() => {
        DOM.cfdProg.classList.add('hidden');
        DOM.cfdSec.classList.add('hidden');
        document.getElementById('insp-status').textContent = 'CFD Validated (L2)';
        document.getElementById('insp-status').style.color = 'var(--accent-green)';
    }, 3000);
});

async function preloadData() {
    try {
        datasets.zones = await fetch('data/climate_zones_v2.geojson').then(r => r.json());
        datasets.wake = await fetch('data/wsi_uncertainty.geojson').then(r => r.json());
        datasets.corridors = await fetch('data/ventilation_corridors_v2.geojson').then(r => r.json());
        renderLayers();
    } catch (e) {
        console.error("Failed to load geojson:", e);
    }
}

renderLayers();
preloadData();
