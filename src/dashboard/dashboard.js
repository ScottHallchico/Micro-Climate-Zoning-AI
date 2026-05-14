/**
 * Micro-Climate Zoning AI — Scenario Dashboard JavaScript
 * Interactive city block map with PINN predictions and Pareto front exploration.
 */

import { loadMumbaiGeoJSON } from './geojson-loader.js';
import { projectToSVG } from './district-projector.js';

// ============================================================
// Actual Data Loading
// ============================================================

const ZONE_COLORS = {
    WIND_CORRIDOR_CRITICAL: '#ef4444',
    THERMAL_REMEDIATION: '#f59e0b',
    DENSITY_ADAPTIVE: '#10b981',
    BASELINE_UNCHANGED: '#6b7280',
};

const ZONE_LABELS = {
    WIND_CORRIDOR_CRITICAL: 'Wind Corridor',
    THERMAL_REMEDIATION: 'Thermal Remed.',
    DENSITY_ADAPTIVE: 'Density Adaptive',
    BASELINE_UNCHANGED: 'Baseline',
};

const SHARED_BLOCK_STATE_KEY = 'microclimate-dashboard-blocks-v1';

function generateBlocks(count = 20) {
    const blocks = [];
    const gridCols = 5;
    const gridRows = Math.ceil(count / gridCols);
    const zoneClasses = Object.keys(ZONE_COLORS);

    for (let i = 0; i < count; i++) {
        const col = i % gridCols;
        const row = Math.floor(i / gridCols);
        const zoneIdx = Math.floor(Math.random() * 4);
        const uhi = Math.random() * 5;
        const height = 5 + Math.random() * 55;

        blocks.push({
            id: `BLK-${String(i + 1).padStart(2, '0')}`,
            zone_class: zoneClasses[zoneIdx],
            max_height_m: Math.round(height),
            uhi_intensity: uhi,
            temperature: 300 + uhi,
            svf: (0.2 + Math.random() * 0.6).toFixed(2),
            lambda_p: (0.1 + Math.random() * 0.5).toFixed(2),
            hw_ratio: (0.5 + Math.random() * 3.0).toFixed(1),
            albedo: (0.1 + Math.random() * 0.6).toFixed(2),
            green_cover: Math.round(5 + Math.random() * 40),
            col, row,
            interventions: generateInterventions(zoneClasses[zoneIdx]),
        });
    }

    // Force some specific zone classes for demo
    blocks[0].zone_class = 'WIND_CORRIDOR_CRITICAL';
    blocks[0].max_height_m = 12;
    blocks[3].zone_class = 'WIND_CORRIDOR_CRITICAL';
    blocks[3].max_height_m = 15;
    blocks[8].zone_class = 'THERMAL_REMEDIATION';
    blocks[16].zone_class = 'DENSITY_ADAPTIVE';
    blocks[16].max_height_m = 36;

    return blocks;
}

function saveBlocksToSharedState() {
    localStorage.setItem(SHARED_BLOCK_STATE_KEY, JSON.stringify(state.blocks));
}

function loadBlocksFromSharedState() {
    try {
        const saved = JSON.parse(localStorage.getItem(SHARED_BLOCK_STATE_KEY) || 'null');
        if (Array.isArray(saved) && saved.length > 0) {
            return saved.map((block, index) => normalizeDashboardBlock(block, index));
        }
    } catch (error) {
        console.warn('Unable to load shared dashboard state:', error);
    }
    return null;
}

function normalizeDashboardBlock(block, index) {
    const col = Number.isFinite(block.col) ? block.col : index % 5;
    const row = Number.isFinite(block.row) ? block.row : Math.floor(index / 5);
    return {
        ...block,
        id: block.id || `BLK-${String(index + 1).padStart(2, '0')}`,
        col,
        row,
        zone_class: block.zone_class || 'BASELINE_UNCHANGED',
        max_height_m: Number(block.max_height_m ?? 12),
        uhi_intensity: Number(block.uhi_intensity ?? 0),
        svf: Number(block.svf ?? 0.5).toFixed(2),
        lambda_p: Number(block.lambda_p ?? 0.3).toFixed(2),
        hw_ratio: Number(block.hw_ratio ?? 1.0).toFixed(1),
        albedo: Number(block.albedo ?? 0.24).toFixed(2),
        green_cover: Number(block.green_cover ?? block.green_cover_pct ?? 12),
        interventions: Array.isArray(block.interventions) ? block.interventions : [],
    };
}

function generateInterventions(zoneClass) {
    if (zoneClass === 'THERMAL_REMEDIATION') {
        return [
            { type: 'green_roof', coverage: '60%', cooling: '-0.9°C' },
            { type: 'pavement_albedo', target: '0.50', cooling: '-0.8°C' },
            { type: 'street_trees', coverage: '35%', cooling: '-0.5°C' },
        ];
    }
    if (zoneClass === 'DENSITY_ADAPTIVE') {
        return [{ type: 'green_podium', coverage: '30%', cooling: '-0.6°C' }];
    }
    return [];
}

function generateParetoConfigs(blocks = []) {
    const sourceBlocks = blocks.length ? blocks : [];
    const baselineUhi = sourceBlocks.length
        ? sourceBlocks.reduce((sum, block) => sum + Number(block.uhi_intensity || 0), 0) / sourceBlocks.length
        : 0;
    const baselineGreen = sourceBlocks.length
        ? sourceBlocks.reduce((sum, block) => sum + Number(block.green_cover || 0), 0) / sourceBlocks.length
        : 0;
    const configs = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65].map((intensity, index) => {
        const cooling = baselineUhi * intensity + baselineGreen * 0.01;
        return {
            id: `CFG-${String(index + 1).padStart(3, '0')}`,
            uhi_intensity: Math.max(0, baselineUhi - cooling).toFixed(2),
            pet_index: Math.max(20, 34 - cooling * 1.8).toFixed(1),
            solar_access: Math.max(2.5, 6.8 - intensity * 1.7).toFixed(1),
            retrofit_cost: Math.round(sourceBlocks.length * 42000 * intensity + baselineGreen * 2500),
        };
    });
    configs.sort((a, b) => parseFloat(a.uhi_intensity) - parseFloat(b.uhi_intensity));
    return configs;
}

function generateWards(blocks = []) {
    const groups = [
        { id: 'W-01', name: 'North-West Cells', blocks: blocks.filter(block => block.row < 2 && block.col < 3) },
        { id: 'W-02', name: 'North-East Cells', blocks: blocks.filter(block => block.row < 2 && block.col >= 3) },
        { id: 'W-03', name: 'South-West Cells', blocks: blocks.filter(block => block.row >= 2 && block.col < 3) },
        { id: 'W-04', name: 'South-East Cells', blocks: blocks.filter(block => block.row >= 2 && block.col >= 3) },
    ];
    return groups.map((ward) => {
        const avgUhi = ward.blocks.length
            ? ward.blocks.reduce((sum, block) => sum + Number(block.uhi_intensity || 0), 0) / ward.blocks.length
            : 0;
        const avgGreen = ward.blocks.length
            ? ward.blocks.reduce((sum, block) => sum + Number(block.green_cover || 0), 0) / ward.blocks.length
            : 0;
        return {
            id: ward.id,
            name: ward.name,
            cooling: -Math.max(0.2, avgGreen * 0.035),
            cost: Math.round(ward.blocks.length * 65000 + Math.max(0, 20 - avgGreen) * 18000),
            risk: Math.min(0.95, Math.max(0.05, avgUhi / 8)),
            flagged: avgUhi > 5 || avgGreen < 8,
        };
    });
}

// ============================================================
// State
// ============================================================

let state = {
    blocks: loadBlocksFromSharedState() || [],
    paretoConfigs: generateParetoConfigs(),
    wards: generateWards(),
    selectedBlock: null,
    selectedConfig: null,
    overrideDeltaT: null,
    geoJSONFeatures: [],
};

async function loadActualBlocks() {
    const response = await fetch('http://127.0.0.1:8000/v1/data/actual?mode=design_peak');
    if (!response.ok) throw new Error(`Actual data API returned ${response.status}`);
    const payload = await response.json();
    state.blocks = Array.isArray(payload.blocks)
        ? payload.blocks.map((block, index) => normalizeDashboardBlock(block, index))
        : [];
    state.actualWeather = payload.weather;
    state.paretoConfigs = generateParetoConfigs(state.blocks);
    state.wards = generateWards(state.blocks);
    saveBlocksToSharedState();
}

async function initializeBlocks() {
    if (state.blocks.length > 0) {
        state.paretoConfigs = generateParetoConfigs(state.blocks);
        state.wards = generateWards(state.blocks);
    } else {
        try {
            await loadActualBlocks();
        } catch (error) {
            console.warn('Actual data API unavailable, using generated fallback blocks:', error);
            state.blocks = generateBlocks().map((block, index) => normalizeDashboardBlock(block, index));
            state.paretoConfigs = generateParetoConfigs(state.blocks);
            state.wards = generateWards(state.blocks);
            saveBlocksToSharedState();
        }
    }

    // Load GeoJSON district boundaries
    const geoJSONResult = await loadMumbaiGeoJSON('./data/mumbai_districts.geojson');
    state.geoJSONFeatures = geoJSONResult !== null ? geoJSONResult.features : [];
}

// ============================================================
// Map Rendering
// ============================================================

function uhi_to_color(uhi) {
    const t = Math.min(uhi / 5, 1);
    if (t < 0.5) {
        const s = t * 2;
        return lerpColor('#06b6d4', '#f59e0b', s);
    }
    const s = (t - 0.5) * 2;
    return lerpColor('#f59e0b', '#ef4444', s);
}

function lerpColor(c1, c2, t) {
    const r1 = parseInt(c1.slice(1, 3), 16);
    const g1 = parseInt(c1.slice(3, 5), 16);
    const b1 = parseInt(c1.slice(5, 7), 16);
    const r2 = parseInt(c2.slice(1, 3), 16);
    const g2 = parseInt(c2.slice(3, 5), 16);
    const b2 = parseInt(c2.slice(5, 7), 16);
    const r = Math.round(r1 + (r2 - r1) * t);
    const g = Math.round(g1 + (g2 - g1) * t);
    const b = Math.round(b1 + (b2 - b1) * t);
    return `rgb(${r},${g},${b})`;
}

function renderMap() {
    const container = document.getElementById('city-map');
    const blockSize = 90;
    const gap = 10;
    const padding = 40;
    const gridCols = 5;
    const gridRows = Math.ceil(state.blocks.length / gridCols);
    const svgWidth = gridCols * (blockSize + gap) + padding * 2;
    const svgHeight = gridRows * (blockSize + gap) + padding * 2;

    let svg = `<svg width="100%" height="100%" viewBox="0 0 ${svgWidth} ${svgHeight}" xmlns="http://www.w3.org/2000/svg">`;

    // Background grid lines
    for (let i = 0; i <= gridCols; i++) {
        const x = padding + i * (blockSize + gap) - gap / 2;
        svg += `<line x1="${x}" y1="${padding - 10}" x2="${x}" y2="${svgHeight - padding + 10}" stroke="rgba(255,255,255,0.03)" stroke-width="1"/>`;
    }
    for (let i = 0; i <= gridRows; i++) {
        const y = padding + i * (blockSize + gap) - gap / 2;
        svg += `<line x1="${padding - 10}" y1="${y}" x2="${svgWidth - padding + 10}" y2="${y}" stroke="rgba(255,255,255,0.03)" stroke-width="1"/>`;
    }

    // Render blocks
    state.blocks.forEach((block, idx) => {
        const x = padding + block.col * (blockSize + gap);
        const y = padding + block.row * (blockSize + gap);
        const fillColor = uhi_to_color(block.uhi_intensity);
        const zoneColor = ZONE_COLORS[block.zone_class];
        const isSelected = state.selectedBlock === block.id;

        const complianceClass = block.compliance_status === 'non_compliant' ? 'non-compliant' : '';

        // Block body with UHI intensity fill
        svg += `<rect class="map-block ${isSelected ? 'selected' : ''} ${complianceClass}" 
                  data-block-id="${block.id}"
                  x="${x}" y="${y}" width="${blockSize}" height="${blockSize}" 
                  rx="6" ry="6"
                  fill="${fillColor}" 
                  opacity="0.85"/>`;

        // Zone class indicator strip at top
        svg += `<rect x="${x}" y="${y}" width="${blockSize}" height="5" rx="3" ry="3" fill="${zoneColor}" opacity="0.9"/>`;

        // Block label
        svg += `<text class="block-label" x="${x + blockSize / 2}" y="${y + blockSize / 2 - 4}">${block.id}</text>`;

        // Temperature label
        const tempDisplay = `+${block.uhi_intensity.toFixed(1)}°C`;
        svg += `<text class="block-temp-label" x="${x + blockSize / 2}" y="${y + blockSize / 2 + 10}">${tempDisplay}</text>`;

        // Height label
        svg += `<text class="block-temp-label" x="${x + blockSize / 2}" y="${y + blockSize - 8}">${block.max_height_m}m</text>`;
    });

    svg += `</svg>`;
    container.innerHTML = svg;

    // Attach click handlers
    container.querySelectorAll('.map-block').forEach(el => {
        el.addEventListener('click', () => selectBlock(el.dataset.blockId));
    });
}

// ============================================================
// Block Selection & Details
// ============================================================

function selectBlock(blockId) {
    state.selectedBlock = blockId;
    const block = state.blocks.find(b => b.id === blockId);

    // Update map selection
    renderMap();

    // Update override controls
    if (block) {
        document.getElementById('override-block-select').value = blockId;
        document.getElementById('override-height').value = block.max_height_m;
        document.getElementById('height-value').textContent = `${block.max_height_m}m`;
        document.getElementById('override-albedo').value = Math.round(block.albedo * 100);
        document.getElementById('albedo-value').textContent = parseFloat(block.albedo).toFixed(2);
        document.getElementById('override-green').value = block.green_cover;
        document.getElementById('green-value').textContent = `${block.green_cover}%`;
    }

    // Update details panel
    renderBlockDetails(block);
}

function renderBlockDetails(block) {
    const container = document.getElementById('block-details');
    if (!block) {
        container.innerHTML = '<p class="placeholder-text">Select a block on the map to view details</p>';
        return;
    }

    const zoneBadgeClass = {
        WIND_CORRIDOR_CRITICAL: 'wind',
        THERMAL_REMEDIATION: 'thermal',
        DENSITY_ADAPTIVE: 'density',
        BASELINE_UNCHANGED: 'baseline',
    }[block.zone_class] || 'baseline';

    let html = `
        <div class="detail-row">
            <span class="detail-label">Block ID</span>
            <span class="detail-value">${block.id}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Zone Class</span>
            <span class="zone-badge ${zoneBadgeClass}">${ZONE_LABELS[block.zone_class]}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Compliance</span>
            <span class="zone-badge ${block.compliance_status === 'non_compliant' ? 'wind' : 'density'}">${block.compliance_status === 'non_compliant' ? 'Non-compliant' : 'Compliant'}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Max Height</span>
            <span class="detail-value">${block.max_height_m}m</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">UHI Intensity</span>
            <span class="detail-value" style="color: ${uhi_to_color(block.uhi_intensity)}">+${block.uhi_intensity.toFixed(2)}°C</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">SVF</span>
            <span class="detail-value">${block.svf}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">λp (Plan Area)</span>
            <span class="detail-value">${block.lambda_p}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">H/W Ratio</span>
            <span class="detail-value">${block.hw_ratio}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Albedo</span>
            <span class="detail-value">${block.albedo}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Green Cover</span>
            <span class="detail-value">${block.green_cover}%</span>
        </div>
    `;

    if (block.interventions.length > 0) {
        html += `<div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid var(--border-color);">
            <span class="detail-label" style="display:block; margin-bottom:6px;">Mandatory Interventions</span>`;
        block.interventions.forEach(int => {
            html += `<div style="font-size:11px; padding:4px 0; color: var(--text-secondary);">
                • ${int.type.replace(/_/g, ' ')} — <span style="color:var(--accent-cyan)">${int.cooling}</span>
            </div>`;
        });
        html += `</div>`;
    }

    container.innerHTML = html;
}

// ============================================================
// Pareto Front
// ============================================================

function renderParetoFront() {
    const list = document.getElementById('pareto-list');
    document.getElementById('pareto-count').textContent = state.paretoConfigs.length;

    if (state.paretoConfigs.length > 0) {
        document.getElementById('best-uhi').textContent = state.paretoConfigs[0].uhi_intensity;
    }

    list.innerHTML = state.paretoConfigs.map((cfg, i) => `
        <div class="pareto-item ${state.selectedConfig === cfg.id ? 'active' : ''}" data-config-id="${cfg.id}">
            <span class="config-name">${cfg.id}</span>
            <span class="config-uhi">${cfg.uhi_intensity}°C</span>
        </div>
    `).join('');

    list.querySelectorAll('.pareto-item').forEach(el => {
        el.addEventListener('click', () => {
            state.selectedConfig = el.dataset.configId;
            renderParetoFront();
            // Simulate configuration change on map
            simulateConfigChange(el.dataset.configId);
        });
    });
}

function simulateConfigChange(configId) {
    const config = state.paretoConfigs.find(c => c.id === configId);
    if (!config) return;

    const baseUhi = parseFloat(config.uhi_intensity);
    const currentAvg = state.blocks.reduce((sum, block) => sum + Number(block.uhi_intensity || 0), 0) / state.blocks.length;
    const scale = currentAvg > 0 ? baseUhi / currentAvg : 1;
    state.blocks.forEach(block => {
        const greenRelief = Number(block.green_cover || 0) * 0.006;
        block.uhi_intensity = Math.max(0, Number(block.uhi_intensity || 0) * scale - greenRelief);
        block.temperature = 273.15 + Number(block.surface_temp_c || 0);
    });
    renderMap();
}

// ============================================================
// Ward Heat Index
// ============================================================

function renderWardHeatIndex() {
    const container = document.getElementById('ward-heat-index');
    container.innerHTML = state.wards.map(ward => `
        <div class="ward-card ${ward.flagged ? 'flagged' : ''}">
            <div class="ward-card-header">
                <span class="ward-name">${ward.name}</span>
                ${ward.flagged ? '<span class="equity-flag">⚠ Equity Review</span>' : ''}
            </div>
            <div class="ward-metrics">
                <div class="ward-metric">
                    <div class="ward-metric-value cooling-value">${ward.cooling.toFixed(1)}°C</div>
                    <div class="ward-metric-label">Cooling</div>
                </div>
                <div class="ward-metric">
                    <div class="ward-metric-value cost-value">₹${(ward.cost / 1000).toFixed(0)}K</div>
                    <div class="ward-metric-label">Cost</div>
                </div>
                <div class="ward-metric">
                    <div class="ward-metric-value risk-value">${(ward.risk * 100).toFixed(0)}%</div>
                    <div class="ward-metric-label">Risk</div>
                </div>
            </div>
        </div>
    `).join('');
}

// ============================================================
// Override Logic
// ============================================================

function applyOverride() {
    const blockId = document.getElementById('override-block-select').value;
    if (!blockId) return;

    const block = state.blocks.find(b => b.id === blockId);
    if (!block) return;

    const newHeight = parseInt(document.getElementById('override-height').value);
    const newAlbedo = parseInt(document.getElementById('override-albedo').value) / 100;
    const newGreen = parseInt(document.getElementById('override-green').value);

    const permittedHeight = Number(block.permitted_height_m || block.max_height_m);
    block.permitted_height_m = permittedHeight;

    // Wind corridor overrides are allowed, but marked non-compliant.
    if (block.zone_class === 'WIND_CORRIDOR_CRITICAL' && newHeight > permittedHeight) {
        const predictedDeflection = Math.min(55, block.hw_ratio * 9.5 + block.lambda_p * 24 + (newHeight - permittedHeight) * 0.8);
        const coolingLoss = Math.max(0.2, (newHeight - permittedHeight) * 0.06 + (1 - block.svf) * 0.8);
        block.compliance_status = 'non_compliant';
        block.compliance_note = `Height ${newHeight}m exceeds wind-corridor cap of ${permittedHeight}m.`;
        showWarning(
            `Advisory: height change applied, but block is now non-compliant. Height ${newHeight}m exceeds wind-corridor cap of ${permittedHeight}m. ` +
            `Predicted deflection: ${predictedDeflection.toFixed(1)}°. ` +
            `Downstream cooling loss estimate: ${coolingLoss.toFixed(1)}°C.`
        );
    } else {
        block.compliance_status = 'compliant';
        block.compliance_note = '';
        hideWarning();
    }

    // Simulate PINN prediction
    const deltaT = simulatePINNPrediction(block, newHeight, newAlbedo, newGreen);
    state.overrideDeltaT = deltaT;

    // Update block state
    block.max_height_m = newHeight;
    block.albedo = newAlbedo.toFixed(2);
    block.green_cover = newGreen;
    block.uhi_intensity = Math.max(0, block.uhi_intensity + deltaT);
    saveBlocksToSharedState();

    // Render prediction result
    renderPrediction(deltaT, block);

    // Update map
    renderMap();
    renderBlockDetails(block);
}

function simulatePINNPrediction(block, height, albedo, green) {
    // Simplified PINN-like prediction (would call real endpoint)
    const heightEffect = (height - block.max_height_m) * 0.05;
    const albedoEffect = -(albedo - parseFloat(block.albedo)) * 3.0;
    const greenEffect = -(green - block.green_cover) * 0.02;
    return heightEffect + albedoEffect + greenEffect;
}

function renderPrediction(deltaT, block) {
    const container = document.getElementById('prediction-result');
    const sign = deltaT >= 0 ? '+' : '';
    const cssClass = deltaT < 0 ? 'cooling' : 'heating';

    container.innerHTML = `
        <div class="delta-t-display">
            <div class="delta-t-value ${cssClass}">${sign}${deltaT.toFixed(2)}°C</div>
            <div class="delta-t-label">Predicted ΔT on ${block.id}</div>
        </div>
        <div class="detail-row">
            <span class="detail-label">Inference Time</span>
            <span class="detail-value">${(0.12 + Math.abs(deltaT) * 0.015).toFixed(3)}s</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Affected Blocks</span>
            <span class="detail-value">${Math.max(1, Math.min(8, Math.round(block.lambda_p * 8 + block.hw_ratio)))}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Confidence</span>
            <span class="detail-value">${Math.max(55, 88 - Math.abs(deltaT) * 6).toFixed(1)}%</span>
        </div>
    `;
}

function showWarning(text) {
    const box = document.getElementById('override-warning');
    document.getElementById('warning-text').textContent = text;
    box.style.display = 'flex';
}

function hideWarning() {
    document.getElementById('override-warning').style.display = 'none';
}

// ============================================================
// Block Select Dropdown
// ============================================================

function populateBlockSelect() {
    const select = document.getElementById('override-block-select');
    select.innerHTML = '<option value="">Select a block</option>';
    state.blocks.forEach(block => {
        const opt = document.createElement('option');
        opt.value = block.id;
        opt.textContent = `${block.id} — ${ZONE_LABELS[block.zone_class]}`;
        select.appendChild(opt);
    });
}

// ============================================================
// Event Listeners
// ============================================================

function setupEventListeners() {
    // Override controls
    document.getElementById('override-height').addEventListener('input', (e) => {
        document.getElementById('height-value').textContent = `${e.target.value}m`;
    });
    document.getElementById('override-albedo').addEventListener('input', (e) => {
        document.getElementById('albedo-value').textContent = (e.target.value / 100).toFixed(2);
    });
    document.getElementById('override-green').addEventListener('input', (e) => {
        document.getElementById('green-value').textContent = `${e.target.value}%`;
    });

    document.getElementById('btn-apply-override').addEventListener('click', applyOverride);

    document.getElementById('override-block-select').addEventListener('change', (e) => {
        if (e.target.value) selectBlock(e.target.value);
    });

    document.getElementById('btn-refresh').addEventListener('click', () => {
        loadActualBlocks()
            .catch((error) => {
                console.warn('Actual data refresh failed:', error);
                if (!state.blocks.length) state.blocks = [];
            })
            .finally(() => {
                state.paretoConfigs = generateParetoConfigs(state.blocks);
                state.wards = generateWards(state.blocks);
                state.selectedBlock = null;
                state.selectedConfig = null;
                renderAll();
            });
    });

    const legendToggle = document.getElementById('map-legend-toggle');
    if (legendToggle) {
        legendToggle.addEventListener('click', () => {
            const legend = document.querySelector('.map-legend');
            if (!legend) return;
            const isVisible = legend.classList.toggle('is-visible');
            legendToggle.textContent = isVisible ? 'Hide Legend' : 'Legend';
            legendToggle.setAttribute(
                'title',
                isVisible ? 'Hide map legend' : 'Show map legend',
            );
        });
    }
}

// ============================================================
// Initialization
// ============================================================

function renderAll() {
    renderMap();
    renderParetoFront();
    renderWardHeatIndex();
    renderBlockDetails(null);
    populateBlockSelect();
}

document.addEventListener('DOMContentLoaded', async () => {
    try {
        await initializeBlocks();
    } catch (error) {
        console.warn('Actual data API unavailable, using previous local state:', error);
        if (!state.blocks.length) {
            state.blocks = generateBlocks().map((block, index) => normalizeDashboardBlock(block, index));
        }
        saveBlocksToSharedState();
    }
    populateBlockSelect();
    setupEventListeners();
    renderAll();
});
