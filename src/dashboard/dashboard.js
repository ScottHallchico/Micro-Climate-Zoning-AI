/**
 * Micro-Climate Zoning AI — Scenario Dashboard JavaScript
 * Interactive city block map with PINN predictions and Pareto front exploration.
 */

// ============================================================
// Synthetic Data Generation (would be replaced by API calls)
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

function generateParetoConfigs(count = 12) {
    const configs = [];
    for (let i = 0; i < count; i++) {
        configs.push({
            id: `CFG-${String(i + 1).padStart(3, '0')}`,
            uhi_intensity: (2 + Math.random() * 8).toFixed(2),
            pet_index: (22 + Math.random() * 15).toFixed(1),
            solar_access: (3 + Math.random() * 5).toFixed(1),
            retrofit_cost: Math.round(50000 + Math.random() * 500000),
        });
    }
    configs.sort((a, b) => parseFloat(a.uhi_intensity) - parseFloat(b.uhi_intensity));
    return configs;
}

function generateWards() {
    return [
        { id: 'W-01', name: 'Central District', cooling: -2.1, cost: 850000, risk: 0.72, flagged: true },
        { id: 'W-02', name: 'Riverside Ward', cooling: -1.4, cost: 420000, risk: 0.31, flagged: false },
        { id: 'W-03', name: 'Industrial Zone', cooling: -3.2, cost: 1200000, risk: 0.15, flagged: false },
        { id: 'W-04', name: 'Old Town', cooling: -0.8, cost: 680000, risk: 0.65, flagged: true },
    ];
}

// ============================================================
// State
// ============================================================

let state = {
    blocks: generateBlocks(),
    paretoConfigs: generateParetoConfigs(),
    wards: generateWards(),
    selectedBlock: null,
    selectedConfig: null,
    overrideDeltaT: null,
};

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

        // Block body with UHI intensity fill
        svg += `<rect class="map-block ${isSelected ? 'selected' : ''}" 
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

    // Redistribute UHI values based on selected config
    const baseUhi = parseFloat(config.uhi_intensity);
    state.blocks.forEach(block => {
        block.uhi_intensity = baseUhi / state.blocks.length + Math.random() * 2;
        block.temperature = 300 + block.uhi_intensity;
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

    // Check wind corridor violation
    if (block.zone_class === 'WIND_CORRIDOR_CRITICAL' && newHeight > block.max_height_m) {
        showWarning(
            `⚠️ Wind corridor violation! Height ${newHeight}m exceeds cap of ${block.max_height_m}m. ` +
            `Predicted deflection: ${(15 + Math.random() * 30).toFixed(1)}°. ` +
            `Downstream blocks BLK-05–BLK-11 would lose ${(0.8 + Math.random() * 1.5).toFixed(1)}°C cooling benefit.`
        );
    } else {
        hideWarning();
    }

    // Simulate PINN prediction
    const deltaT = simulatePINNPrediction(block, newHeight, newAlbedo, newGreen);
    state.overrideDeltaT = deltaT;

    // Update block state
    block.albedo = newAlbedo.toFixed(2);
    block.green_cover = newGreen;
    block.uhi_intensity = Math.max(0, block.uhi_intensity + deltaT);

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
            <span class="detail-value">${(0.1 + Math.random() * 0.2).toFixed(3)}s</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Affected Blocks</span>
            <span class="detail-value">${2 + Math.floor(Math.random() * 6)}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Confidence</span>
            <span class="detail-value">${(92 + Math.random() * 7).toFixed(1)}%</span>
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
        state.blocks = generateBlocks();
        state.paretoConfigs = generateParetoConfigs();
        state.wards = generateWards();
        state.selectedBlock = null;
        state.selectedConfig = null;
        renderAll();
    });
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

document.addEventListener('DOMContentLoaded', () => {
    populateBlockSelect();
    setupEventListeners();
    renderAll();
});
