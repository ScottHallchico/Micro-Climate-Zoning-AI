import * as THREE from "three";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.164.1/examples/jsm/controls/OrbitControls.js";

const ZONE_COLORS = {
    WIND_CORRIDOR_CRITICAL: 0xef4444,
    THERMAL_REMEDIATION: 0xf59e0b,
    DENSITY_ADAPTIVE: 0x10b981,
    BASELINE_UNCHANGED: 0x6b7280,
};

const ZONE_LABELS = {
    WIND_CORRIDOR_CRITICAL: "Wind Corridor Critical",
    THERMAL_REMEDIATION: "Thermal Remediation",
    DENSITY_ADAPTIVE: "Density Adaptive",
    BASELINE_UNCHANGED: "Baseline Unchanged",
};

const SHARED_BLOCK_STATE_KEY = "microclimate-dashboard-blocks-v1";
const SUN_POSITION = new THREE.Vector3(-30, 28, -24);
const SUN_TARGET = new THREE.Vector3(4, 0, 6);
const SUN_DIRECTION = new THREE.Vector3().subVectors(SUN_TARGET, SUN_POSITION).normalize();
const GROUND_SUN_DIRECTION = new THREE.Vector3(SUN_DIRECTION.x, 0, SUN_DIRECTION.z).normalize();
const SHADOW_DIRECTION = GROUND_SUN_DIRECTION.clone().multiplyScalar(-1);

let blocks = loadBlocksFromSharedState() || [];
const canvas = document.getElementById("map3d-canvas");
const details = document.getElementById("block-details");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0e1a);
scene.fog = new THREE.Fog(0x0a0e1a, 45, 115);

const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 200);
camera.position.set(42, 34, 48);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 18;
controls.maxDistance = 90;
controls.maxPolarAngle = Math.PI * 0.47;
controls.target.set(0, 2, 0);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const blockMeshes = [];
const animatedPlumes = [];
const sunlightMeshes = [];
const shadowMeshes = [];
const sunBeamMeshes = [];
const windFlowMeshes = [];
const thermalMeshes = [];
const thermalLabels = [];
const pinnMeshes = [];
const pinnLabels = [];
const blockGroups = [];
const blockLabels = [];
let selectedMesh = null;
let selectedBlockGroup = null;
let selectedHighlight = null;
let selectedBlockId = null;

setupLights();
createGround();
createSunSystem();
initScene();

document.getElementById("btn-reset-camera").addEventListener("click", () => {
    camera.position.set(42, 34, 48);
    controls.target.set(0, 2, 0);
    controls.update();
});

window.addEventListener("resize", resize);
canvas.addEventListener("click", onCanvasClick);
setupEditorControls();

function generateBlocks(count = 20) {
    const generated = [];
    const gridCols = 5;
    const zoneClasses = Object.keys(ZONE_COLORS);

    for (let i = 0; i < count; i++) {
        const col = i % gridCols;
        const row = Math.floor(i / gridCols);
        const zoneClass = zoneClasses[(i * 7) % zoneClasses.length];
        const uhi = 0.6 + ((i * 1.37) % 4.4);
        const height = 8 + ((i * 9) % 52);

        generated.push({
            id: `BLK-${String(i + 1).padStart(2, "0")}`,
            zone_class: zoneClass,
            max_height_m: Math.round(height),
            uhi_intensity: Number(uhi.toFixed(2)),
            svf: Number((0.2 + ((i * 0.11) % 0.62)).toFixed(2)),
            lambda_p: Number((0.12 + ((i * 0.09) % 0.48)).toFixed(2)),
            hw_ratio: Number((0.5 + ((i * 0.37) % 3.0)).toFixed(1)),
            albedo: Number((0.12 + ((i * 0.07) % 0.55)).toFixed(2)),
            green_cover: Math.round(8 + ((i * 13) % 42)),
            wind_speed: Number((1.4 + ((i * 0.47) % 3.2)).toFixed(1)),
            corridor_bearing: Math.round(60 + ((i * 19) % 120)),
            solar_access_hours: Number((3.2 + ((i * 0.39) % 4.5)).toFixed(1)),
            solar_radiation_wm2: Number((420 + ((i * 41) % 340)).toFixed(0)),
            shadow_coverage_pct: Number((12 + ((i * 9) % 58)).toFixed(0)),
            ventilation_score: Number((0.35 + ((i * 0.07) % 0.48)).toFixed(2)),
            heat_burden_score: Number((0.24 + ((i * 0.08) % 0.62)).toFixed(2)),
            surface_temp_c: Number((35 + uhi * 1.25).toFixed(1)),
            air_temp_c: Number((32 + uhi * 0.85).toFixed(1)),
            heat_storage_wm2: Number((120 + uhi * 42).toFixed(0)),
            thermal_risk: uhi > 4.2 ? "extreme" : uhi > 3.2 ? "high" : "moderate",
            col,
            row,
        });
    }

    generated[0].zone_class = "WIND_CORRIDOR_CRITICAL";
    generated[0].max_height_m = 12;
    generated[2].zone_class = "DENSITY_ADAPTIVE";
    generated[2].uhi_intensity = 4.93;
    generated[2].max_height_m = 27;
    generated[2].svf = 0.5;
    generated[2].lambda_p = 0.49;
    generated[2].hw_ratio = 0.8;
    generated[2].albedo = 0.26;
    generated[2].green_cover = 20;
    generated[8].zone_class = "THERMAL_REMEDIATION";
    generated[16].zone_class = "DENSITY_ADAPTIVE";
    generated[16].max_height_m = 36;

    return generated;
}

function loadBlocksFromSharedState() {
    try {
        const saved = JSON.parse(localStorage.getItem(SHARED_BLOCK_STATE_KEY) || "null");
        if (!Array.isArray(saved) || saved.length === 0) return null;
        return saved.map((block, index) => normalizeSharedBlock(block, index));
    } catch (error) {
        console.warn("Unable to load shared dashboard state:", error);
        return null;
    }
}

async function loadActualBlocksFromApi() {
    const response = await fetch("http://127.0.0.1:8000/v1/data/actual?mode=design_peak");
    if (!response.ok) throw new Error(`Actual data API returned ${response.status}`);
    const payload = await response.json();
    blocks = Array.isArray(payload.blocks)
        ? payload.blocks.map((block, index) => normalizeSharedBlock(block, index))
        : [];
    localStorage.setItem(SHARED_BLOCK_STATE_KEY, JSON.stringify(blocks));
}

async function initScene() {
    try {
        if (!blocks.length) {
            await loadActualBlocksFromApi();
        }
    } catch (error) {
        console.warn("Actual data API unavailable for 3D map:", error);
        if (!blocks.length) {
            blocks = generateBlocks().map((block, index) => normalizeSharedBlock(block, index));
            localStorage.setItem(SHARED_BLOCK_STATE_KEY, JSON.stringify(blocks));
        }
    }
    createBlocks();
    resize();
    animate();
    loadCFDMicroclimate();
}

function normalizeSharedBlock(block, index) {
    const col = Number.isFinite(block.col) ? block.col : index % 5;
    const row = Number.isFinite(block.row) ? block.row : Math.floor(index / 5);
    const uhi = Number(block.uhi_intensity ?? 1.5);
    return {
        ...block,
        id: block.id || `BLK-${String(index + 1).padStart(2, "0")}`,
        zone_class: block.zone_class || "BASELINE_UNCHANGED",
        max_height_m: Number(block.max_height_m ?? 24),
        uhi_intensity: Number.isFinite(uhi) ? uhi : 1.5,
        svf: Number(block.svf ?? 0.5),
        lambda_p: Number(block.lambda_p ?? 0.35),
        hw_ratio: Number(block.hw_ratio ?? 1.4),
        albedo: Number(block.albedo ?? 0.28),
        green_cover: Number(block.green_cover ?? 20),
        wind_speed: Number(block.wind_speed ?? 2.5),
        corridor_bearing: Number(block.corridor_bearing ?? 85),
        solar_access_hours: Number(block.solar_access_hours ?? 4.0),
        solar_radiation_wm2: Number(block.solar_radiation_wm2 ?? 480),
        shadow_coverage_pct: Number(block.shadow_coverage_pct ?? 24),
        ventilation_score: Number(block.ventilation_score ?? 0.5),
        heat_burden_score: Number(block.heat_burden_score ?? 0.4),
        surface_temp_c: Number(block.surface_temp_c ?? 36 + uhi),
        air_temp_c: Number(block.air_temp_c ?? 33 + uhi * 0.7),
        heat_storage_wm2: Number(block.heat_storage_wm2 ?? 160 + uhi * 35),
        thermal_risk: block.thermal_risk || (uhi > 4 ? "high" : "moderate"),
        col,
        row,
    };
}

async function loadCFDMicroclimate() {
    try {
        const response = await fetch("http://127.0.0.1:8000/v1/cfd/microclimate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                blocks: blocks.map((block) => ({
                    id: block.id,
                    zone_class: block.zone_class,
                    max_height_m: block.max_height_m,
                    uhi_intensity: block.uhi_intensity,
                    svf: block.svf,
                    lambda_p: block.lambda_p,
                    hw_ratio: block.hw_ratio,
                    albedo: block.albedo,
                    green_cover: block.green_cover,
                })),
                sun_altitude_deg: 57,
                sun_azimuth_deg: 132,
                weather_mode: "design_peak",
            }),
        });
        if (!response.ok) throw new Error(`CFD API returned ${response.status}`);
        const payload = await response.json();
        applyCFDResults(payload.results || []);
        updateHud("CFD + solar model loaded from backend");
        await loadPINNPredictions();
    } catch (error) {
        updateHud("Actual CFD/weather API unavailable");
        console.warn("CFD API unavailable:", error);
        await loadPINNPredictions();
    }
}

function applyCFDResults(results) {
    const byBlock = new Map(results.map((item) => [item.block_id, item]));
    blocks.forEach((block) => {
        const result = byBlock.get(block.id);
        if (!result) return;
        block.wind_speed = result.wind.avg_wind_speed_ms;
        block.corridor_bearing = result.wind.wind_direction_deg + result.wind.deflection_deg;
        block.wind_deflection_deg = result.wind.deflection_deg;
        block.pressure_drop_pa = result.wind.pressure_drop_pa;
        block.ventilation_score = result.wind.ventilation_score;
        block.solar_access_hours = result.solar.solar_access_hours;
        block.solar_radiation_wm2 = result.solar.solar_radiation_wm2;
        block.shadow_coverage_pct = result.solar.shadow_coverage_pct;
        block.shadow_length_m = result.solar.shadow_length_m;
        block.heat_burden_score = result.microclimate.heat_burden_score;
        block.combined_risk = result.microclimate.combined_risk;
        if (result.thermal) {
            block.surface_temp_c = result.thermal.surface_temp_c;
            block.air_temp_c = result.thermal.air_temp_c;
            block.thermal_uhi_intensity_c = result.thermal.uhi_intensity_c;
            block.heat_storage_wm2 = result.thermal.heat_storage_wm2;
            block.thermal_risk = result.thermal.thermal_risk;
        }
    });
    refreshSimulationLayers();
}

async function loadPINNPredictions() {
    try {
        const response = await fetch("http://127.0.0.1:8000/v1/pinn/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                blocks: blocks.map((block) => ({
                    id: block.id,
                    zone_class: block.zone_class,
                    max_height_m: block.max_height_m,
                    uhi_intensity: block.uhi_intensity,
                    svf: block.svf,
                    lambda_p: block.lambda_p,
                    hw_ratio: block.hw_ratio,
                    albedo: block.albedo,
                    green_cover: block.green_cover,
                    wind_speed: block.wind_speed,
                    solar_radiation_wm2: block.solar_radiation_wm2,
                    surface_temp_c: block.surface_temp_c,
                })),
            }),
        });
        if (!response.ok) throw new Error(`PINN API returned ${response.status}`);
        const payload = await response.json();
        applyPINNResults(payload.results || []);
        updatePINNBadge(true);
    } catch (error) {
        updatePINNBadge(false);
        console.warn("PINN API unavailable:", error);
    }
}

function applyPINNResults(results) {
    const byBlock = new Map(results.map((item) => [item.block_id, item.pinn || {}]));
    blocks.forEach((block) => {
        const result = byBlock.get(block.id);
        if (!result) return;
        block.pinn_delta_temp_c = Number(result.predicted_delta_temp_c ?? 0);
        block.pinn_surface_temp_c = Number(result.predicted_surface_temp_c ?? block.surface_temp_c);
        block.pinn_wind_factor = Number(result.predicted_wind_factor ?? 0.5);
        block.pinn_height_sensitivity = Number(result.height_sensitivity_c_per_10m ?? 0);
        block.pinn_green_sensitivity = Number(result.green_cover_sensitivity_c_per_10pct ?? 0);
        block.pinn_confidence = Number(result.confidence ?? 0.5);
        block.pinn_ood_warning = Boolean(result.ood_warning);
        block.pinn_ood_features = Array.isArray(result.ood_features) ? result.ood_features : [];
    });
    refreshPINNLayers();
}

function updatePINNBadge(usingBackend) {
    const badge = document.getElementById("pinn-status-badge");
    if (!badge) return;
    badge.textContent = usingBackend ? "PINN Actual Inputs" : "PINN Offline";
    badge.classList.remove("loading", "active", "fallback");
    badge.classList.add(usingBackend ? "active" : "fallback");
}

function updateHud(message) {
    const copy = document.querySelector(".hud-copy");
    if (copy) {
        copy.textContent = `${message}. Drag to rotate, scroll to zoom, click a block to inspect it.`;
    }
    const badge = document.getElementById("cfd-status-badge");
    if (badge) {
        const usingBackend = message.toLowerCase().includes("backend");
        badge.textContent = usingBackend ? "Actual CFD Active" : "Actual CFD Offline";
        badge.classList.remove("loading", "active", "fallback");
        badge.classList.add(usingBackend ? "active" : "fallback");
    }
}

function saveBlocksToSharedState() {
    localStorage.setItem(SHARED_BLOCK_STATE_KEY, JSON.stringify(blocks));
}

function setupLights() {
    const ambient = new THREE.HemisphereLight(0xc7d2fe, 0x111827, 2.3);
    scene.add(ambient);

    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(18, 32, 20);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 1;
    key.shadow.camera.far = 80;
    key.shadow.camera.left = -34;
    key.shadow.camera.right = 34;
    key.shadow.camera.top = 34;
    key.shadow.camera.bottom = -34;
    scene.add(key);
}

function createGround() {
    const ground = new THREE.Mesh(
        new THREE.PlaneGeometry(76, 62),
        new THREE.MeshStandardMaterial({
            color: 0x0f172a,
            roughness: 0.95,
            metalness: 0.02,
        }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    createHeatSurface();
    createRoadNetwork();
    createStreetTrees();
    createCrosswalks();

    const grid = new THREE.GridHelper(76, 20, 0x26314c, 0x1d263b);
    grid.position.y = 0.02;
    scene.add(grid);
}

function createSunSystem() {
    const sun = new THREE.Mesh(
        new THREE.SphereGeometry(1.05, 32, 16),
        new THREE.MeshBasicMaterial({ color: 0xfacc15 }),
    );
    sun.position.copy(SUN_POSITION);
    scene.add(sun);

    const glow = new THREE.Sprite(new THREE.SpriteMaterial({
        map: makeRadialTexture("#fef08a", "#f97316"),
        transparent: true,
        opacity: 0.72,
        depthWrite: false,
    }));
    glow.position.copy(sun.position);
    glow.scale.set(8, 8, 1);
    scene.add(glow);

    const beamMaterial = new THREE.MeshBasicMaterial({
        color: 0xfacc15,
        transparent: true,
        opacity: 0.22,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
    });
    const landingPoints = [
        [-22, -12], [-12, -6], [-2, -15], [8, -8], [18, -12],
        [-18, 4], [-6, 8], [6, 2], [16, 7], [24, 0],
        [-10, 19], [2, 17], [14, 18],
    ];
    landingPoints.forEach(([x, z], index) => {
        const end = new THREE.Vector3(x, 0.55, z);
        const start = SUN_POSITION.clone().lerp(end, 0.08);
        const mid = SUN_POSITION.clone().lerp(end, 0.52);
        const curve = new THREE.CatmullRomCurve3([start, mid, end]);
        const ray = new THREE.Mesh(
            new THREE.TubeGeometry(curve, 20, index % 3 === 0 ? 0.045 : 0.03, 8, false),
            beamMaterial.clone(),
        );
        ray.material.opacity = index % 3 === 0 ? 0.28 : 0.18;
        scene.add(ray);
    });

    const targetMarker = new THREE.Mesh(
        new THREE.RingGeometry(4.8, 5.05, 80),
        new THREE.MeshBasicMaterial({
            color: 0xfacc15,
            transparent: true,
            opacity: 0.2,
            side: THREE.DoubleSide,
        }),
    );
    targetMarker.rotation.x = -Math.PI / 2;
    targetMarker.position.set(SUN_TARGET.x, 0.12, SUN_TARGET.z);
    scene.add(targetMarker);
}

function alignObjectToVector(object, direction) {
    const quaternion = new THREE.Quaternion();
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize());
    object.quaternion.copy(quaternion);
}

function orientFlatToGroundDirection(mesh, direction) {
    mesh.rotation.x = -Math.PI / 2;
    mesh.rotation.z = Math.atan2(direction.x, direction.z);
}

function worldSunBeamLength(block, tallest) {
    const radiation = block.solar_radiation_wm2 || 480;
    return Math.max(5, tallest + 3.5 + radiation / 170);
}

function localSunStart(length) {
    return SUN_DIRECTION.clone().multiplyScalar(-length * 0.34);
}

function localSunEnd(length) {
    return SUN_DIRECTION.clone().multiplyScalar(length * 0.34);
}

function sunGroundOffset(distance) {
    return new THREE.Vector3(SHADOW_DIRECTION.x * distance, 0, SHADOW_DIRECTION.z * distance);
}

function sunlitGroundOffset(distance) {
    return new THREE.Vector3(GROUND_SUN_DIRECTION.x * distance, 0, GROUND_SUN_DIRECTION.z * distance);
}

function setVectorPosition(object, vector) {
    object.position.set(vector.x, vector.y, vector.z);
}

function makeSunBeamCurve(length, lateralOffset = 0) {
    const start = localSunStart(length);
    const end = localSunEnd(length);
    const lateral = new THREE.Vector3(-GROUND_SUN_DIRECTION.z, 0, GROUND_SUN_DIRECTION.x).multiplyScalar(lateralOffset);
    return new THREE.CatmullRomCurve3([
        start.clone().add(lateral),
        new THREE.Vector3(0, 1.1, 0).add(lateral),
        end.clone().add(lateral),
    ]);
}

function createHeatSurface() {
    const spacing = 9;
    const xOffset = -2 * spacing;
    const zOffset = -1.5 * spacing;
    blocks.forEach((block) => {
        const cell = new THREE.Mesh(
            new THREE.PlaneGeometry(7.2, 7.2),
            new THREE.MeshBasicMaterial({
                color: heatColor(block.uhi_intensity),
                transparent: true,
                opacity: 0.16 + block.uhi_intensity * 0.035,
                depthWrite: false,
            }),
        );
        cell.rotation.x = -Math.PI / 2;
        cell.position.set(xOffset + block.col * spacing, 0.075, zOffset + block.row * spacing);
        scene.add(cell);
    });
}

function createRoadNetwork() {
    const roadMaterial = new THREE.MeshStandardMaterial({
        color: 0x1f2937,
        roughness: 0.88,
        metalness: 0.02,
    });
    const sidewalkMaterial = new THREE.MeshStandardMaterial({
        color: 0x475569,
        roughness: 0.82,
        metalness: 0.02,
    });
    const laneMaterial = new THREE.MeshBasicMaterial({
        color: 0x94a3b8,
        transparent: true,
        opacity: 0.32,
    });
    const spacing = 9;

    for (let i = -3; i <= 3; i++) {
        const road = new THREE.Mesh(new THREE.PlaneGeometry(2.1, 58), roadMaterial);
        road.rotation.x = -Math.PI / 2;
        road.position.set(i * spacing + 4.5, 0.04, 0);
        scene.add(road);

        [-1.55, 1.55].forEach((offset) => {
            const walk = new THREE.Mesh(new THREE.PlaneGeometry(0.68, 58), sidewalkMaterial);
            walk.rotation.x = -Math.PI / 2;
            walk.position.set(i * spacing + 4.5 + offset, 0.062, 0);
            scene.add(walk);
        });

        const lane = new THREE.Mesh(new THREE.PlaneGeometry(0.08, 54), laneMaterial);
        lane.rotation.x = -Math.PI / 2;
        lane.position.set(i * spacing + 4.5, 0.055, 0);
        scene.add(lane);
    }

    for (let i = -3; i <= 3; i++) {
        const road = new THREE.Mesh(new THREE.PlaneGeometry(74, 2.1), roadMaterial);
        road.rotation.x = -Math.PI / 2;
        road.position.set(0, 0.045, i * spacing + 4.5);
        scene.add(road);

        [-1.55, 1.55].forEach((offset) => {
            const walk = new THREE.Mesh(new THREE.PlaneGeometry(74, 0.68), sidewalkMaterial);
            walk.rotation.x = -Math.PI / 2;
            walk.position.set(0, 0.064, i * spacing + 4.5 + offset);
            scene.add(walk);
        });

        const lane = new THREE.Mesh(new THREE.PlaneGeometry(70, 0.08), laneMaterial);
        lane.rotation.x = -Math.PI / 2;
        lane.position.set(0, 0.06, i * spacing + 4.5);
        scene.add(lane);
    }
}

function createCrosswalks() {
    const stripeMaterial = new THREE.MeshBasicMaterial({
        color: 0xe5e7eb,
        transparent: true,
        opacity: 0.42,
    });
    const intersections = [-13.5, -4.5, 4.5, 13.5];
    intersections.forEach((x) => {
        intersections.forEach((z) => {
            for (let i = -2; i <= 2; i++) {
                const stripeA = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 0.12), stripeMaterial);
                stripeA.rotation.x = -Math.PI / 2;
                stripeA.position.set(x + i * 0.42, 0.082, z + 3.0);
                scene.add(stripeA);

                const stripeB = new THREE.Mesh(new THREE.PlaneGeometry(0.12, 1.4), stripeMaterial);
                stripeB.rotation.x = -Math.PI / 2;
                stripeB.position.set(x + 3.0, 0.084, z + i * 0.42);
                scene.add(stripeB);
            }
        });
    });
}

function createStreetTrees() {
    const treePositions = [];
    const roadLines = [-22.5, -13.5, -4.5, 4.5, 13.5, 22.5];
    const corridorStops = [-18, -9, 0, 9, 18];

    roadLines.forEach((roadX, roadIndex) => {
        corridorStops.forEach((z, stopIndex) => {
            if ((roadIndex + stopIndex) % 2 === 0) {
                treePositions.push({
                    x: roadX - 1.95,
                    z: z + 0.65,
                    scale: 0.95 + ((roadIndex + stopIndex) % 3) * 0.12,
                    type: "street",
                });
                treePositions.push({
                    x: roadX + 1.95,
                    z: z - 0.65,
                    scale: 0.9 + ((roadIndex + stopIndex + 1) % 3) * 0.1,
                    type: "street",
                });
            }
        });
    });

    roadLines.forEach((roadZ, roadIndex) => {
        corridorStops.forEach((x, stopIndex) => {
            if ((roadIndex + stopIndex) % 2 === 1) {
                treePositions.push({
                    x: x - 0.65,
                    z: roadZ - 1.95,
                    scale: 0.9 + ((roadIndex + stopIndex) % 4) * 0.08,
                    type: "street",
                });
                treePositions.push({
                    x: x + 0.65,
                    z: roadZ + 1.95,
                    scale: 0.95 + ((roadIndex + stopIndex + 2) % 3) * 0.1,
                    type: "street",
                });
            }
        });
    });

    const cornerPlazas = [
        [-27.5, -18.5], [27.5, -18.5], [-27.5, 18.5], [27.5, 18.5],
        [-9.0, -18.5], [9.0, 18.5],
    ];
    cornerPlazas.forEach(([x, z], index) => {
        treePositions.push({ x, z, scale: 1.2, type: index % 2 === 0 ? "round" : "tall" });
        treePositions.push({ x: x + 1.2, z: z + 0.7, scale: 0.92, type: "round" });
    });

    treePositions.forEach((tree, index) => {
        createTree(scene, tree.x, tree.z, tree.scale, tree.type, index);
    });
}

function createTree(parent, x, z, scale = 1, type = "round", index = 0) {
    const trunk = new THREE.Mesh(
        new THREE.CylinderGeometry(0.055 * scale, 0.095 * scale, 0.72 * scale, 8),
        new THREE.MeshStandardMaterial({ color: 0x7c2d12, roughness: 0.88 }),
    );
    trunk.position.set(x, 0.38 * scale, z);
    trunk.castShadow = true;
    parent.add(trunk);

    const palette = [0x166534, 0x15803d, 0x16a34a, 0x65a30d];
    const canopyMaterial = new THREE.MeshStandardMaterial({
        color: palette[index % palette.length],
        roughness: 0.94,
    });

    if (type === "tall") {
        const canopy = new THREE.Mesh(
            new THREE.ConeGeometry(0.42 * scale, 1.05 * scale, 10),
            canopyMaterial,
        );
        canopy.position.set(x, 1.12 * scale, z);
        canopy.castShadow = true;
        parent.add(canopy);
    } else {
        const canopy = new THREE.Group();
        const offsets = [
            [0, 0, 0],
            [0.22, -0.05, 0.08],
            [-0.18, 0.02, -0.16],
        ];
        offsets.forEach(([ox, oy, oz], partIndex) => {
            const crown = new THREE.Mesh(
                new THREE.SphereGeometry((0.34 + partIndex * 0.035) * scale, 12, 8),
                canopyMaterial,
            );
            crown.position.set(x + ox * scale, (0.86 + oy) * scale, z + oz * scale);
            crown.castShadow = true;
            canopy.add(crown);
        });
        parent.add(canopy);
    }

    if (type === "street") {
        const treePit = new THREE.Mesh(
            new THREE.CylinderGeometry(0.42 * scale, 0.42 * scale, 0.035, 18),
            new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.8 }),
        );
        treePit.position.set(x, 0.085, z);
        parent.add(treePit);
    }
}

function createBench(parent, x, z, rotation = 0) {
    const seatMaterial = new THREE.MeshStandardMaterial({ color: 0x854d0e, roughness: 0.72 });
    const metalMaterial = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.35, roughness: 0.45 });
    const bench = new THREE.Group();
    const seat = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.09, 0.22), seatMaterial);
    seat.position.y = 0.34;
    bench.add(seat);
    [-0.32, 0.32].forEach((offset) => {
        const leg = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.32, 0.06), metalMaterial);
        leg.position.set(offset, 0.18, 0);
        bench.add(leg);
    });
    bench.position.set(x, 0, z);
    bench.rotation.y = rotation;
    parent.add(bench);
}

function createPlanter(parent, x, z, w = 0.9, d = 0.35) {
    const planter = new THREE.Mesh(
        new THREE.BoxGeometry(w, 0.22, d),
        new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.78 }),
    );
    planter.position.set(x, 0.17, z);
    parent.add(planter);
    const hedge = new THREE.Mesh(
        new THREE.BoxGeometry(w * 0.82, 0.16, d * 0.7),
        new THREE.MeshStandardMaterial({ color: 0x22c55e, roughness: 0.9 }),
    );
    hedge.position.set(x, 0.36, z);
    parent.add(hedge);
}

function createParcelAmenities(group, block) {
    const hasCourtyard = block.green_cover >= 18;
    if (hasCourtyard) {
        const plaza = new THREE.Mesh(
            new THREE.PlaneGeometry(2.2, 1.55),
            new THREE.MeshStandardMaterial({ color: 0x64748b, roughness: 0.85 }),
        );
        plaza.rotation.x = -Math.PI / 2;
        plaza.position.set(-2.45, 0.19, 2.55);
        group.add(plaza);
        createBench(group, -2.65, 2.35, 0.2);
        createPlanter(group, -1.92, 2.82, 0.72, 0.32);
    }

    if (block.zone_class === "THERMAL_REMEDIATION") {
        createPlanter(group, 2.65, -2.75, 1.15, 0.4);
        createPlanter(group, 2.65, -2.15, 1.15, 0.4);
    }
}

function createFacadeDetails(group, body, w, h, d, block, buildingIndex) {
    const windowMaterial = new THREE.MeshBasicMaterial({
        color: block.albedo > 0.42 ? 0xbae6fd : 0x93c5fd,
        transparent: true,
        opacity: 0.5,
    });
    const floors = Math.max(1, Math.floor(h / 0.85));
    const columns = Math.max(1, Math.floor(w / 0.55));

    for (let floor = 0; floor < floors; floor++) {
        for (let col = 0; col < columns; col++) {
            if ((floor + col + buildingIndex) % 3 === 0) continue;
            const windowPane = new THREE.Mesh(new THREE.PlaneGeometry(0.22, 0.18), windowMaterial);
            const x = body.position.x - w / 2 + 0.35 + col * 0.48;
            const y = body.position.y - h / 2 + 0.48 + floor * 0.72;
            const z = body.position.z + d / 2 + 0.006;
            windowPane.position.set(x, y, z);
            group.add(windowPane);
        }
    }
}

function createRooftopEquipment(group, roof, block, w, d) {
    if (block.max_height_m < 18) return;
    const unit = new THREE.Mesh(
        new THREE.BoxGeometry(Math.min(0.55, w * 0.28), 0.22, Math.min(0.44, d * 0.25)),
        new THREE.MeshStandardMaterial({ color: 0x94a3b8, roughness: 0.62, metalness: 0.18 }),
    );
    unit.position.set(roof.position.x + w * 0.24, roof.position.y + 0.19, roof.position.z - d * 0.22);
    group.add(unit);
}

function createMicroclimateMarker(group, block) {
    const markerColor = block.heat_burden_score > 0.55 ? 0xef4444 : block.ventilation_score > 0.6 ? 0x38bdf8 : 0xf59e0b;
    const marker = new THREE.Mesh(
        new THREE.CylinderGeometry(0.12, 0.12, 0.04, 16),
        new THREE.MeshBasicMaterial({ color: markerColor }),
    );
    marker.position.set(3.05, 0.28, 3.05);
    group.add(marker);
}

function createBlocks() {
    const spacing = 9;
    const xOffset = -2 * spacing;
    const zOffset = -1.5 * spacing;

    blocks.forEach((block) => {
        const baseX = xOffset + block.col * spacing;
        const baseZ = zOffset + block.row * spacing;
        const group = new THREE.Group();
        group.userData.block = block;
        group.position.set(baseX, 0, baseZ);
        scene.add(group);
        blockGroups.push(group);

        createParcel(group, block);
        createBlockHitArea(group, block);
        const tallest = createBuildingCluster(group, block);
        createGreenCover(group, block);
        createParcelAmenities(group, block);
        createMicroclimateMarker(group, block);
        createThermalLayer(group, block);
        createPINNLayer(group, block);
        createHeatPlume(group, block, tallest);
        createSunlightPatch(group, block);
        createSolarBeam(group, block, tallest);
        createBlockShadow(group, block, tallest);
        createZoneMarker(group, block, tallest);
        createWindFlow(group, block);
        createSensorMast(group, block, tallest);

        const label = makeLabel(block.id);
        label.position.set(baseX, tallest + 1.1, baseZ);
        scene.add(label);
        blockLabels.push(label);
    });

    populateBlockEditor();
}

function clearBlockScene() {
    blockGroups.forEach((group) => scene.remove(group));
    blockLabels.forEach((label) => scene.remove(label));
    blockGroups.length = 0;
    blockLabels.length = 0;
    blockMeshes.length = 0;
    animatedPlumes.length = 0;
    sunlightMeshes.length = 0;
    shadowMeshes.length = 0;
    sunBeamMeshes.length = 0;
    windFlowMeshes.length = 0;
    thermalMeshes.length = 0;
    thermalLabels.length = 0;
    pinnMeshes.length = 0;
    pinnLabels.length = 0;
    selectedMesh = null;
    selectedBlockGroup = null;
    if (selectedHighlight) {
        selectedHighlight.parent?.remove(selectedHighlight);
        selectedHighlight.geometry.dispose();
        selectedHighlight.material.dispose();
        selectedHighlight = null;
    }
}

function rebuildBlockScene() {
    clearBlockScene();
    createBlocks();
    if (selectedBlockId) {
        selectBlockById(selectedBlockId);
    }
}

function populateBlockEditor() {
    const select = document.getElementById("edit-block-select");
    if (!select) return;
    select.innerHTML = '<option value="">Select a block</option>';
    blocks.forEach((block) => {
        const option = document.createElement("option");
        option.value = block.id;
        option.textContent = `${block.id} - ${ZONE_LABELS[block.zone_class]}`;
        select.appendChild(option);
    });
    if (selectedBlockId) {
        select.value = selectedBlockId;
    }
}

function syncEditorControls(block) {
    const select = document.getElementById("edit-block-select");
    const height = document.getElementById("edit-height");
    const green = document.getElementById("edit-green-cover");
    const albedo = document.getElementById("edit-albedo");
    if (!select || !height || !green || !albedo) return;
    select.value = block.id;
    height.value = Math.round(block.max_height_m);
    green.value = Math.round(block.green_cover);
    albedo.value = Number(block.albedo).toFixed(2);
    updateEditorValueLabels();
}

function updateEditorValueLabels() {
    const height = document.getElementById("edit-height");
    const green = document.getElementById("edit-green-cover");
    const albedo = document.getElementById("edit-albedo");
    const heightValue = document.getElementById("edit-height-value");
    const greenValue = document.getElementById("edit-green-cover-value");
    const albedoValue = document.getElementById("edit-albedo-value");
    if (height && heightValue) heightValue.textContent = `${height.value}m`;
    if (green && greenValue) greenValue.textContent = `${green.value}%`;
    if (albedo && albedoValue) albedoValue.textContent = Number(albedo.value).toFixed(2);
}

function setupEditorControls() {
    ["edit-height", "edit-green-cover", "edit-albedo"].forEach((id) => {
        document.getElementById(id)?.addEventListener("input", updateEditorValueLabels);
    });
    document.getElementById("edit-block-select")?.addEventListener("change", (event) => {
        if (!event.target.value) return;
        selectBlockById(event.target.value);
    });
    document.getElementById("btn-apply-3d-edits")?.addEventListener("click", apply3DEdits);
}

function apply3DEdits() {
    const blockId = document.getElementById("edit-block-select")?.value;
    if (!blockId) return;
    const block = blocks.find((item) => item.id === blockId);
    if (!block) return;

    const newHeight = Number(document.getElementById("edit-height").value);
    const newGreen = Number(document.getElementById("edit-green-cover").value);
    const newAlbedo = Number(document.getElementById("edit-albedo").value);

    const heightDelta = newHeight - Number(block.max_height_m);
    const greenDelta = newGreen - Number(block.green_cover);
    const albedoDelta = newAlbedo - Number(block.albedo);
    const uhiShift = heightDelta * 0.028 - greenDelta * 0.018 - albedoDelta * 1.9;

    block.max_height_m = newHeight;
    block.green_cover = newGreen;
    block.albedo = Number(newAlbedo.toFixed(2));
    block.uhi_intensity = Math.max(0, Number((block.uhi_intensity + uhiShift).toFixed(2)));
    block.surface_temp_c = Number(((block.surface_temp_c || 36) + uhiShift * 0.85).toFixed(1));
    block.air_temp_c = Number(((block.air_temp_c || 33) + uhiShift * 0.45).toFixed(1));
    block.heat_storage_wm2 = Math.max(80, Number(((block.heat_storage_wm2 || 160) + heightDelta * 2.8 - greenDelta * 0.9).toFixed(0)));

    selectedBlockId = block.id;
    saveBlocksToSharedState();
    rebuildBlockScene();
    loadCFDMicroclimate();
}

function createBlockHitArea(group, block) {
    const hitArea = new THREE.Mesh(
        new THREE.BoxGeometry(8.05, 0.5, 8.05),
        new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0,
            depthWrite: false,
        }),
    );
    hitArea.position.y = 0.32;
    hitArea.userData.block = block;
    hitArea.userData.group = group;
    hitArea.userData.isHitArea = true;
    hitArea.renderOrder = -1;
    blockMeshes.push(hitArea);
    group.add(hitArea);
}

function createParcel(group, block) {
    const parcel = new THREE.Mesh(
        new THREE.BoxGeometry(7.25, 0.14, 7.25),
        new THREE.MeshStandardMaterial({
            color: 0x162033,
            roughness: 0.86,
            metalness: 0.02,
        }),
    );
    parcel.position.y = 0.07;
    parcel.receiveShadow = true;
    group.add(parcel);

    const outline = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(7.32, 0.16, 7.32)),
        new THREE.LineBasicMaterial({ color: ZONE_COLORS[block.zone_class], transparent: true, opacity: 0.68 }),
    );
    outline.position.y = 0.12;
    group.add(outline);
}

function createBuildingCluster(group, block) {
    const heat = heatColor(block.uhi_intensity);
    const massing = buildingMassingForBlock(block);
    let tallest = 0;

    for (let i = 0; i < massing.length; i++) {
        const spec = massing[i];
        const h = spec.height;
        tallest = Math.max(tallest, h);
        const w = spec.width;
        const d = spec.depth;

        const body = new THREE.Mesh(
            new THREE.BoxGeometry(w, h, d),
            new THREE.MeshStandardMaterial({
                color: heat.clone().lerp(new THREE.Color(spec.colorTint), 0.18),
                emissive: heat,
                emissiveIntensity: block.uhi_intensity > 3.8 ? 0.09 : 0.025,
                roughness: spec.roughness,
                metalness: spec.metalness,
            }),
        );
        body.position.set(spec.x, h / 2 + 0.14, spec.z);
        body.castShadow = true;
        body.receiveShadow = true;
        body.userData.block = block;
        body.userData.group = group;
        blockMeshes.push(body);
        group.add(body);

        addFacadeLines(group, body, w, h, d);
        createFacadeDetails(group, body, w, h, d, block, i);
        addRoofTreatment(group, block, body, w, d);

        if (spec.setbackHeight > 0) {
            createTowerSetback(group, block, body, spec, heat, i);
        }
    }

    return tallest + 0.16;
}

function buildingMassingForBlock(block) {
    const planArea = Number(block.lambda_p || 0.3);
    const blockHeight = Math.max(7, Number(block.max_height_m || 12));
    const osmCount = Math.max(0, Number(block.osm_building_count || 0));
    const densityBias = Math.min(1, planArea * 1.35 + osmCount / 28);
    const compactness = Math.min(1, Math.max(0, Number(block.hw_ratio || 1.1) / 2.3));
    const towerHeight = Math.max(2.4, blockHeight * 0.3);

    if (densityBias > 0.82) {
        return [
            {
                x: -0.15,
                z: 0.1,
                width: 4.9,
                depth: 3.9,
                height: Math.max(2.1, towerHeight * 0.36),
                setbackHeight: Math.max(4.6, towerHeight * 1.08),
                setbackWidth: 1.95,
                setbackDepth: 1.8,
                setbackOffsetX: -0.25,
                setbackOffsetZ: 0.05,
                colorTint: 0xe5e7eb,
                roughness: 0.5,
                metalness: 0.12,
            },
            {
                x: 2.2,
                z: -2.1,
                width: 1.1,
                depth: 1.55,
                height: Math.max(2.1, towerHeight * 0.34),
                setbackHeight: Math.max(2.8, towerHeight * 0.56),
                setbackWidth: 0.82,
                setbackDepth: 0.9,
                setbackOffsetX: -0.04,
                setbackOffsetZ: 0.02,
                colorTint: 0xcbd5e1,
                roughness: 0.56,
                metalness: 0.1,
            },
        ];
    }

    if (densityBias > 0.56) {
        return [
            {
                x: -1.65,
                z: -0.15,
                width: 2.35,
                depth: 4.5,
                height: Math.max(2.0, towerHeight * (0.44 + compactness * 0.16)),
                setbackHeight: Math.max(2.8, towerHeight * 0.76),
                setbackWidth: 1.28,
                setbackDepth: 2.05,
                setbackOffsetX: 0.22,
                setbackOffsetZ: 0.18,
                colorTint: 0xe2e8f0,
                roughness: 0.56,
                metalness: 0.08,
            },
            {
                x: 1.55,
                z: 0.25,
                width: 2.1,
                depth: 4.05,
                height: Math.max(1.8, towerHeight * (0.38 + densityBias * 0.18)),
                setbackHeight: Math.max(2.0, towerHeight * 0.42),
                setbackWidth: 1.05,
                setbackDepth: 1.55,
                setbackOffsetX: 0.14,
                setbackOffsetZ: -0.12,
                colorTint: 0xdbeafe,
                roughness: 0.58,
                metalness: 0.08,
            },
        ];
    }

    if (densityBias > 0.3) {
        return [
            {
                x: -1.85,
                z: -1.2,
                width: 2.2,
                depth: 2.65,
                height: Math.max(1.6, towerHeight * 0.36),
                setbackHeight: Math.max(1.2, towerHeight * 0.26),
                setbackWidth: 1.25,
                setbackDepth: 1.45,
                setbackOffsetX: -0.08,
                setbackOffsetZ: 0.03,
                colorTint: 0xe2e8f0,
                roughness: 0.62,
                metalness: 0.05,
            },
            {
                x: 1.35,
                z: -0.55,
                width: 1.75,
                depth: 2.35,
                height: Math.max(1.45, towerHeight * 0.3),
                setbackHeight: 0,
                colorTint: 0xdbeafe,
                roughness: 0.64,
                metalness: 0.05,
            },
            {
                x: -0.2,
                z: 1.85,
                width: 3.15,
                depth: 1.55,
                height: Math.max(1.2, towerHeight * 0.22),
                setbackHeight: 0,
                colorTint: 0xf1f5f9,
                roughness: 0.68,
                metalness: 0.02,
            },
        ];
    }

    return [
        {
            x: -1.6,
            z: -0.2,
            width: 1.95,
            depth: 2.35,
            height: Math.max(1.35, towerHeight * 0.24),
            setbackHeight: 0,
            colorTint: 0xe5e7eb,
            roughness: 0.68,
            metalness: 0.02,
        },
        {
            x: 1.25,
            z: 0.1,
            width: 1.75,
            depth: 2.05,
            height: Math.max(1.15, towerHeight * 0.2),
            setbackHeight: 0,
            colorTint: 0xdbeafe,
            roughness: 0.68,
            metalness: 0.02,
        },
    ];
}

function createTowerSetback(group, block, body, spec, heat, buildingIndex) {
    const setback = new THREE.Mesh(
        new THREE.BoxGeometry(spec.setbackWidth, spec.setbackHeight, spec.setbackDepth),
        new THREE.MeshStandardMaterial({
            color: heat.clone().lerp(new THREE.Color(0xf8fafc), 0.3),
            emissive: heat,
            emissiveIntensity: block.uhi_intensity > 3.8 ? 0.07 : 0.02,
            roughness: Math.max(0.48, spec.roughness - 0.08),
            metalness: spec.metalness + 0.03,
        }),
    );
    setback.position.set(
        body.position.x + (spec.setbackOffsetX || 0),
        body.position.y + body.geometry.parameters.height / 2 + spec.setbackHeight / 2,
        body.position.z + (spec.setbackOffsetZ || 0),
    );
    setback.castShadow = true;
    setback.receiveShadow = true;
    setback.userData.block = block;
    setback.userData.group = group;
    blockMeshes.push(setback);
    group.add(setback);

    addFacadeLines(group, setback, spec.setbackWidth, spec.setbackHeight, spec.setbackDepth);
    createFacadeDetails(group, setback, spec.setbackWidth, spec.setbackHeight, spec.setbackDepth, block, buildingIndex + 5);
    addRoofTreatment(group, block, setback, spec.setbackWidth, spec.setbackDepth);
}

function addFacadeLines(group, body, w, h, d) {
    const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(w + 0.02, h + 0.02, d + 0.02)),
        new THREE.LineBasicMaterial({ color: 0xf8fafc, transparent: true, opacity: 0.16 }),
    );
    edges.position.copy(body.position);
    group.add(edges);
}

function addRoofTreatment(group, block, body, w, d) {
    const roofColor = block.albedo > 0.45 ? 0xe2e8f0 : 0x334155;
    const roof = new THREE.Mesh(
        new THREE.BoxGeometry(w + 0.08, 0.08, d + 0.08),
        new THREE.MeshStandardMaterial({
            color: roofColor,
            roughness: block.albedo > 0.45 ? 0.42 : 0.78,
        }),
    );
    roof.position.set(body.position.x, body.position.y + body.geometry.parameters.height / 2 + 0.06, body.position.z);
    group.add(roof);
    createRooftopEquipment(group, roof, block, w, d);

    if (block.green_cover > 25) {
        const greenRoof = new THREE.Mesh(
            new THREE.BoxGeometry(w * 0.64, 0.09, d * 0.62),
            new THREE.MeshStandardMaterial({ color: 0x22c55e, roughness: 0.9 }),
        );
        greenRoof.position.set(roof.position.x, roof.position.y + 0.07, roof.position.z);
        group.add(greenRoof);
    }

    if (block.albedo > 0.38) {
        const panels = new THREE.Group();
        for (let i = 0; i < 2; i++) {
            const panel = new THREE.Mesh(
                new THREE.BoxGeometry(w * 0.35, 0.035, d * 0.22),
                new THREE.MeshStandardMaterial({
                    color: 0x0f172a,
                    roughness: 0.35,
                    metalness: 0.35,
                    emissive: 0x1e40af,
                    emissiveIntensity: 0.08,
                }),
            );
            panel.position.set((i - 0.5) * w * 0.32, 0, 0);
            panels.add(panel);
        }
        panels.rotation.y = -0.25;
        panels.position.set(roof.position.x, roof.position.y + 0.1, roof.position.z);
        group.add(panels);
    }

    if (block.zone_class === "THERMAL_REMEDIATION") {
        const coolPavement = new THREE.Mesh(
            new THREE.PlaneGeometry(1.35, 1.35),
            new THREE.MeshBasicMaterial({ color: 0x93c5fd, transparent: true, opacity: 0.38 }),
        );
        coolPavement.rotation.x = -Math.PI / 2;
        coolPavement.position.set(body.position.x, 0.22, body.position.z + d * 0.9);
        group.add(coolPavement);
    }
}

function createGreenCover(group, block) {
    const count = Math.max(1, Math.round(block.green_cover / 14));
    const pocketSites = [
        [-3.05, 2.85, "round"],
        [-2.35, 2.15, "street"],
        [2.85, 2.85, "tall"],
        [2.85, -2.85, "round"],
        [-2.85, -2.85, "street"],
    ];
    for (let i = 0; i < count; i++) {
        const site = pocketSites[(i + block.col + block.row) % pocketSites.length];
        const jitterX = ((block.max_height_m + i * 3) % 7) * 0.035;
        const jitterZ = ((block.green_cover + i * 5) % 7) * 0.035;
        createTree(group, site[0] + jitterX, site[1] - jitterZ, 0.72 + i * 0.04, site[2], i + block.max_height_m);
    }
}

function createHeatPlume(group, block, tallest) {
    if (block.uhi_intensity < 2.4) return;
    const plume = new THREE.Mesh(
        new THREE.CylinderGeometry(2.7, 1.4, 3.5 + block.uhi_intensity * 0.45, 32, 1, true),
        new THREE.MeshBasicMaterial({
            color: heatColor(block.uhi_intensity),
            transparent: true,
            opacity: Math.min(0.3, 0.05 + block.heat_burden_score * 0.24),
            side: THREE.DoubleSide,
            depthWrite: false,
        }),
    );
    plume.position.y = tallest + 1.8;
    plume.userData.spin = 0.0015 + block.uhi_intensity * 0.0008;
    group.add(plume);
    animatedPlumes.push(plume);
}

function createThermalLayer(group, block) {
    const thermalColor = thermalColorForTemp(block.surface_temp_c);
    const ring = new THREE.Mesh(
        new THREE.RingGeometry(2.65, 3.25, 72),
        new THREE.MeshBasicMaterial({
            color: thermalColor,
            transparent: true,
            opacity: thermalOpacity(block),
            side: THREE.DoubleSide,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
        }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(0, 0.255, 0);
    ring.userData.block = block;
    thermalMeshes.push(ring);
    group.add(ring);

    const storageDisk = new THREE.Mesh(
        new THREE.CircleGeometry(1.05, 36),
        new THREE.MeshBasicMaterial({
            color: thermalColor,
            transparent: true,
            opacity: Math.min(0.46, 0.12 + (block.heat_storage_wm2 || 120) / 650),
            depthWrite: false,
            blending: THREE.AdditiveBlending,
        }),
    );
    storageDisk.rotation.x = -Math.PI / 2;
    storageDisk.position.set(-2.55, 0.265, -2.55);
    storageDisk.userData.block = block;
    thermalMeshes.push(storageDisk);
    group.add(storageDisk);

    const label = makeThermalLabel(block);
    label.position.set(2.35, 1.05, 2.55);
    label.userData.block = block;
    thermalLabels.push(label);
    group.add(label);
}

function createPINNLayer(group, block) {
    const color = pinnColor(block);
    const halo = new THREE.Mesh(
        new THREE.RingGeometry(3.42, 3.82, 80),
        new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: pinnOpacity(block),
            side: THREE.DoubleSide,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
        }),
    );
    halo.rotation.x = -Math.PI / 2;
    halo.position.set(0, 0.305, 0);
    halo.userData.block = block;
    pinnMeshes.push(halo);
    group.add(halo);

    const responseDisk = new THREE.Mesh(
        new THREE.CircleGeometry(0.72, 36),
        new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: Math.min(0.5, 0.18 + Math.abs(block.pinn_delta_temp_c || 0) * 0.08),
            depthWrite: false,
            blending: THREE.AdditiveBlending,
        }),
    );
    responseDisk.rotation.x = -Math.PI / 2;
    responseDisk.position.set(2.65, 0.315, -2.45);
    responseDisk.userData.block = block;
    pinnMeshes.push(responseDisk);
    group.add(responseDisk);

    const label = makePINNLabel(block);
    label.position.set(-2.35, 1.08, -2.55);
    label.userData.block = block;
    pinnLabels.push(label);
    group.add(label);
}

function createSunlightPatch(group, block) {
    const patch = new THREE.Mesh(
        new THREE.PlaneGeometry(6.65, 6.65),
        new THREE.MeshBasicMaterial({
            color: sunlightColor(block.solar_radiation_wm2),
            transparent: true,
            opacity: sunlightOpacity(block),
            depthWrite: false,
            blending: THREE.AdditiveBlending,
        }),
    );
    patch.rotation.x = -Math.PI / 2;
    patch.position.set(0, 0.205, 0);
    patch.userData.block = block;
    sunlightMeshes.push(patch);
    group.add(patch);

    const hotspot = new THREE.Mesh(
        new THREE.CircleGeometry(2.1, 40),
        new THREE.MeshBasicMaterial({
            color: 0xfef08a,
            transparent: true,
            opacity: Math.min(0.38, sunlightOpacity(block) * 0.9),
            depthWrite: false,
            blending: THREE.AdditiveBlending,
        }),
    );
    hotspot.rotation.x = -Math.PI / 2;
    hotspot.position.set(0.8, 0.215, -0.7);
    hotspot.userData.block = block;
    sunlightMeshes.push(hotspot);
    group.add(hotspot);
}

function createSolarBeam(group, block, tallest) {
    const length = worldSunBeamLength(block, tallest);
    const beam = new THREE.Mesh(
        new THREE.TubeGeometry(makeSunBeamCurve(length), 28, 0.16, 18, false),
        new THREE.MeshBasicMaterial({
            color: 0xfacc15,
            transparent: true,
            opacity: Math.min(0.25, 0.08 + (block.solar_radiation_wm2 || 450) / 3800),
            side: THREE.DoubleSide,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
        }),
    );
    const offset = sunlitGroundOffset(0.75);
    setVectorPosition(beam, new THREE.Vector3(offset.x, tallest + 1.7, offset.z));
    beam.userData.block = block;
    sunBeamMeshes.push(beam);
    group.add(beam);

    const beamWide = new THREE.Mesh(
        new THREE.TubeGeometry(makeSunBeamCurve(length * 0.9, 0.35), 28, 0.34, 18, false),
        new THREE.MeshBasicMaterial({
            color: 0xfef08a,
            transparent: true,
            opacity: Math.min(0.13, 0.04 + (block.solar_radiation_wm2 || 450) / 7200),
            side: THREE.DoubleSide,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
        }),
    );
    setVectorPosition(beamWide, new THREE.Vector3(offset.x, tallest + 1.45, offset.z));
    beamWide.userData.block = block;
    sunBeamMeshes.push(beamWide);
    group.add(beamWide);
}

function createBlockShadow(group, block, tallest) {
    const shadowLength = Math.max(2.8, (block.shadow_length_m || tallest * 3.2) * 0.16);
    const shadow = new THREE.Mesh(
        new THREE.PlaneGeometry(shadowLength, 2.35),
        new THREE.MeshBasicMaterial({
            color: 0x020617,
            transparent: true,
            opacity: Math.min(0.42, 0.12 + (block.shadow_coverage_pct || 20) / 160),
            depthWrite: false,
        }),
    );
    orientFlatToGroundDirection(shadow, SHADOW_DIRECTION);
    const offset = sunGroundOffset(1.35 + shadowLength * 0.35);
    shadow.position.set(offset.x, 0.23, offset.z);
    shadow.userData.block = block;
    shadowMeshes.push(shadow);
    group.add(shadow);
}

function createZoneMarker(group, block, tallest) {
    const ring = new THREE.Mesh(
        new THREE.TorusGeometry(3.85, 0.045, 8, 80),
        new THREE.MeshBasicMaterial({ color: ZONE_COLORS[block.zone_class] }),
    );
    ring.rotation.x = Math.PI / 2;
    ring.position.y = tallest + 0.24;
    group.add(ring);
}

function createWindFlow(group, block) {
    const bearing = THREE.MathUtils.degToRad(block.corridor_bearing || 85);
    const dx = Math.sin(bearing);
    const dz = Math.cos(bearing);
    const crossX = Math.sin(bearing + Math.PI / 2);
    const crossZ = Math.cos(bearing + Math.PI / 2);
    const speed = block.wind_speed || 1.5;
    const flowCount = block.zone_class === "WIND_CORRIDOR_CRITICAL" ? 3 : 2;

    for (let i = 0; i < flowCount; i++) {
        const offset = (i - (flowCount - 1) / 2) * 0.78;
        const length = 4.6 + speed * 0.82;
        const sway = block.zone_class === "WIND_CORRIDOR_CRITICAL" ? 0.35 : 0.18;
        const points = [
            new THREE.Vector3(-dx * length * 0.5 + crossX * offset, 0.72 + i * 0.08, -dz * length * 0.5 + crossZ * offset),
            new THREE.Vector3(crossX * (offset + sway), 0.92 + speed * 0.04, crossZ * (offset + sway)),
            new THREE.Vector3(dx * length * 0.5 + crossX * offset, 0.72 + i * 0.08, dz * length * 0.5 + crossZ * offset),
        ];
        const curve = new THREE.CatmullRomCurve3(points);
        const tube = new THREE.Mesh(
            new THREE.TubeGeometry(curve, 36, 0.055 + speed * 0.012, 10, false),
            new THREE.MeshBasicMaterial({
                color: block.zone_class === "WIND_CORRIDOR_CRITICAL" ? 0x7dd3fc : 0x60a5fa,
                transparent: true,
                opacity: Math.min(0.64, 0.22 + speed * 0.08),
                depthWrite: false,
                blending: THREE.AdditiveBlending,
            }),
        );
        tube.visible = speed >= 1.7 || block.zone_class === "WIND_CORRIDOR_CRITICAL";
        tube.userData.block = block;
        tube.userData.baseOpacity = tube.material.opacity;
        tube.userData.phase = i * 0.9 + block.max_height_m * 0.03;
        windFlowMeshes.push(tube);
        group.add(tube);

        const glow = new THREE.Mesh(
            new THREE.TubeGeometry(curve, 36, 0.16 + speed * 0.018, 10, false),
            new THREE.MeshBasicMaterial({
                color: 0x38bdf8,
                transparent: true,
                opacity: Math.min(0.18, 0.04 + speed * 0.025),
                depthWrite: false,
                blending: THREE.AdditiveBlending,
            }),
        );
        glow.visible = tube.visible;
        glow.userData.block = block;
        glow.userData.baseOpacity = glow.material.opacity;
        glow.userData.phase = tube.userData.phase + 0.4;
        windFlowMeshes.push(glow);
        group.add(glow);
    }
}

function refreshSimulationLayers() {
    sunlightMeshes.forEach((patch) => {
        const block = patch.userData.block;
        patch.material.color.copy(sunlightColor(block.solar_radiation_wm2));
        patch.material.opacity = patch.geometry.type === "CircleGeometry"
            ? Math.min(0.42, sunlightOpacity(block) * 0.95)
            : sunlightOpacity(block);
    });
    shadowMeshes.forEach((shadow) => {
        const block = shadow.userData.block;
        shadow.material.opacity = Math.min(0.44, 0.12 + (block.shadow_coverage_pct || 20) / 155);
    });
    sunBeamMeshes.forEach((beam) => {
        const block = beam.userData.block;
        beam.material.opacity = Math.min(0.28, 0.08 + (block.solar_radiation_wm2 || 450) / 3600);
    });
    windFlowMeshes.forEach((flow) => {
        const block = flow.userData.block;
        const speed = block.wind_speed || 1.5;
        flow.visible = speed >= 1.7 || block.zone_class === "WIND_CORRIDOR_CRITICAL";
        flow.material.opacity = Math.min(flow.userData.baseOpacity || 0.45, 0.18 + speed * 0.08);
    });
    thermalMeshes.forEach((mesh) => {
        const block = mesh.userData.block;
        mesh.material.color.copy(thermalColorForTemp(block.surface_temp_c));
        mesh.material.opacity = mesh.geometry.type === "CircleGeometry"
            ? Math.min(0.5, 0.13 + (block.heat_storage_wm2 || 120) / 640)
            : thermalOpacity(block);
    });
    thermalLabels.forEach((label) => {
        const block = label.userData.block;
        const texture = makeThermalLabelTexture(block);
        label.material.map.dispose();
        label.material.map = texture;
        label.material.needsUpdate = true;
    });
}

function refreshPINNLayers() {
    pinnMeshes.forEach((mesh) => {
        const block = mesh.userData.block;
        mesh.material.color.copy(pinnColor(block));
        mesh.material.opacity = mesh.geometry.type === "CircleGeometry"
            ? Math.min(0.54, 0.18 + Math.abs(block.pinn_delta_temp_c || 0) * 0.08)
            : pinnOpacity(block);
    });
    pinnLabels.forEach((label) => {
        const block = label.userData.block;
        const texture = makePINNLabelTexture(block);
        label.material.map.dispose();
        label.material.map = texture;
        label.material.needsUpdate = true;
        label.visible = Number.isFinite(block.pinn_delta_temp_c);
    });
}

function pinnColor(block) {
    const delta = block.pinn_delta_temp_c || 0;
    const t = Math.min(Math.max((delta + 1.5) / 6.0, 0), 1);
    return new THREE.Color(0x22d3ee).lerp(new THREE.Color(0xd946ef), t);
}

function pinnOpacity(block) {
    const confidence = block.pinn_confidence || 0.45;
    const delta = Math.abs(block.pinn_delta_temp_c || 0);
    return Math.min(0.66, 0.16 + confidence * 0.3 + delta * 0.035);
}

function thermalColorForTemp(tempC = 36) {
    const t = Math.min(Math.max((tempC - 32) / 12, 0), 1);
    return new THREE.Color(0xf59e0b).lerp(new THREE.Color(0xef4444), t);
}

function thermalOpacity(block) {
    const temp = block.surface_temp_c || 36;
    return Math.min(0.72, Math.max(0.22, (temp - 31) / 16));
}

function makeThermalLabel(block) {
    const texture = makeThermalLabelTexture(block);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        depthWrite: false,
    }));
    sprite.scale.set(2.6, 0.9, 1);
    return sprite;
}

function makeThermalLabelTexture(block) {
    const labelCanvas = document.createElement("canvas");
    labelCanvas.width = 220;
    labelCanvas.height = 80;
    const ctx = labelCanvas.getContext("2d");
    ctx.fillStyle = "rgba(15, 23, 42, 0.78)";
    roundRect(ctx, 10, 10, 200, 54, 10);
    ctx.fill();
    ctx.font = "800 22px Inter, sans-serif";
    ctx.fillStyle = block.surface_temp_c >= 39 ? "#fecaca" : "#fde68a";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`${(block.surface_temp_c || 0).toFixed(1)}C`, 110, 31);
    ctx.font = "600 11px Inter, sans-serif";
    ctx.fillStyle = "#cbd5e1";
    ctx.fillText("surface temp", 110, 51);
    return new THREE.CanvasTexture(labelCanvas);
}

function makePINNLabel(block) {
    const texture = makePINNLabelTexture(block);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        depthWrite: false,
    }));
    sprite.scale.set(2.7, 0.9, 1);
    return sprite;
}

function makePINNLabelTexture(block) {
    const labelCanvas = document.createElement("canvas");
    labelCanvas.width = 230;
    labelCanvas.height = 80;
    const ctx = labelCanvas.getContext("2d");
    ctx.fillStyle = "rgba(30, 27, 75, 0.78)";
    roundRect(ctx, 10, 10, 210, 54, 10);
    ctx.fill();
    ctx.font = "800 18px Inter, sans-serif";
    ctx.fillStyle = "#f5d0fe";
    ctx.textAlign = "center";
    const delta = Number(block.pinn_delta_temp_c ?? 0);
    const sign = delta >= 0 ? "+" : "";
    ctx.fillText(`PINN ${sign}${delta.toFixed(1)}C`, 115, 34);
    ctx.font = "700 11px Inter, sans-serif";
    ctx.fillStyle = "#c4b5fd";
    const confidence = Math.round((block.pinn_confidence || 0) * 100);
    ctx.fillText(`${confidence}% confidence`, 115, 52);
    return new THREE.CanvasTexture(labelCanvas);
}

function sunlightColor(radiation) {
    const t = Math.min(Math.max((radiation - 250) / 600, 0), 1);
    return new THREE.Color(0xfef08a).lerp(new THREE.Color(0xfb923c), t);
}

function sunlightOpacity(block) {
    const radiation = block.solar_radiation_wm2 || 450;
    const shadow = (block.shadow_coverage_pct || 20) / 100;
    return Math.min(0.68, Math.max(0.26, (radiation / 780) * (1 - shadow * 0.35)));
}

function makeRadialTexture(inner, outer) {
    const textureCanvas = document.createElement("canvas");
    textureCanvas.width = 128;
    textureCanvas.height = 128;
    const ctx = textureCanvas.getContext("2d");
    const gradient = ctx.createRadialGradient(64, 64, 4, 64, 64, 62);
    gradient.addColorStop(0, inner);
    gradient.addColorStop(0.45, outer);
    gradient.addColorStop(1, "rgba(249, 115, 22, 0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 128, 128);
    return new THREE.CanvasTexture(textureCanvas);
}

function createSensorMast(group, block, tallest) {
    if (block.uhi_intensity < 3.5 && block.zone_class !== "WIND_CORRIDOR_CRITICAL") return;
    const mast = new THREE.Mesh(
        new THREE.CylinderGeometry(0.035, 0.045, 1.35, 8),
        new THREE.MeshStandardMaterial({ color: 0xcbd5e1, metalness: 0.55, roughness: 0.35 }),
    );
    mast.position.set(2.8, tallest + 0.85, -2.7);
    group.add(mast);

    const sensor = new THREE.Mesh(
        new THREE.SphereGeometry(0.16, 12, 8),
        new THREE.MeshBasicMaterial({ color: 0x38bdf8 }),
    );
    sensor.position.set(2.8, tallest + 1.6, -2.7);
    group.add(sensor);
}

function heatColor(uhi) {
    const t = Math.min(Math.max(uhi / 5, 0), 1);
    const cool = new THREE.Color(0x06b6d4);
    const warm = new THREE.Color(0xf59e0b);
    const hot = new THREE.Color(0xef4444);
    return t < 0.5 ? cool.lerp(warm, t * 2) : warm.lerp(hot, (t - 0.5) * 2);
}

function makeLabel(text) {
    const labelCanvas = document.createElement("canvas");
    labelCanvas.width = 180;
    labelCanvas.height = 64;
    const ctx = labelCanvas.getContext("2d");
    ctx.fillStyle = "rgba(10, 14, 26, 0.72)";
    roundRect(ctx, 16, 10, 148, 40, 10);
    ctx.fill();
    ctx.font = "700 22px Inter, sans-serif";
    ctx.fillStyle = "#f8fafc";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, 90, 31);

    const texture = new THREE.CanvasTexture(labelCanvas);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true }));
    sprite.scale.set(4.2, 1.5, 1);
    return sprite;
}

function roundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + width, y, x + width, y + height, radius);
    ctx.arcTo(x + width, y + height, x, y + height, radius);
    ctx.arcTo(x, y + height, x, y, radius);
    ctx.arcTo(x, y, x + width, y, radius);
    ctx.closePath();
}

function onCanvasClick(event) {
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(blockMeshes, false);
    const hit = hits.find((item) => item.object.userData.isHitArea) || hits[0];
    if (!hit) return;

    selectBlock(hit.object.userData.block, hit.object);
}

function selectBlock(block, mesh) {
    if (selectedBlockGroup) {
        selectedBlockGroup.scale.set(1, 1, 1);
    }
    if (selectedHighlight) {
        selectedHighlight.parent?.remove(selectedHighlight);
        selectedHighlight.geometry.dispose();
        selectedHighlight.material.dispose();
        selectedHighlight = null;
    }
    selectedMesh = mesh;
    selectedBlockId = block.id;
    selectedBlockGroup = mesh.userData.group || mesh.parent;
    selectedBlockGroup.scale.set(1.04, 1.04, 1.04);

    selectedHighlight = new THREE.Mesh(
        new THREE.RingGeometry(4.4, 4.75, 80),
        new THREE.MeshBasicMaterial({
            color: 0xf8fafc,
            transparent: true,
            opacity: 0.72,
            side: THREE.DoubleSide,
        }),
    );
    selectedHighlight.rotation.x = -Math.PI / 2;
    selectedHighlight.position.y = 0.19;
    selectedBlockGroup.add(selectedHighlight);

    const zoneClassName = zoneClassCss(block.zone_class);
    const pinnDelta = Number(block.pinn_delta_temp_c ?? 0);
    const pinnSurface = Number(block.pinn_surface_temp_c ?? block.surface_temp_c ?? 0);
    const pinnConfidence = Math.round(Number(block.pinn_confidence ?? 0) * 100);
    const heightSensitivity = Number(block.pinn_height_sensitivity ?? 0);
    const greenSensitivity = Number(block.pinn_green_sensitivity ?? 0);
    const oodText = block.pinn_ood_warning ? `Check ${block.pinn_ood_features?.join(", ") || "inputs"}` : "Normal";
    const complianceText = block.compliance_status === "non_compliant" ? "Non-compliant" : "Compliant";
    const complianceClass = block.compliance_status === "non_compliant" ? "wind" : "density";
    details.innerHTML = `
        <div class="detail-row"><span class="detail-label">Block ID</span><span class="detail-value">${block.id}</span></div>
        <div class="detail-row"><span class="detail-label">Zone Class</span><span class="zone-pill ${zoneClassName}">${ZONE_LABELS[block.zone_class]}</span></div>
        <div class="detail-row"><span class="detail-label">Compliance</span><span class="zone-pill ${complianceClass}">${complianceText}</span></div>
        ${block.compliance_note ? `<div class="detail-row"><span class="detail-label">Advisory</span><span class="detail-value">${block.compliance_note}</span></div>` : ""}
        <div class="detail-row"><span class="detail-label">Max Height</span><span class="detail-value">${block.max_height_m}m</span></div>
        <div class="detail-row"><span class="detail-label">UHI Intensity</span><span class="detail-value">+${block.uhi_intensity.toFixed(2)}C</span></div>
        <div class="detail-row"><span class="detail-label">SVF</span><span class="detail-value">${block.svf.toFixed(2)}</span></div>
        <div class="detail-row"><span class="detail-label">Plan Area</span><span class="detail-value">${block.lambda_p.toFixed(2)}</span></div>
        <div class="detail-row"><span class="detail-label">H/W Ratio</span><span class="detail-value">${block.hw_ratio.toFixed(1)}</span></div>
        <div class="detail-row"><span class="detail-label">Albedo</span><span class="detail-value">${block.albedo.toFixed(2)}</span></div>
        <div class="detail-row"><span class="detail-label">Green Cover</span><span class="detail-value">${block.green_cover}%</span></div>
        <div class="detail-row"><span class="detail-label">Wind Speed</span><span class="detail-value">${block.wind_speed.toFixed(1)} m/s</span></div>
        <div class="detail-row"><span class="detail-label">Wind Deflection</span><span class="detail-value">${(block.wind_deflection_deg || 0).toFixed(1)} deg</span></div>
        <div class="detail-row"><span class="detail-label">Ventilation</span><span class="detail-value">${Math.round(block.ventilation_score * 100)}%</span></div>
        <div class="detail-row"><span class="detail-label">Solar Access</span><span class="detail-value">${block.solar_access_hours.toFixed(1)} h</span></div>
        <div class="detail-row"><span class="detail-label">Shadow Cover</span><span class="detail-value">${block.shadow_coverage_pct.toFixed(0)}%</span></div>
        <div class="detail-row"><span class="detail-label">Solar Radiation</span><span class="detail-value">${block.solar_radiation_wm2.toFixed(0)} W/m2</span></div>
        <div class="detail-row"><span class="detail-label">Surface Temp</span><span class="detail-value">${block.surface_temp_c.toFixed(1)} C</span></div>
        <div class="detail-row"><span class="detail-label">Air Temp</span><span class="detail-value">${block.air_temp_c.toFixed(1)} C</span></div>
        <div class="detail-row"><span class="detail-label">Thermal UHI</span><span class="detail-value">+${(block.thermal_uhi_intensity_c || block.uhi_intensity).toFixed(2)} C</span></div>
        <div class="detail-row"><span class="detail-label">Heat Storage</span><span class="detail-value">${block.heat_storage_wm2.toFixed(0)} W/m2</span></div>
        <div class="detail-row"><span class="detail-label">Thermal Risk</span><span class="detail-value">${block.thermal_risk}</span></div>
        <div class="detail-row"><span class="detail-label">PINN Temp Delta</span><span class="detail-value">${pinnDelta >= 0 ? "+" : ""}${pinnDelta.toFixed(2)} C</span></div>
        <div class="detail-row"><span class="detail-label">PINN Surface</span><span class="detail-value">${pinnSurface.toFixed(1)} C</span></div>
        <div class="detail-row"><span class="detail-label">PINN Confidence</span><span class="detail-value">${pinnConfidence}%</span></div>
        <div class="detail-row"><span class="detail-label">Height Sens.</span><span class="detail-value">${heightSensitivity >= 0 ? "+" : ""}${heightSensitivity.toFixed(2)} C / 10m</span></div>
        <div class="detail-row"><span class="detail-label">Green Sens.</span><span class="detail-value">${greenSensitivity.toFixed(2)} C / 10%</span></div>
        <div class="detail-row"><span class="detail-label">PINN OOD</span><span class="detail-value">${oodText}</span></div>
        <div class="detail-row"><span class="detail-label">Heat Risk</span><span class="detail-value">${block.combined_risk || 'actual data pending'}</span></div>
        <div class="detail-row"><span class="detail-label">3D Meaning</span><span class="detail-value">${block.max_height_m}m height, +${block.uhi_intensity.toFixed(2)}C heat</span></div>
    `;
    syncEditorControls(block);
}

function selectBlockById(blockId) {
    const mesh = blockMeshes.find((item) => item.userData.block?.id === blockId);
    if (!mesh) return;
    selectBlock(mesh.userData.block, mesh);
}

function zoneClassCss(zoneClass) {
    return {
        WIND_CORRIDOR_CRITICAL: "wind",
        THERMAL_REMEDIATION: "thermal",
        DENSITY_ADAPTIVE: "density",
        BASELINE_UNCHANGED: "baseline",
    }[zoneClass];
}

function resize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    camera.aspect = rect.width / rect.height;
    camera.updateProjectionMatrix();
    renderer.setSize(rect.width, rect.height, false);
}

function animate() {
    requestAnimationFrame(animate);
    const t = performance.now() * 0.001;
    animatedPlumes.forEach((plume, index) => {
        plume.rotation.y += plume.userData.spin || 0.002;
        plume.material.opacity = 0.12 + Math.sin(t * 1.4 + index) * 0.025;
    });
    sunBeamMeshes.forEach((beam, index) => {
        beam.material.opacity += Math.sin(t * 1.8 + index) * 0.0008;
    });
    windFlowMeshes.forEach((flow) => {
        const base = flow.userData.baseOpacity || 0.35;
        flow.material.opacity = Math.max(0.06, base + Math.sin(t * 2.6 + flow.userData.phase) * 0.08);
    });
    if (selectedHighlight) {
        selectedHighlight.material.opacity = 0.46 + Math.sin(t * 3.2) * 0.16;
    }
    controls.update();
    renderer.render(scene, camera);
}
