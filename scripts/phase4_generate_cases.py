#!/usr/bin/env python3
import sys, json, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.config import DATA_DIR

CFD_GEOMETRY = DATA_DIR / "cfd_geometry"
CFD_CONDITIONS = DATA_DIR / "cfd_conditions"
CFD_CASES = DATA_DIR / "cfd_cases"

TARGET_ARCHETYPES = [9]

def write_blockMeshDict(sys_dir, nx, ny, nz, bounds):
    content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
\\*---------------------------------------------------------------------------*/
FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}

scale   1;
vertices
(
    ({bounds['x_min']:.3f} {bounds['y_min']:.3f} {bounds['z_min']:.3f})
    ({bounds['x_max']:.3f} {bounds['y_min']:.3f} {bounds['z_min']:.3f})
    ({bounds['x_max']:.3f} {bounds['y_max']:.3f} {bounds['z_min']:.3f})
    ({bounds['x_min']:.3f} {bounds['y_max']:.3f} {bounds['z_min']:.3f})
    ({bounds['x_min']:.3f} {bounds['y_min']:.3f} {bounds['z_max']:.3f})
    ({bounds['x_max']:.3f} {bounds['y_min']:.3f} {bounds['z_max']:.3f})
    ({bounds['x_max']:.3f} {bounds['y_max']:.3f} {bounds['z_max']:.3f})
    ({bounds['x_min']:.3f} {bounds['y_max']:.3f} {bounds['z_max']:.3f})
);

blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary
(
    inlet {{ type patch; faces ((0 4 7 3)); }}
    outlet {{ type patch; faces ((1 2 6 5)); }}
    sides {{ type symmetry; faces ((0 1 5 4) (3 7 6 2)); }}
    ground {{ type wall; faces ((0 3 2 1)); }}
    top {{ type symmetry; faces ((4 5 6 7)); }}
);
mergePatchPairs ();
"""
    (sys_dir / "blockMeshDict").write_text(content)

def write_snappyHexMeshDict(sys_dir, bounds):
    content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
\\*---------------------------------------------------------------------------*/
FoamFile {{ version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }}

castellatedMesh true;
snap            false;
addLayers       false;

geometry
{{
    buildings.stl {{ type triSurfaceMesh; name buildings; }}
    terrain.stl {{ type triSurfaceMesh; name terrain; }}
    trees.stl {{ type triSurfaceMesh; name trees; }}
    canopies.stl {{ type triSurfaceMesh; name canopies; }}
}}

castellatedMeshControls
{{
    maxLocalCells 1000000;
    maxGlobalCells 20000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;

    features
    (
        {{ file "buildings.eMesh"; level 4; }}
        {{ file "terrain.eMesh"; level 2; }}
    );

    refinementSurfaces
    {{
        buildings {{ level (0 0); patchInfo {{ type wall; }} }}
        terrain {{ level (0 0); patchInfo {{ type wall; }} }}
        trees {{ level (0 0); patchInfo {{ type wall; }} }}
    }}

    resolveFeatureAngle 30;

    refinementRegions
    {{
        canopies {{ mode inside; levels ((1.0 3)); }}
    }}

    locationInMesh ({bounds['x_min'] + 5.0:.1f} {bounds['y_min'] + 5.0:.1f} {bounds['z_max'] - 5.0:.1f});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes true; layers {{ buildings {{ nSurfaceLayers 2; }} }}
    expansionRatio 1.2; finalLayerThickness 0.5; minThickness 0.1; nGrow 0;
    featureAngle 60; slipFeatureAngle 30; nRelaxIter 5; nSmoothSurfaceNormals 1;
    nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3; minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0; nLayerIter 50;
}}

meshQualityControls
{{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
    maxConcave 80; minVol 1e-13; minTetQuality 1e-30; minArea -1;
    minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.05; minVolRatio 0.01;
    minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
}}

writeFlags ( scalarLevels layerSets layerFields );
mergeTolerance 1e-6;
"""
    (sys_dir / "snappyHexMeshDict").write_text(content)

def write_surfaceFeatureExtractDict(sys_dir):
    content = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
\\*---------------------------------------------------------------------------*/
FoamFile { version 2.0; format ascii; class dictionary; object surfaceFeatureExtractDict; }

buildings.stl { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj yes; }
terrain.stl { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj yes; }
"""
    (sys_dir / "surfaceFeatureExtractDict").write_text(content)

def write_controlDict(sys_dir):
    content = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
\\*---------------------------------------------------------------------------*/
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }

application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         3;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
    (sys_dir / "controlDict").write_text(content)

def write_fv_files(sys_dir):
    fvs = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
\\*---------------------------------------------------------------------------*/
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }

ddtSchemes { default steadyState; }
gradSchemes { default cellMDLimited Gauss linear 0.5; grad(U) cellMDLimited Gauss linear 0.5; grad(p) Gauss linear; }
divSchemes { default none; div(phi,U) bounded Gauss upwind; div(phi,k) bounded Gauss upwind; div(phi,omega) bounded Gauss upwind; div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear limited 0.33; }
interpolationSchemes { default linear; }
snGradSchemes { default limited 0.33; }
wallDist { method meshWave; }
"""
    fvs_sol = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
\\*---------------------------------------------------------------------------*/
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }

solvers {
    p { solver GAMG; tolerance 1e-6; relTol 0.1; smoother GaussSeidel; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-5; relTol 0.1; }
    k { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-5; relTol 0.1; }
    omega { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-5; relTol 0.1; }
}
SIMPLE {
    nNonOrthogonalCorrectors 1; pRefCell 0; pRefValue 0;
    residualControl { p 1e-4; U 1e-5; k 1e-5; omega 1e-5; }
}
relaxationFactors { fields { p 0.3; } equations { U 0.7; k 0.7; omega 0.7; } }
"""
    (sys_dir / "fvSchemes").write_text(fvs)
    (sys_dir / "fvSolution").write_text(fvs_sol)

def write_constant_files(const_dir):
    tp = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
\\*---------------------------------------------------------------------------*/
FoamFile { version 2.0; format ascii; class dictionary; object transportProperties; }
transportModel Newtonian; nu [0 2 -1 0 0 0 0] 1.5e-05;
"""
    turb = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
\\*---------------------------------------------------------------------------*/
FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }
simulationType RAS; RAS { RASModel kOmegaSST; turbulence on; printCoeffs on; }
"""
    (const_dir / "transportProperties").write_text(tp)
    (const_dir / "turbulenceProperties").write_text(turb)

def write_topoSetDict(sys_dir, geom_dir):
    if not (geom_dir / "canopies.stl").exists():
        return
        
    content = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
\\*---------------------------------------------------------------------------*/
FoamFile { version 2.0; format ascii; class dictionary; object topoSetDict; }

actions
(
    {
        name    canopies;
        type    cellSet;
        action  new;
        source  surfaceToCell;
        file    "constant/triSurface/canopies.stl";
        outsidePoints ((0 0 5000));
        includeCut true;
        includeInside true;
        includeOutside false;
        nearDistance -1;
        curvature 100;
    }
    {
        name    canopies;
        type    cellZoneSet;
        action  new;
        source  setToCellZone;
        set     canopies;
    }
);
"""
    (sys_dir / "topoSetDict").write_text(content)

def write_fvOptions(sys_dir, meta, geom_dir):
    if not (geom_dir / "canopies.stl").exists():
        return
        
    content = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
\\*---------------------------------------------------------------------------*/
FoamFile { version 2.0; format ascii; class dictionary; object fvOptions; }

porosity
{
    type            explicitPorositySource;
    active          yes;

    explicitPorositySourceCoeffs
    {
        selectionMode   cellZone;
        cellZone        canopies;
        
        type            DarcyForchheimer;

        d   (0 0 0);
        f   (5 5 5); // Example Forchheimer drag derived from LAD
        coordinateSystem
        {
            type    cartesian;
            origin  (0 0 0);
            coordinateRotation
            {
                type    axesRotation;
                e1      (1 0 0);
                e2      (0 1 0);
            }
        }
    }
}
"""
    (sys_dir / "fvOptions").write_text(content)

def write_0_files(zero_dir, u_in, wind_dir, z0):
    import math
    u_x = u_in * math.cos(math.radians(wind_dir))
    u_y = u_in * math.sin(math.radians(wind_dir))
    
    flow_dir = f"({math.cos(math.radians(wind_dir)):.5f} {math.sin(math.radians(wind_dir)):.5f} 0)"
    
    u = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
\\*---------------------------------------------------------------------------*/
FoamFile {{ version 2.0; format ascii; class volVectorField; object U; }}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({u_x:.2f} {u_y:.2f} 0);
boundaryField
{{
    inlet 
    {{ 
        type            atmBoundaryLayerInletVelocity;
        flowDir         {flow_dir};
        zDir            (0 0 1);
        Uref            {u_in:.2f};
        Zref            10.0;
        z0              uniform {z0:.3f};
        zGround         uniform 0.0;
        d               uniform 0.0;
    }}
    outlet {{ type zeroGradient; }}
    sides {{ type symmetry; }}
    ground {{ type noSlip; }}
    top {{ type symmetry; }}
    buildings {{ type noSlip; }}
    terrain {{ type noSlip; }}
    trees {{ type noSlip; }}
}}
"""
    p = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
\\*---------------------------------------------------------------------------*/
FoamFile {{ version 2.0; format ascii; class volScalarField; object p; }}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{{
    inlet {{ type zeroGradient; }}
    outlet {{ type fixedValue; value uniform 0; }}
    sides {{ type symmetry; }}
    ground {{ type zeroGradient; }}
    top {{ type symmetry; }}
    buildings {{ type zeroGradient; }}
    terrain {{ type zeroGradient; }}
    trees {{ type zeroGradient; }}
}}
"""
    k = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
\\*---------------------------------------------------------------------------*/
FoamFile {{ version 2.0; format ascii; class volScalarField; object k; }}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0.1;
boundaryField
{{
    inlet 
    {{ 
        type            atmBoundaryLayerInletK;
        flowDir         {flow_dir};
        zDir            (0 0 1);
        Uref            {u_in:.2f};
        Zref            10.0;
        z0              uniform {z0:.3f};
        zGround         uniform 0.0;
        d               uniform 0.0;
    }}
    outlet {{ type zeroGradient; }}
    sides {{ type symmetry; }}
    ground {{ type kqRWallFunction; value uniform 0.1; }}
    top {{ type symmetry; }}
    buildings {{ type kqRWallFunction; value uniform 0.1; }}
    terrain {{ type kqRWallFunction; value uniform 0.1; }}
    trees {{ type kqRWallFunction; value uniform 0.1; }}
}}
"""
    omega = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
\\*---------------------------------------------------------------------------*/
FoamFile {{ version 2.0; format ascii; class volScalarField; object omega; }}
dimensions      [0 0 -1 0 0 0 0];
internalField   uniform 10.0;
boundaryField
{{
    inlet 
    {{ 
        type            atmBoundaryLayerInletOmega;
        flowDir         {flow_dir};
        zDir            (0 0 1);
        Uref            {u_in:.2f};
        Zref            10.0;
        z0              uniform {z0:.3f};
        zGround         uniform 0.0;
        d               uniform 0.0;
    }}
    outlet {{ type zeroGradient; }}
    sides {{ type symmetry; }}
    ground {{ type omegaWallFunction; value uniform 10.0; }}
    top {{ type symmetry; }}
    buildings {{ type omegaWallFunction; value uniform 10.0; }}
    terrain {{ type omegaWallFunction; value uniform 10.0; }}
    trees {{ type omegaWallFunction; value uniform 10.0; }}
}}
"""
    nut = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
\\*---------------------------------------------------------------------------*/
FoamFile {{ version 2.0; format ascii; class volScalarField; object nut; }}
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;
boundaryField
{{
    inlet {{ type calculated; value uniform 0; }}
    outlet {{ type calculated; value uniform 0; }}
    sides {{ type symmetry; }}
    ground {{ type nutkWallFunction; value uniform 0; }}
    top {{ type symmetry; }}
    buildings {{ type nutkWallFunction; value uniform 0; }}
    terrain {{ type nutkWallFunction; value uniform 0; }}
    trees {{ type nutkWallFunction; value uniform 0; }}
}}
"""
    (zero_dir / "U").write_text(u)
    (zero_dir / "p").write_text(p)
    (zero_dir / "k").write_text(k)
    (zero_dir / "omega").write_text(omega)
    (zero_dir / "nut").write_text(nut)


def generate_case(cid, scenario_name, scenario_data):
    geom_dir = CFD_GEOMETRY / f"archetype_{cid:02d}"
    if not geom_dir.exists():
        print(f"Skipping Archetype {cid}: geometry not found.")
        return
        
    case_dir = CFD_CASES / f"archetype_{cid:02d}" / scenario_name
    case_dir.mkdir(parents=True, exist_ok=True)
    
    sys_dir = case_dir / "system"
    const_dir = case_dir / "constant"
    tri_dir = const_dir / "triSurface"
    zero_dir = case_dir / "0"
    
    sys_dir.mkdir(parents=True, exist_ok=True)
    tri_dir.mkdir(parents=True, exist_ok=True)
    zero_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy STLs
    for stl in ["buildings.stl", "terrain.stl", "trees.stl", "canopies.stl", "domain/domain.stl"]:
        src = geom_dir / stl
        if src.exists():
            shutil.copy(src, tri_dir / Path(stl).name)
            
    with open(geom_dir / "geometry_meta.json") as f:
        meta = json.load(f)
        
    bds = meta["domain_bounds"]
    nx = max(2, int(round((bds["x_max"] - bds["x_min"]) / 100.0)))
    ny = max(2, int(round((bds["y_max"] - bds["y_min"]) / 100.0)))
    nz = max(2, int(round((bds["z_max"] - bds["z_min"]) / 100.0)))
    
    write_blockMeshDict(sys_dir, nx, ny, nz, bds)
    write_snappyHexMeshDict(sys_dir, bds)
    write_surfaceFeatureExtractDict(sys_dir)
    write_topoSetDict(sys_dir, geom_dir)
    write_controlDict(sys_dir)
    write_fv_files(sys_dir)
    write_constant_files(const_dir)
    
    u_in = scenario_data.get("wind_speed_ms", 5.0)
    wind_dir = scenario_data.get("wind_direction_deg", 0.0)
    mean_h = meta.get("mean_building_height", 10.0)
    z0 = max(0.01, 0.1 * mean_h) # Roughness approximation
    write_0_files(zero_dir, u_in, wind_dir, z0)
    # write_fvOptions(sys_dir, meta, geom_dir)
    
    print(f"Generated case: Archetype {cid} | Scenario {scenario_name}")

def main():
    CFD_CASES.mkdir(parents=True, exist_ok=True)
    
    for cid in TARGET_ARCHETYPES:
        cond_dir = CFD_CONDITIONS / f"archetype_{cid:02d}"
        if not cond_dir.exists():
            continue
            
        for scenario_file in cond_dir.glob("*.json"):
            if "metadata" in scenario_file.name or "era5" in scenario_file.name:
                continue
            with open(scenario_file) as f:
                scen_data = json.load(f)
            generate_case(cid, scenario_file.stem, scen_data)
            
if __name__ == "__main__":
    main()
