#!/usr/bin/env python3
import sys, json, shutil
from pathlib import Path

def main():
    arch_dir = Path("data/cfd_geometry/archetype_09")
    if not arch_dir.exists():
        print("Error: archetype_09 geometry directory not found.")
        sys.exit(1)
        
    sys_dir = arch_dir / "system"
    const_dir = arch_dir / "constant"
    tri_dir = const_dir / "triSurface"
    
    sys_dir.mkdir(parents=True, exist_ok=True)
    tri_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy STLs
    for stl in ["buildings.stl", "terrain.stl", "trees.stl", "canopies.stl", "domain/domain.stl"]:
        src = arch_dir / stl
        if src.exists():
            shutil.copy(src, tri_dir / src.name)
            
    with open(arch_dir / "geometry_meta.json") as f:
        meta = json.load(f)
        
    bds = meta["domain_bounds"]
    
    x_min, x_max = bds["x_min"], bds["x_max"]
    y_min, y_max = bds["y_min"], bds["y_max"]
    z_min, z_max = bds["z_min"], bds["z_max"]
    
    nx = int(round((x_max - x_min) / 10.0))
    ny = int(round((y_max - y_min) / 10.0))
    nz = int(round((z_max - z_min) / 10.0))
    
    # 1. blockMeshDict
    block_mesh = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale   1;

vertices
(
    ({x_min:.3f} {y_min:.3f} {z_min:.3f})
    ({x_max:.3f} {y_min:.3f} {z_min:.3f})
    ({x_max:.3f} {y_max:.3f} {z_min:.3f})
    ({x_min:.3f} {y_max:.3f} {z_min:.3f})
    ({x_min:.3f} {y_min:.3f} {z_max:.3f})
    ({x_max:.3f} {y_min:.3f} {z_max:.3f})
    ({x_max:.3f} {y_max:.3f} {z_max:.3f})
    ({x_min:.3f} {y_max:.3f} {z_max:.3f})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            (1 2 6 5)
        );
    }}
    sides
    {{
        type patch;
        faces
        (
            (0 1 5 4)
            (3 7 6 2)
        );
    }}
    ground
    {{
        type wall;
        faces
        (
            (0 3 2 1)
        );
    }}
    top
    {{
        type patch;
        faces
        (
            (4 5 6 7)
        );
    }}
);

mergePatchPairs
(
);
"""

    with open(sys_dir / "blockMeshDict", "w") as f:
        f.write(block_mesh)
        
    # 2. snappyHexMeshDict
    shm = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    buildings.stl
    {{
        type triSurfaceMesh;
        name buildings;
    }}
    terrain.stl
    {{
        type triSurfaceMesh;
        name terrain;
    }}
    trees.stl
    {{
        type triSurfaceMesh;
        name trees;
    }}
    canopies.stl
    {{
        type triSurfaceMesh;
        name canopies;
    }}
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
        {{
            file "buildings.eMesh";
            level 4;
        }}
        {{
            file "terrain.eMesh";
            level 2;
        }}
    );

    refinementSurfaces
    {{
        buildings
        {{
            level (4 4);
            patchInfo
            {{
                type wall;
            }}
        }}
        terrain
        {{
            level (2 2);
            patchInfo
            {{
                type wall;
            }}
        }}
        trees
        {{
            level (3 3);
            patchInfo
            {{
                type wall;
            }}
        }}
    }}

    resolveFeatureAngle 30;

    refinementRegions
    {{
        canopies
        {{
            mode inside;
            levels ((1.0 3));
        }}
    }}

    locationInMesh (0.0 0.0 {z_max - 5.0:.1f});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes true;
    layers
    {{
        buildings
        {{
            nSurfaceLayers 2;
        }}
    }}
    expansionRatio 1.2;
    finalLayerThickness 0.5;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    slipFeatureAngle 30;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}

meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-30;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    triangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}

writeFlags
(
    scalarLevels
    layerSets
    layerFields
);

mergeTolerance 1e-6;

// ************************************************************************* //
"""
    with open(sys_dir / "snappyHexMeshDict", "w") as f:
        f.write(shm)
        
    # 3. controlDict
    ctrl = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

"""
    with open(sys_dir / "controlDict", "w") as f:
        f.write(ctrl)
        
    # 4. fvSchemes
    fvs = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         cellMDLimited Gauss linear 0.5;
    grad(U)         cellMDLimited Gauss linear 0.5;
    grad(p)         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss upwind;
    div(phi,k)      bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear limited 0.33;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         limited 0.33;
}
"""
    with open(sys_dir / "fvSchemes", "w") as f:
        f.write(fvs)
        
    # 5. fvSolution
    fvs_sol = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-6;
        relTol          0.1;
        smoother        GaussSeidel;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-5;
        relTol          0.1;
    }

    k
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-5;
        relTol          0.1;
    }

    epsilon
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-5;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;

    residualControl
    {
        p               1e-4;
        U               1e-4;
        k               1e-4;
        epsilon         1e-4;
    }
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
        k               0.7;
        epsilon         0.7;
    }
}
"""
    with open(sys_dir / "fvSolution", "w") as f:
        f.write(fvs_sol)
        
    # 6. transportProperties
    tp = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

transportModel  Newtonian;

nu              [0 2 -1 0 0 0 0] 1.5e-05;

"""
    with open(const_dir / "transportProperties", "w") as f:
        f.write(tp)
        
    # 7. turbulenceProperties
    turb = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2212                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

simulationType RAS;

RAS
{
    RASModel        kEpsilon;
    turbulence      on;
    printCoeffs     on;
}
"""
    with open(const_dir / "turbulenceProperties", "w") as f:
        f.write(turb)
        
    print("OpenFOAM configuration generated successfully for Archetype 09.")

if __name__ == "__main__":
    main()
