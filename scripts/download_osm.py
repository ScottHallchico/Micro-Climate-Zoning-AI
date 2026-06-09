#!/usr/bin/env python3
import os
import sys
import geopandas as gpd
import osmnx as ox
from pathlib import Path

# Paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.config import RAW_DIR

def main():
    ox.settings.use_cache = True
    ox.settings.log_console = True
    
    out_edges = RAW_DIR / "nyc_roads_edges.parquet"
    out_nodes = RAW_DIR / "nyc_roads_nodes.parquet"
    
    if out_edges.exists() and out_nodes.exists():
        print("OSM road network already cached.")
        return
        
    print("Downloading NYC road network from OpenStreetMap...")
    # Use drive network to capture major streets/canyons
    G = ox.graph_from_place('New York City, New York, USA', network_type='drive')
    
    print("Calculating edge bearings...")
    # Add bearing (orientation) to each edge
    try:
        G = ox.add_edge_bearings(G)
    except AttributeError:
        # Compatibility with newer/older osmnx
        import osmnx.bearing as oxb
        G = oxb.add_edge_bearings(G)
        
    print("Converting to GeoDataFrames...")
    try:
        nodes, edges = ox.graph_to_gdfs(G)
    except AttributeError:
        nodes, edges = ox.utils_graph.graph_to_gdfs(G)
    
    # Clean up lists in columns (parquet doesn't like lists in columns if types are mixed)
    for col in edges.columns:
        if col != 'geometry' and edges[col].apply(lambda x: isinstance(x, list)).any():
            edges[col] = edges[col].astype(str)
            
    for col in nodes.columns:
        if col != 'geometry' and nodes[col].apply(lambda x: isinstance(x, list)).any():
            nodes[col] = nodes[col].astype(str)
            
    print(f"Saving {len(edges)} edges and {len(nodes)} nodes to {RAW_DIR}...")
    edges.to_parquet(out_edges)
    nodes.to_parquet(out_nodes)
    print("Done.")

if __name__ == "__main__":
    main()
