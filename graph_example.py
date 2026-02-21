import time
import folium
import requests

import numpy as np

import pickle

from dataloader import *
from omegaconf import OmegaConf
from branca.element import MacroElement, Template
from folium.features import DivIcon

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]


class SmoothZoomFit(MacroElement):
    def __init__(self, south, west, north, east,
                 zoom_snap=0.1, zoom_delta=0.1, wheel_px_per_zoom=400,
                 pad_px=(10, 10)):
        super().__init__()
        self._name = "SmoothZoomFit"

        self.south = float(south)
        self.west  = float(west)
        self.north = float(north)
        self.east  = float(east)

        self.zoom_snap = float(zoom_snap)
        self.zoom_delta = float(zoom_delta)
        self.wheel_px_per_zoom = int(wheel_px_per_zoom)

        self.pad0 = int(pad_px[0])
        self.pad1 = int(pad_px[1])

        self._template = Template(u"""
        {% macro script(this, kwargs) %}
        (function () {
            var map = {{ this._parent.get_name() }};
            if (!map) return;

            map.options.zoomSnap = {{ this.zoom_snap }};
            map.options.zoomDelta = {{ this.zoom_delta }};
            map.options.wheelPxPerZoom = {{ this.wheel_px_per_zoom }};

            if (map.scrollWheelZoom) map.scrollWheelZoom.enable();

            map.fitBounds(
                [[{{ this.south }}, {{ this.west }}], [{{ this.north }}, {{ this.east }}]],
                {padding: [{{ this.pad0 }}, {{ this.pad1 }}]}
            );
        })();
        {% endmacro %}
        """)


def _add_text_label(m, lat, lon, text, dy=-8, font_size=10):
    """
    Draw a permanent text label near a point using a DivIcon.
    dy: vertical pixel offset (negative moves up)
    """
    folium.Marker(
        location=[lat, lon],
        icon=DivIcon(
            icon_size=(250, 16),
            icon_anchor=(0, dy),
            html=f"""
            <div style="
                font-size:{font_size}px;
                color:#111;
                background:rgba(255,255,255,0.85);
                padding:1px 3px;
                border:1px solid rgba(0,0,0,0.25);
                border-radius:3px;
                display:inline-block;
                white-space:nowrap;
            ">{text}</div>
            """
        )
    ).add_to(m)


def visualize_virtual_station_graph_html_detailed(
    ds: "GWaterDataset",
    loc_key,
    out_html="virtual_station_graph.html",
    tiles="CartoDB positron",
    zoom_start=13,
    edge_color="#6a1b9a",
):
    """
    Visualize trimmed graph for ONE virtual station using detailed polylines from:
      ds.data[a]['graph'][b]['nodes']

    Node highlighting:
      - only nodes from ds.build_trim_graph(loc_key) are shown
      - target node: red
      - direct neighbors of target (ds.data[loc_key]['neighbor'].keys()): green
      - other nodes in trimmed component: gray
    Edge drawing:
      - use detailed polyline if available, else straight segment
      - dashed if dir_flag == 0, solid otherwise
    """

    # -------------------------
    # Coordinate helpers
    # -------------------------
    def _as_latlon(pt):
        """pt is a tuple (a,b). Return (lat,lon) robustly."""
        a, b = float(pt[0]), float(pt[1])
        if (-90 <= a <= 90) and (-180 <= b <= 180):
            return a, b
        return b, a

    def _path_to_latlon_list(path_nodes):
        """Convert list of tuple coords -> [[lat,lon], ...]"""
        out = []
        for p in path_nodes:
            lat, lon = _as_latlon(p)
            out.append([lat, lon])
        return out

    # -------------------------
    # Halo marker helper
    # -------------------------
    def _add_halo_circle(m, lat, lon, color, popup_html, r=5):
        folium.CircleMarker(
            location=[lat, lon],
            radius=r + 3,
            color="black",
            weight=0,
            fill=True,
            fill_color="black",
            fill_opacity=0.25,
        ).add_to(m)

        folium.CircleMarker(
            location=[lat, lon],
            radius=r,
            color="black",
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(m)

    # -------------------------
    # 1) Build trimmed graph (your logic)
    # -------------------------
    if loc_key not in ds.data:
        raise KeyError(f"loc_key not found in ds.data: {loc_key}")

    selected_nodes, selected_edges = ds.build_trim_graph(loc_key)

    # direct neighbors for coloring
    neighbor_nodes = set(ds.data[loc_key].get("neighbor", {}).keys())

    # -------------------------
    # 2) Map center/bounds from selected nodes
    # -------------------------
    all_latlon = [_as_latlon(n) for n in selected_nodes]
    if len(all_latlon) == 0:
        raise RuntimeError("Trimmed graph has zero nodes.")

    lats = np.array([p[0] for p in all_latlon], float)
    lons = np.array([p[1] for p in all_latlon], float)

    m = folium.Map(
        location=[float(lats.mean()), float(lons.mean())],
        zoom_start=zoom_start,
        tiles=tiles,
        control_scale=True,
    )
    m.fit_bounds([[float(lats.min()), float(lons.min())],
                  [float(lats.max()), float(lons.max())]])

    # Overlay OSM waterways for context (your existing helper)
    # try:
    #     add_osm_waterways_overlay_from_points(m, lats, lons)
    # except Exception as e:
    #     print("[WARN] add_osm_waterways_overlay_from_points failed:", e)

    # Optional smooth zoom (your helper)
    try:
        add_ultra_smooth_zoom_and_fit(m, lats, lons, zoom_snap=0.1, zoom_delta=0.1, wheel_px_per_zoom=450)
    except Exception:
        pass

    # -------------------------
    # 3) Find detailed polyline path for an edge u->v
    # -------------------------
    def _get_detailed_path(u, v):
        """
        Try to retrieve a polyline from ds.data[u]['graph'][v]['nodes'] (or reverse).
        Returns: list of [lat,lon] or None
        """
        # Try forward
        if u in ds.data and "graph" in ds.data[u]:
            g = ds.data[u]["graph"]
            if isinstance(g, dict) and v in g and isinstance(g[v], dict) and "nodes" in g[v]:
                nodes = g[v]["nodes"]
                if nodes and len(nodes) >= 2:
                    return _path_to_latlon_list(nodes)

        # Try reverse
        if v in ds.data and "graph" in ds.data[v]:
            g = ds.data[v]["graph"]
            if isinstance(g, dict) and u in g and isinstance(g[u], dict) and "nodes" in g[u]:
                nodes = g[u]["nodes"]
                if nodes and len(nodes) >= 2:
                    # reverse the polyline so it goes u->v visually (optional)
                    return _path_to_latlon_list(list(reversed(nodes)))

        # Special case: target-centric storage (often only ds.data[loc_key]['graph'][neighbor] exists)
        if u == loc_key and "graph" in ds.data[loc_key]:
            g = ds.data[loc_key]["graph"]
            if isinstance(g, dict) and v in g and "nodes" in g[v]:
                nodes = g[v]["nodes"]
                if nodes and len(nodes) >= 2:
                    return _path_to_latlon_list(nodes)

        if v == loc_key and "graph" in ds.data[loc_key]:
            g = ds.data[loc_key]["graph"]
            if isinstance(g, dict) and u in g and "nodes" in g[u]:
                nodes = g[u]["nodes"]
                if nodes and len(nodes) >= 2:
                    return _path_to_latlon_list(list(reversed(nodes)))

        return None

    # -------------------------
    # 4) Draw edges using detailed path when possible + collect path nodes
    # -------------------------
    path_nodes_keep = set()

    def _draw_edge_and_collect(u, v, dir_flag):
        dashed = (dir_flag == 0)
        path = _get_detailed_path(u, v)

        if path is None:
            # fallback straight line (no intermediate nodes to collect)
            lat1, lon1 = _as_latlon(u)
            lat2, lon2 = _as_latlon(v)
            path = [[lat1, lon1], [lat2, lon2]]
        else:
            # Collect intermediate nodes from the ORIGINAL graph storage if possible
            # We want the actual tuple nodes, not only lat/lon lists.
            # So we query the same source again but keep the raw "nodes" list.
            raw_nodes = None

            # Try u->v raw
            if u in ds.data and "graph" in ds.data[u]:
                g = ds.data[u]["graph"]
                if isinstance(g, dict) and v in g and isinstance(g[v], dict) and "nodes" in g[v]:
                    raw_nodes = g[v]["nodes"]

            # Try v->u raw
            if raw_nodes is None and v in ds.data and "graph" in ds.data[v]:
                g = ds.data[v]["graph"]
                if isinstance(g, dict) and u in g and isinstance(g[u], dict) and "nodes" in g[u]:
                    raw_nodes = list(reversed(g[u]["nodes"]))

            # Try loc_key storage raw
            if raw_nodes is None and u == loc_key and "graph" in ds.data[loc_key]:
                g = ds.data[loc_key]["graph"]
                if isinstance(g, dict) and v in g and "nodes" in g[v]:
                    raw_nodes = g[v]["nodes"]

            if raw_nodes is None and v == loc_key and "graph" in ds.data[loc_key]:
                g = ds.data[loc_key]["graph"]
                if isinstance(g, dict) and u in g and "nodes" in g[u]:
                    raw_nodes = list(reversed(g[u]["nodes"]))

            if raw_nodes is not None:
                for pn in raw_nodes:
                    path_nodes_keep.add(tuple(pn))

        folium.PolyLine(
            locations=path,
            color=edge_color,
            weight=3.2 if not dashed else 2.6,
            opacity=0.90 if not dashed else 0.70,
            dash_array="6,6" if dashed else None,
        ).add_to(m)

    # Draw only neighbor-to-target edges (your intent) and collect nodes on those paths
    for (src, dst, dir_flag) in selected_edges:
        if src in selected_nodes and dst in selected_nodes:
            if (src != loc_key and src in neighbor_nodes):
                _draw_edge_and_collect(src, loc_key, dir_flag)
            if (dst != loc_key and dst in neighbor_nodes):
                _draw_edge_and_collect(loc_key, dst, dir_flag)

    # -------------------------
    # 5) Draw nodes (neighbors + nodes on paths only)
    # -------------------------
    nodes_to_plot = (set([loc_key]) | set(neighbor_nodes) | set(path_nodes_keep)) & set(selected_nodes)
    nodes_to_plot = (vv for vv in nodes_to_plot if vv != (51.13943939050268, 3.6138691996185384))

    # Target node
    lat0, lon0 = _as_latlon(loc_key)
    _add_halo_circle(
        m, lat0, lon0,
        color="#ff1744",
        popup_html=f"<b>Virtual station (target)</b><br>{loc_key}",
        r=8
    )
    _add_text_label(
        m, lat0, lon0,
        text=f"{lat0:.5f}, {lon0:.5f}",
        dy=-10,
        font_size=11
    )

    for n in nodes_to_plot:
        if n == loc_key:
            continue

        lat, lon = _as_latlon(n)

        if n in neighbor_nodes:
            _add_halo_circle(
                m, lat, lon,
                color="#00c853",
                popup_html=f"<b>Neighbor station</b><br>{n}",
                r=6
            )
        else:
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,  # smaller since these are intermediate path nodes
                color="#333333",
                weight=1,
                fill=True,
                fill_color="#bdbdbd",
                fill_opacity=0.9,
                popup=folium.Popup(f"<b>Path node</b><br>{n}", max_width=320),
            ).add_to(m)
        _add_text_label(
            m, lat, lon,
            text=f"{lat:.5f}, {lon:.5f}",
            dy=-8,
            font_size=9
        )

    # -------------------------
    # 6) Legend
    # -------------------------
    legend_html = f"""
    <div style="
      position: fixed;
      top: 20px; left: 20px;
      z-index: 9999;
      background: rgba(255,255,255,0.95);
      padding: 10px 12px;
      border: 2px solid #333;
      border-radius: 6px;
      font-size: 13px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    ">
      <div style="font-weight:700;margin-bottom:6px;">Graph legend</div>

      <div style="display:flex;align-items:center;margin:4px 0;">
        <div style="width:14px;height:14px;background:#ff1744;border:1px solid #111;margin-right:8px;"></div>
        <div>Target virtual station</div>
      </div>

      <div style="display:flex;align-items:center;margin:4px 0;">
        <div style="width:14px;height:14px;background:#00c853;border:1px solid #111;margin-right:8px;"></div>
        <div>Direct neighbors</div>
      </div>

      <div style="display:flex;align-items:center;margin:4px 0;">
        <div style="width:14px;height:14px;background:#bdbdbd;border:1px solid #111;margin-right:8px;"></div>
        <div>Other trimmed nodes</div>
      </div>

      <div style="display:flex;align-items:center;margin:4px 0;">
        <div style="width:22px;height:0;border-top:3px solid {edge_color};margin-right:8px;"></div>
        <div>Directed edge</div>
      </div>

      <div style="display:flex;align-items:center;margin:4px 0;">
        <div style="width:22px;height:0;border-top:3px dashed {edge_color};margin-right:8px;"></div>
        <div>Bidirectional/unknown edge</div>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(out_html)
    print(f"Saved: {out_html}")
    print(f"Nodes: {len(selected_nodes)}, edges: {len(selected_edges)}")
    return out_html


def add_osm_waterways_overlay_from_points(
    m,
    lats,
    lons,
    pad_ratio=0.02,
    name="OSM Waterways",
    timeout=30,
    max_tries=6,
):
    """
    Robust waterways overlay:
      - tries multiple Overpass endpoints
      - retries with smaller bbox / fewer classes
      - does NOT raise if Overpass fails; caller can wrap too
    """
    lats = [float(x) for x in lats]
    lons = [float(x) for x in lons]
    if len(lats) == 0 or len(lons) == 0:
        raise ValueError("Empty lats/lons passed to waterways overlay")

    # bbox
    south, north = min(lats), max(lats)
    west,  east  = min(lons), max(lons)

    # pad bbox
    lat_pad = (north - south) * pad_ratio if north > south else 0.01
    lon_pad = (east - west) * pad_ratio if east > west else 0.01
    south -= lat_pad; north += lat_pad
    west  -= lon_pad; east  += lon_pad

    # Try progressively lighter queries
    #  - start broad: river|canal|stream|drain|ditch
    #  - then drop small: river|canal|stream
    #  - then only main: river|canal
    classes_schedule = [
        "river|canal|stream|drain|ditch",
        "river|canal|stream",
        "river|canal",
    ]

    # also shrink bbox progressively if still too big / timeout
    shrink_schedule = [1.0, 0.8, 0.6, 0.45, 0.35]

    def _shrink_bbox(s, w, n, e, factor):
        # shrink around center
        cy = 0.5 * (s + n)
        cx = 0.5 * (w + e)
        hy = 0.5 * (n - s) * factor
        hx = 0.5 * (e - w) * factor
        return cy - hy, cx - hx, cy + hy, cx + hx

    last_err = None
    attempt = 0

    for cls in classes_schedule:
        for shrink in shrink_schedule:
            s2, w2, n2, e2 = _shrink_bbox(south, west, north, east, shrink)

            query = f"""
            [out:json][timeout:25];
            (
              way["waterway"~"{cls}"]({s2},{w2},{n2},{e2});
            );
            out geom;
            """

            for url in OVERPASS_URLS:
                attempt += 1
                if attempt > max_tries:
                    break

                try:
                    r = requests.post(url, data={"data": query}, timeout=timeout)
                    r.raise_for_status()
                    data = r.json()

                    feats = []
                    for el in data.get("elements", []):
                        geom = el.get("geometry")
                        if not geom:
                            continue
                        coords = [[pt["lon"], pt["lat"]] for pt in geom]
                        props = {"waterway": el.get("tags", {}).get("waterway", "waterway")}
                        feats.append({
                            "type": "Feature",
                            "geometry": {"type": "LineString", "coordinates": coords},
                            "properties": props,
                        })

                    if not feats:
                        # success but nothing found — acceptable
                        print(f"[Waterways] No features found (cls={cls}, shrink={shrink:.2f}).")
                        return m

                    gj = {"type": "FeatureCollection", "features": feats}
                    folium.GeoJson(
                        gj,
                        name=name,
                        style_function=lambda f: {"color": "#1976d2", "weight": 1.0, "opacity": 0.85},
                        tooltip=folium.GeoJsonTooltip(fields=["waterway"], aliases=["waterway"]),
                    ).add_to(m)

                    print(f"[Waterways] Added {len(feats)} features (endpoint={url}, cls={cls}, shrink={shrink:.2f}).")
                    return m

                except Exception as e:
                    last_err = e
                    # brief backoff
                    time.sleep(0.6)

            if attempt > max_tries:
                break

    # If all failed, don't crash — just warn
    raise RuntimeError(f"Overpass failed after {attempt} tries. Last error: {last_err}")


def add_ultra_smooth_zoom_and_fit(
    m,
    lats, lons,
    pad_ratio=0.03,
    zoom_snap=0.1,
    zoom_delta=0.1,
    wheel_px_per_zoom=400,
    pad_px=(10, 10),
):
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)

    good = np.isfinite(lats) & np.isfinite(lons)
    lats = lats[good]
    lons = lons[good]

    if lats.size == 0:
        print("[add_ultra_smooth_zoom_and_fit] No valid (finite) station coords -> skip smooth zoom")
        return m

    south, north = float(lats.min()), float(lats.max())
    west,  east  = float(lons.min()), float(lons.max())

    lat_span = max(north - south, 1e-6)
    lon_span = max(east - west, 1e-6)

    south -= lat_span * pad_ratio
    north += lat_span * pad_ratio
    west  -= lon_span * pad_ratio
    east  += lon_span * pad_ratio

    m.add_child(SmoothZoomFit(
        south, west, north, east,
        zoom_snap=zoom_snap,
        zoom_delta=zoom_delta,
        wheel_px_per_zoom=wheel_px_per_zoom,
        pad_px=pad_px,
    ))
    return m


cfg = OmegaConf.load("config/idw.yaml")
with open("data/split.pkl", "rb") as f:
    split = pickle.load(f)
test_nb = split["test"]

ds = GWaterDataset(
    path="data/selected_stats_rainfall_segment.pkl",
    train=False,
    selected_stations=test_nb,
    input_type=cfg.dataset.inputs
)

loc_key = test_nb[0]
visualize_virtual_station_graph_html_detailed(ds, loc_key, out_html="graph_example.html")