"""Implementation of the wildfire gas analysis module."""
from pathlib import Path
from .english import install_plot_language
install_plot_language()
import io
import textwrap
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyproj import Geod
from .map_style import STUDY, LARGEST, BOUNDARY, north_arrow, coordinates, scale, study_legend, boundary_legend, clip_turkey
INK = '#173b4c'
MUTED = '#59727c'
RED = '#d65140'
BLUE = '#087f9c'
SEA = '#eaf3f6'
NAMES = dict(zip('Adiyaman Agri Aydin Balikesir Bartin Bingol Bitlis Canakkale Cankiri Corum Diyarbakir Duzce Elazig Eskisehir Gumushane Hakkari Igdir Istanbul Izmir Kahramanmaras Karabuk Kirikkale Kirklareli Kirsehir Kutahya Mugla Mus Nevsehir Nigde Sanliurfa Sirnak Tekirdag Usak'.split(), 'Adiyaman Agri Aydin Balikesir Bartin Bingol Bitlis Canakkale Cankiri Corum Diyarbakir Duzce Elazig Eskisehir Gumushane Hakkari Igdir Istanbul Izmir Kahramanmaras Karabuk Kirikkale Kirklareli Kirsehir Kutahya Mugla Mus Nevsehir Nigde Sanliurfa Sirnak Tekirdag Usak'.split()))

def tr(s):
    return NAMES.get(str(s), str(s))

def fmt(v):
    return f'{v:,.1f}'
scale = scale

def satellite_basemap(extent, width=4096):
    """Implementation of satellite_basemap."""
    import hashlib, json, urllib.request, urllib.parse
    from PIL import Image
    left, right, bottom, top = extent
    height = max(128, min(4096, round(width * (top - bottom) / (right - left))))
    params = {'bbox': ','.join((f'{v:.9f}' for v in (left, bottom, right, top))), 'bboxSR': 4326, 'imageSR': 4326, 'size': f'{width},{height}', 'format': 'jpg', 'transparent': 'false', 'f': 'json'}
    query = urllib.parse.urlencode(params)
    cache = Path(__file__).parent / 'satellite_cache'
    cache.mkdir(exist_ok=True)
    key = hashlib.sha256(query.encode()).hexdigest()[:24]
    jp = cache / (key + '.jpg')
    meta = cache / (key + '.json')
    if jp.exists() and meta.exists():
        try:
            with Image.open(jp) as im:
                arr = np.asarray(im.convert('RGB')).copy()
            info = json.loads(meta.read_text(encoding='utf-8'))
            return (arr, info['extent'])
        except (OSError, ValueError, KeyError):
            pass
    url = 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?' + query
    try:
        with urllib.request.urlopen(url, timeout=45) as response:
            data = json.load(response)
        if 'error' in data:
            raise ValueError(data['error'].get('message', 'Service error'))
        e = data['extent']
        bounds = [e['xmin'], e['xmax'], e['ymin'], e['ymax']]
        with urllib.request.urlopen(data['href'], timeout=45) as response:
            content = response.read()
        with Image.open(io.BytesIO(content)) as im:
            arr = np.asarray(im.convert('RGB')).copy()
        jp.write_bytes(content)
        meta.write_text(json.dumps({'extent': bounds, 'source': 'Esri World Imagery'}), encoding='utf-8')
        return (arr, bounds)
    except Exception as exc:
        raise RuntimeError('Could not retrieve satellite imagery. Check the internet connection or select the outline map.') from exc

def labels(fig, ax, provinces, satellite=False):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    used = []
    for row in provinces.assign(_area=provinces.to_crs(6933).area.values).sort_values('_area').itertuples():
        p = row.geometry.representative_point()
        name = tr(row.province)
        offsets = [(0, 0), (0, 7), (0, -7), (8, 0), (-8, 0), (0, 14), (0, -14), (13, 9), (-13, -9), (20, 0), (-20, 0)]
        best = None
        for dx, dy in offsets:
            t = ax.annotate(name, (p.x, p.y), xytext=(dx, dy), textcoords='offset points', ha='center', va='center', fontsize=7.1, color='white' if satellite else INK, zorder=10, path_effects=[pe.withStroke(linewidth=2, foreground='#132b30' if satellite else 'white', alpha=0.9)])
            box = t.get_window_extent(renderer).expanded(1.08, 1.2)
            if not any((box.overlaps(b) for b in used)):
                best = (box, dx, dy)
                break
            t.remove()
        if best is None:
            t = ax.annotate(name, (p.x, p.y), xytext=(0, 20), textcoords='offset points', ha='center', fontsize=6.7, color='white' if satellite else INK, zorder=10, path_effects=[pe.withStroke(linewidth=2, foreground='#132b30' if satellite else 'white')])
            box = t.get_window_extent(renderer)
            best = (box, 0, 20)
        used.append(best[0])
        if best[1] or best[2]:
            ax.annotate('', (p.x, p.y), xytext=(best[1], best[2]), textcoords='offset points', arrowprops=dict(arrowstyle='-', lw=0.4, color=MUTED), zorder=5)

def frame(ax):
    for spine in ax.spines.values():
        spine.set_color('#b6cbd2')
    ax.set_facecolor(SEA)
MAP_DIMENSIONS = {'turkey_location': (14, 15), 'fire_locations': (16, 11.4)}

def turkey_location_page(provinces, world, country, dpi, satellite, formats):
    region_extent = (19.0, 53.0, 31.0, 46.0)
    left, bottom, right, top = country.total_bounds
    extent = (left - 0.5, right + 0.5, bottom - 0.35, top + 0.35)
    regional = satellite_basemap(region_extent, 3072) if satellite else None
    national = satellite_basemap(extent) if satellite else None
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.grid': False}):
        fig = plt.figure(figsize=MAP_DIMENSIONS['turkey_location'], facecolor='white')
        fig.text(0.075, 0.97, 'Turkey Location Map', fontsize=25, fontweight='bold', color=INK)
        fig.text(0.075, 0.928, 'TURKEY AND SURROUNDING REGION', fontsize=12, fontweight='bold', color='black')
        upper = fig.add_axes([0.08, 0.55, 0.82, 0.35])
        upper.set_label('bolgesel_location')
        frame(upper)
        if satellite:
            upper.imshow(regional[0], extent=regional[1], origin='upper', zorder=0)
            world.boundary.plot(ax=upper, color='white', linewidth=0.6)
        else:
            world.plot(ax=upper, facecolor='#e4eadf', edgecolor='#657a70', linewidth=0.6)
        country.boundary.plot(ax=upper, color=BOUNDARY, linewidth=1.5, zorder=3)
        entries = [('TURKEY', 34.8, 39.1, 15), ('GREECE', 22.5, 39.5, 9), ('BULGARIA', 25, 42.8, 9), ('GEORGIA', 43.5, 42.5, 9), ('ARMENIA', 44.3, 40.7, 8), ('AZERBAIJAN', 49.9, 39.9, 8), ('IRAN', 49, 36.8, 10), ('IRAQ', 44.3, 33.2, 10), ('SYRIA', 38.8, 34.6, 10), ('CYPRUS', 33.3, 35.2, 8), ('ROMANIA', 25, 45, 9), ('RUSSIA', 48.4, 44.3, 9)]
        for name, x, y, size in entries:
            upper.text(x, y, name, ha='center', va='center', fontsize=size, fontweight='bold', color=('#ffff00' if name == 'TURKEY' else 'white') if satellite else INK, zorder=4, path_effects=[pe.withStroke(linewidth=2.5, foreground='#172d35' if satellite else 'white')])
        for name, x, y in [('BLACK SEA', 34.5, 43.5), ('MEDITERRANEAN SEA', 29, 32.8)]:
            upper.text(x, y, name, ha='center', fontsize=9, fontstyle='italic', color='#c9eaf3' if satellite else BLUE)
        upper.set(xlim=region_extent[:2], ylim=region_extent[2:])
        upper.set_aspect(1 / np.cos(np.deg2rad(38.5)))
        coordinates(upper, fontsize=8, nx=7, ny=4, bold=True)
        north_arrow(fig, [0.93, 0.77, 0.035, 0.1])
        fig.canvas.draw()
        scale(fig, upper, 0.505, 1000000)
        fig.text(0.075, 0.491, 'TURKEY', fontsize=12, fontweight='bold', color='black')
        lower = fig.add_axes([0.08, 0.12, 0.82, 0.34])
        lower.set_label('turkey_detay')
        lower.set_facecolor('white')
        if satellite:
            im = lower.imshow(national[0], extent=national[1], origin='upper', zorder=0)
            clip_turkey(im, country.geometry, lower)
            provinces.boundary.plot(ax=lower, color='white', linewidth=0.4, alpha=0.65)
        else:
            provinces.plot(ax=lower, facecolor='#e7ecdf', edgecolor='#91a392', linewidth=0.5)
        country.boundary.plot(ax=lower, color=BOUNDARY, linewidth=1.5)
        lower.set(xlim=extent[:2], ylim=extent[2:])
        lower.set_aspect(1 / np.cos(np.deg2rad((bottom + top) / 2)))
        labels(fig, lower, provinces, satellite=satellite)
        coordinates(lower, fontsize=8, nx=7, ny=5, bold=True)
        north_arrow(fig, [0.93, 0.475, 0.035, 0.065])
        fig.canvas.draw()
        scale(fig, lower, 0.076, 400000)
        fig.legend(handles=[boundary_legend()], loc='upper left', bbox_to_anchor=(0.69, 0.092), frameon=True, fancybox=False, edgecolor='black', facecolor='white', framealpha=1, fontsize=10)
        fig.text(0.075, 0.042, "WGS 84 · Scale bars are calculated at each map's central latitude.", fontsize=8, color=MUTED)
        fig.text(0.075, 0.026, 'Province boundaries: FAO GAUL 2025 (CC BY 4.0) · Country boundaries: Natural Earth and the existing Turkey boundary.', fontsize=7, color=MUTED)
        if satellite:
            fig.text(0.075, 0.011, 'Imagery: Esri, Vantor, Earthstar Geographics and GIS User Community.', fontsize=7, color=MUTED)
        out = {}
        for extension in formats:
            buf = io.BytesIO()
            fig.savefig(buf, format=extension, dpi=dpi, facecolor='white')
            out[extension] = buf.getvalue()
        plt.close(fig)
        return out

def location_maps(gdf, boundary_file=None, period='', dpi=120, satellite=False, map=None, formats=('png',)):
    if dpi not in (120, 300, 600):
        raise ValueError('DPI must be 120, 300 or 600.')
    if map not in (None, 'turkey_location', 'fire_locations'):
        raise ValueError('Unknown map.')
    root = Path(__file__).parent
    provinces = gpd.read_file(root / 'turkey_provinces.geojson').to_crs(4326)
    from shapely.ops import unary_union

    def polygons(geom):
        if geom.geom_type == 'Polygon':
            return [geom]
        if hasattr(geom, 'geoms'):
            return [p for child in geom.geoms for p in polygons(child)]
        return []
    provinces.geometry = provinces.geometry.map(lambda g: unary_union(polygons(g)))
    world = gpd.read_file(root / 'world_boundary.geojson').to_crs(4326)
    country = gpd.read_file(root / 'turkey_boundary.geojson').to_crs(4326)
    results = {}
    if map in (None, 'turkey_location'):
        results['turkey_location'] = turkey_location_page(provinces, world, country, dpi, satellite, formats)
        if map == 'turkey_location':
            return results
    fires = gdf.to_crs(4326).copy()
    areas = pd.to_numeric(fires['area_ha'], errors='coerce')
    if fires.empty or not areas.notna().any():
        raise ValueError('Fire geometry and area in hectares are required.')
    largest = fires.iloc[[int(np.nanargmax(areas.to_numpy()))]]
    row = largest.iloc[0]
    point = row.geometry.representative_point()
    loc = ' · '.join(dict.fromkeys((tr(row.get(k, '')) for k in ('district', 'province') if pd.notna(row.get(k)) and str(row.get(k, '')))))
    if not loc:
        loc = str(row.get('name', 'Fire area'))
    left, bottom, right, top = country.total_bounds
    extent = (left - 0.5, right + 0.5, bottom - 0.35, top + 0.35)
    x0, y0, x1, y1 = largest.total_bounds
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    dx = max(x1 - x0, 0.025) * 0.72
    dy = max(y1 - y0, 0.025) * 0.72
    dx = max(dx, dy * 1.8 / np.cos(np.deg2rad(cy)))
    dy = max(dy, dx * np.cos(np.deg2rad(cy)) / 1.8)
    zoom_extent = (cx - dx, cx + dx, cy - dy, cy + dy)
    main_bg = satellite_basemap(extent) if satellite else None
    zoom_bg = satellite_basemap(zoom_extent, 2048) if satellite else None
    fire_color = STUDY
    accent = LARGEST
    for fire in (True,):
        name = 'fire_locations' if fire else 'turkey_location'
        if map is not None and name != map:
            continue
        with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.grid': False}):
            fig = plt.figure(figsize=MAP_DIMENSIONS['fire_locations'], facecolor='white')
            fig.text(0.045, 0.957, 'WILDFIRE GAS ANALYSIS  /  LOCATION ATLAS', fontsize=9, color=BLUE, fontweight='bold')
            fig.text(0.045, 0.914, 'Wildfire Locations' if fire else 'Turkey and Its Provinces', fontsize=25, color=INK, fontweight='bold')
            fig.text(0.045, 0.88, f'{period}  ·  Listed {len(fires)} fire areas' if fire else 'Surrounding region → Turkey → Largest wildfire area', fontsize=11, color=MUTED)
            ax = fig.add_axes([0.045, 0.405, 0.91, 0.44])
            frame(ax)
            palette = ['#e8ece0', '#f0eddf', '#e0e9e2', '#e9e9e0']
            if satellite:
                sat = ax.imshow(main_bg[0], extent=main_bg[1], origin='upper', zorder=0)
                clip_turkey(sat, country.geometry, ax)
                ax.set_facecolor('white')
                provinces.boundary.plot(ax=ax, color='#edf3ee', linewidth=0.5, alpha=0.8, zorder=2)
                country.boundary.plot(ax=ax, color=BOUNDARY, linewidth=1.1, zorder=2)
            else:
                provinces.plot(ax=ax, color=[palette[i % 4] for i in range(len(provinces))], edgecolor='#8fa49d', linewidth=0.55)
                country.boundary.plot(ax=ax, color=BOUNDARY, linewidth=0.85)
            if fire:
                fires.plot(ax=ax, facecolor='none', edgecolor=STUDY, linewidth=0.9, zorder=3)
                others = fires.iloc[np.arange(len(fires)) != int(np.nanargmax(areas.to_numpy()))]
                pts = others.geometry.representative_point()
                ax.scatter(pts.x, pts.y, s=48, c=fire_color, edgecolors='#273b40', linewidths=0.7, zorder=11)
            largest.plot(ax=ax, facecolor='none', edgecolor=STUDY, linewidth=1, zorder=8)
            ax.scatter([point.x], [point.y], marker='o', s=155, c=accent, edgecolors='white', linewidths=1.3, zorder=12)
            ax.set(xlim=extent[:2], ylim=extent[2:], xlabel='Longitude (°E)', ylabel='Latitude (°N)')
            ax.set_aspect(1 / np.cos(np.deg2rad((bottom + top) / 2)))
            ax.tick_params(labelsize=8, colors=MUTED)
            ax.grid(True, lw=0.35, color='#c7d8dd', alpha=0.6)
            ax.set_axisbelow(True)
            labels(fig, ax, provinces, satellite=satellite)
            north_arrow(fig, [0.958, 0.7, 0.032, 0.1])
            coordinates(ax, fontsize=7, nx=9, ny=5, bold=True)
            fig.canvas.draw()
            scale(fig, ax, 0.35, 400000)
            fig.text(0.075, 0.324, 'LARGEST WILDFIRE · CLOSE-UP', fontsize=10, fontweight='bold', color=INK)
            zoom = fig.add_axes([0.075, 0.11, 0.36, 0.194])
            frame(zoom)
            x0, y0, x1, y1 = largest.total_bounds
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            dx = max(x1 - x0, 0.025) * 0.72
            dy = max(y1 - y0, 0.025) * 0.72
            dx = max(dx, dy * 1.8 / np.cos(np.deg2rad(cy)))
            dy = max(dy, dx * np.cos(np.deg2rad(cy)) / 1.8)
            if satellite:
                zoom.imshow(zoom_bg[0], extent=zoom_bg[1], origin='upper', zorder=0)
                provinces.boundary.plot(ax=zoom, color='white', linewidth=0.5, alpha=0.65)
                largest.plot(ax=zoom, facecolor='none', edgecolor=STUDY, linewidth=1.6, zorder=4)
            else:
                provinces.plot(ax=zoom, facecolor='#edf0e7', edgecolor='#a4b5ad', linewidth=0.5)
                largest.plot(ax=zoom, facecolor=STUDY, edgecolor='black', linewidth=0.8)
            zoom.set(xlim=(cx - dx, cx + dx), ylim=(cy - dy, cy + dy))
            zoom.set_aspect(1 / np.cos(np.deg2rad(cy)))
            zoom.set_xlabel('Longitude (°E)', fontsize=7)
            zoom.set_ylabel('Latitude (°N)', fontsize=7)
            zoom.tick_params(labelsize=6, colors=MUTED)
            zoom.ticklabel_format(useOffset=False)
            coordinates(zoom, fontsize=5, nx=4, ny=4, bold=True)
            north_arrow(fig, [0.44, 0.24, 0.025, 0.075])
            fig.canvas.draw()
            scale(fig, zoom, 0.061)
            fig.text(0.54, 0.296, 'LARGEST LISTED AREA', fontsize=9, fontweight='bold', color=BLUE)
            fig.text(0.54, 0.265, '\n'.join(textwrap.wrap(loc, 31)), fontsize=13, fontweight='bold', color=INK, va='top')
            fig.text(0.54, 0.207, f'{fmt(float(row.area_ha))} ha', fontsize=23, fontweight='bold', color=BLUE)
            handles = [study_legend(), boundary_legend(), Line2D([], [], marker='o', color='none', markerfacecolor=LARGEST, markeredgecolor='black', markersize=10, label='Largest wildfire')]
            if fire:
                handles.insert(2, Line2D([], [], marker='o', color='none', markerfacecolor=STUDY, markeredgecolor='black', markersize=6, label='Fire location'))
            fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.534, 0.185), frameon=True, fancybox=False, edgecolor='black', facecolor='white', framealpha=1, fontsize=8)
            fig.text(0.045, 0.053, "WGS 84 · Scale bars use each map's central latitude. Location circles do not represent area.", fontsize=7, color=MUTED)
            fig.text(0.045, 0.035, 'Province boundaries: FAO GAUL 2025 (CC BY 4.0), simplified at 150 m. Country boundaries: Natural Earth.', fontsize=7, color=MUTED)
            fig.text(0.045, 0.018, 'Imagery: Esri, Vantor, Earthstar Geographics, GIS User Community. The basemap does not show the fire date.' if satellite else 'The largest area is selected from the current list. Symbol sizes do not represent actual area.', fontsize=7, color=MUTED)
            outputs = {}
            for fmtname in formats:
                buf = io.BytesIO()
                fig.savefig(buf, format=fmtname, dpi=dpi, facecolor=fig.get_facecolor())
                outputs[fmtname] = buf.getvalue()
            plt.close(fig)
            results['fire_locations' if fire else 'turkey_location'] = outputs
    return results
