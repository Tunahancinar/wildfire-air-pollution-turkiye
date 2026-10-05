"""Implementation of the wildfire gas analysis module."""
import numpy as np
from matplotlib.patches import Polygon, Rectangle, Patch, PathPatch
from matplotlib.path import Path
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pyproj import Geod
STUDY = '#ffff00'
LARGEST = '#00d4f0'
BOUNDARY = '#111111'

def north_arrow(fig, rect):
    ax = fig.add_axes(rect)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ax.add_patch(Polygon([(0.5, 0.8), (0.19, 0.13), (0.5, 0.32)], closed=True, facecolor='black', edgecolor='black', lw=0.7))
    ax.add_patch(Polygon([(0.5, 0.8), (0.81, 0.13), (0.5, 0.32)], closed=True, facecolor='white', edgecolor='black', lw=0.7))
    ax.text(0.5, 0.91, 'N', ha='center', va='center', fontsize=9, fontweight='bold', color='black')
    ax.set_label('combined_north_read')
    return ax

def coordinates(ax, fontsize=7, nx=6, ny=5, bold=False):
    ax.xaxis.set_major_locator(MaxNLocator(nx))
    ax.yaxis.set_major_locator(MaxNLocator(ny))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{abs(x):.2f}°' + ('E' if x >= 0 else 'W')))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{abs(y):.2f}°' + ('N' if y >= 0 else 'S')))
    ax.tick_params(top=True, right=True, labeltop=True, labelright=True, direction='out', length=4 if bold else 3, width=1.4 if bold else 0.7, labelsize=fontsize, colors='black', pad=3)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color('black')
        sp.set_linewidth(1.6 if bold else 0.7)
    if bold:
        for label in ax.get_xticklabels(which='both') + ax.get_yticklabels(which='both'):
            label.set_fontweight('bold')

def scale(fig, ax, y, length=None):
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    lat = (bottom + top) / 2
    geod = Geod(ellps='WGS84')
    available = geod.inv(left, lat, left + (right - left) * 0.42, lat)[2]
    if length is None:
        power = 10 ** np.floor(np.log10(available))
        length = max((v * power for v in (0.1, 0.2, 0.5, 1, 2, 5) if v * power <= available))
    x = left + (right - left) * 0.02
    lo, hi = (0.0, right - left)
    for _ in range(50):
        span = (lo + hi) / 2
        if geod.inv(x, lat, x + span, lat)[2] < length:
            lo = span
        else:
            hi = span
    pos = ax.get_position()
    bar = fig.add_axes([pos.x0, y, pos.width, 0.032])
    bar.axis('off')
    bar.set(xlim=(left, right), ylim=(0, 1))
    for i in range(5):
        bar.add_patch(Rectangle((x + i * span / 5, 0.1), span / 5, 0.32, facecolor='black' if i % 2 == 0 else 'white', edgecolor='black', lw=1.0))
    spacing = pos.width * (fig.bbox.width / fig.dpi) * 72 * span / (right - left) / 5
    required = len(f'{length / 1000:g}') * 5.3 + 6
    ticks = range(6) if spacing >= required else (0, 2, 5)
    for i in ticks:
        bar.text(x + i * span / 5, 0.62, f'{length * i / 5000:g}', ha='center', fontsize=9, fontweight='bold', color='black')
    bar.text(x + span + (right - left) * 0.015, 0.1, 'km', fontsize=10, fontweight='bold', color='black', va='bottom')
    bar.set_label('combined_parcali_scale')
    bar._distance_m = length
    bar._longitude_span = span
    bar._reference_lat = lat
    bar._start_lon = x
    return bar

def study_legend():
    return Patch(facecolor=STUDY, edgecolor='black', linewidth=0.8, label='Study fire boundary')

def boundary_legend():
    return Patch(facecolor='white', edgecolor='black', linewidth=0.8, label='Turkey boundary')

def clip_turkey(image, geometries, ax):
    """Implementation of clip_turkey."""
    from shapely.geometry.polygon import orient
    paths = []

    def parts(g):
        if g.geom_type == 'Polygon':
            return [g]
        return [p for sub in getattr(g, 'geoms', []) for p in parts(sub)]
    for geom in geometries:
        for poly in parts(geom):
            poly = orient(poly, sign=1.0)
            for ring in [poly.exterior, *poly.interiors]:
                vertices = np.asarray(ring.coords)[:, :2]
                codes = np.full(len(vertices), Path.LINETO)
                codes[0] = Path.MOVETO
                codes[-1] = Path.CLOSEPOLY
                paths.append(Path(vertices, codes))
    patch = PathPatch(Path.make_compound_path(*paths), transform=ax.transData)
    image.set_clip_path(patch)
