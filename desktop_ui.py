"""Implementation of the wildfire gas analysis module."""
from .english import tr as _best
import os
import queue
import subprocess
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
BACKGROUND = '#f7f7f5'
CARD = '#ffffff'
BLUE = '#2a78d6'
INK = '#1f1f1d'
SECONDARY = '#6b6a66'

class _Redirect:

    def __init__(self, q):
        self.q = q

    def write(self, s):
        self.q.put(_best(s))

    def flush(self):
        pass

def _style(root):
    st = ttk.Style(root)
    try:
        st.theme_use('clam')
    except tk.TclError:
        pass
    root.configure(bg=BACKGROUND)
    st.configure('.', background=BACKGROUND, foreground=INK, font=('Segoe UI', 10))
    st.configure('Kart.TLabelframe', background=CARD, bordercolor='#dddcd7', relief='solid')
    st.configure('Kart.TLabelframe.Label', background=BACKGROUND, foreground=BLUE, font=('Segoe UI', 10, 'bold'))
    st.configure('Kart.TFrame', background=CARD)
    st.configure('Kart.TLabel', background=CARD)
    st.configure('Ipucu.TLabel', background=CARD, foreground=SECONDARY, font=('Segoe UI', 8))
    st.configure('Kart.TCheckbutton', background=CARD)
    st.configure('Title.TLabel', font=('Segoe UI', 15, 'bold'))
    st.configure('Alt.TLabel', foreground=SECONDARY, font=('Segoe UI', 9))
    st.configure('Ana.TButton', font=('Segoe UI', 10, 'bold'), foreground='white', background=BLUE, padding=(18, 7))
    st.map('Ana.TButton', background=[('active', '#1c5cab'), ('disabled', '#9ec5f4')])
    st.configure('TButton', padding=(8, 3))
    st.configure('TEntry', fieldbackground='white', padding=3)

def run(on_fill=None, example_daily=None, screen_image=None):
    """Implementation of run."""
    root = tk.Tk()
    root.title(_best('YanginGazAnaliz — Wildfire Gas Analysis'))
    root.geometry('980x900')
    root.minsize(900, 700)
    _style(root)
    area = {}
    upper = ttk.Frame(root, padding=(16, 12, 16, 4))
    upper.pack(fill='x')
    ttk.Label(upper, text=_best('Wildfire Gas Analysis'), style='Title.TLabel').pack(anchor='w')
    ttk.Label(upper, text=_best('Sentinel-5P (CO, NO2, CH2O, SO2, O3) + ground stations · automatic fire dates and gas change detection'), style='Alt.TLabel').pack(anchor='w')
    body = ttk.Frame(root, padding=(16, 6))
    body.pack(fill='x')
    body.columnconfigure(0, weight=1)
    body.columnconfigure(1, weight=1)

    def card(title, r, c, cs=1):
        lf = ttk.LabelFrame(body, text=_best(f'  {title}  '), style='Kart.TLabelframe', padding=10)
        lf.grid(row=r, column=c, columnspan=cs, sticky='nsew', padx=5, pady=5)
        lf.columnconfigure(1, weight=1)
        return lf

    def row(parent, r, label, key, file=None, directory=False, default='', hint=None, gen=30):
        ttk.Label(parent, text=_best(label), style='Kart.TLabel').grid(row=r, column=0, sticky='w', pady=3, padx=(0, 8))
        v = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=v, width=gen).grid(row=r, column=1, sticky='we', pady=3)
        if file or directory:

            def select():
                y = filedialog.askdirectory() if directory else filedialog.askopenfilename(filetypes=file)
                if y:
                    v.set(y)
            ttk.Button(parent, text=_best('Browse...'), command=select).grid(row=r, column=2, padx=(6, 0))
        if hint:
            ttk.Label(parent, text=_best(hint), style='Ipucu.TLabel').grid(row=r + 1, column=1, columnspan=2, sticky='w')
        area[key] = v
    geo = [('Vector', '*.shp *.geojson *.gpkg *.json'), ('Table', '*.csv *.xlsx'), ('All', '*.*')]
    tab = [('Table', '*.csv *.xlsx *.xls'), ('All', '*.*')]
    k1 = card('1  Fire area', 0, 0, cs=2)
    row(k1, 0, 'Fire polygons (SHP/GPKG)', 'shp', geo, hint='Each polygon is analyzed as a separate fire.', gen=70)
    row(k1, 2, 'Search year (if date is unknown)', 'year', default='2025', gen=12)
    row(k1, 3, 'Name column (optional)', 'adsutun', gen=12)
    row(k1, 4, 'EPSG (if .prj is missing)', 'epsg', hint='TM30 = 5254 · TM27 = 5253 · UTM 36N = 32636', gen=12)
    row(k1, 6, 'Minimum area (ha)', 'minalan', default='100', hint='Sentinel-5P pixel ≈ 5.5 × 3.5 km — small fires cannot be resolved', gen=12)
    k2 = card('2  Data sources', 1, 0)
    row(k2, 0, 'GEE project ID', 'project')
    row(k2, 1, 'Ground station (μg/m3)', 'list_stats', tab)
    row(k2, 2, 'Station directory (batch)', 'istk', directory=True)
    row(k2, 3, 'Meteorology', 'met', tab, hint='ERA5-Land if left blank')
    row(k2, 5, 'Existing Sentinel-5P time series', 'satellite', tab, hint='Retrieved from GEE if left blank')
    k3 = card('3  Analysis settings', 1, 1)
    row(k3, 0, 'Pre-fire baseline (days)', 'before', default='10', gen=10)
    row(k3, 1, 'Post-fire period (days)', 'after', default='8', gen=10)
    row(k3, 2, 'z-score threshold', 'z', default='2.5', gen=10)
    ttk.Label(k3, text=_best('Sentinel-5P product'), style='Kart.TLabel').grid(row=3, column=0, sticky='w', pady=3)
    product = tk.StringVar(value='OFFL')
    ttk.Combobox(k3, textvariable=product, values=['OFFL', 'NRTI'], width=8, state='readonly').grid(row=3, column=1, sticky='w')
    row(k3, 4, 'CH2O factor', 'factor', hint='Not measured at stations')
    row(k3, 6, 'Output directory', 'output', directory=True, default='output')
    only = tk.BooleanVar(value=False)
    ttk.Checkbutton(k3, text=_best('Detect fire dates only (FIRMS / MCD64A1)'), variable=only, style='Kart.TCheckbutton').grid(row=7, column=0, columnspan=3, sticky='w', pady=(6, 0))
    lower = ttk.Frame(root, padding=(21, 4))
    lower.pack(fill='x')
    progress = ttk.Progressbar(lower, mode='indeterminate', length=220)
    status = tk.StringVar(value='Ready')
    ttk.Label(lower, textvariable=status, style='Alt.TLabel').pack(side='left')
    logf = ttk.Frame(root, padding=(21, 2, 21, 12))
    logf.pack(fill='both', expand=True)
    log = tk.Text(logf, height=14, font=('Consolas', 9), bg='#1e1e1c', fg='#e8e7e1', insertbackground='white', relief='flat', padx=10, pady=8)
    sb = ttk.Scrollbar(logf, command=log.yview)
    log.configure(yscrollcommand=sb.set)
    sb.pack(side='right', fill='y')
    log.pack(fill='both', expand=True)
    q = queue.Queue()

    def stream():
        while not q.empty():
            log.insert('end', q.get())
            log.see('end')
        root.after(200, stream)

    def finished():
        progress.stop()
        progress.pack_forget()
        btn.config(state='normal')
        status.set(_best('Completed'))

    def worker_thread(a):
        previous = sys.stdout
        sys.stdout = _Redirect(q)
        try:
            from .__main__ import main
            print(_best('Running: python -m wildfire_gas'), _best(' '.join(a)), _best('\n'))
            main(a)
            print(_best('\nDONE.'))
        except SystemExit:
            pass
        except Exception:
            print(_best(traceback.format_exc()))
        finally:
            sys.stdout = previous
            root.after(0, finished)

    def arguments():
        v = {k: x.get().strip() for k, x in area.items()}
        a = []
        if v['shp']:
            a += ['--shp', v['shp']]
        for k, b in (('satellite', '--satellite'), ('list_stats', '--station'), ('istk', '--station-directory'), ('met', '--weather'), ('project', '--project'), ('output', '--output'), ('year', '--year'), ('epsg', '--epsg'), ('adsutun', '--name-column'), ('factor', '--factor'), ('before', '--before-day'), ('after', '--after-day'), ('z', '--z-threshold')):
            if v[k]:
                a += [b, v[k]]
        if v['minalan'] and float(v['minalan']) > 0:
            a += ['--min-area-ha', v['minalan']]
        a += ['--product', product.get()]
        if only.get():
            a += ['--only-date']
        return a

    def launch():
        if not area['shp'].get() and (not area['satellite'].get()):
            messagebox.showwarning(_best('Missing input'), _best('Select a fire polygon (SHP) or an existing satellite time series.'))
            return
        btn.config(state='disabled')
        status.set(_best('Running...'))
        progress.pack(side='left', padx=10)
        progress.start(12)
        threading.Thread(target=worker_thread, args=(arguments(),), daemon=True).start()

    def directory_open():
        y = os.path.abspath(area['output'].get() or 'output')
        os.makedirs(y, exist_ok=True)
        if sys.platform.startswith('win'):
            os.startfile(y)
        else:
            subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', y])
    btn = ttk.Button(lower, text=_best('▶  Run analysis'), style='Ana.TButton', command=launch)
    btn.pack(side='right')
    ttk.Button(lower, text=_best('Open output directory'), command=directory_open).pack(side='right', padx=8)
    if on_fill:
        for k, v in on_fill.items():
            if k in area:
                area[k].set(v)
    if example_daily:
        log.insert('end', example_daily)
        status.set(_best('Running...'))
        progress.pack(side='left', padx=10)
        progress.start(12)
    if screen_image:
        root.after(1500, lambda: (screen_image(root), root.destroy()))
    stream()
    root.mainloop()
    return 0
