"""Wrap raw plotly 3D HTML files with the a9l.im site-themed shell.

Two wrappers:

1. **Procrustes overlay**: takes the multi-scene raw plotly HTML output by
   `26_v3_quadrant_procrustes.py` and extracts only the overlay scene
   (scene6 for 5 models, scene N+1 generally) into a single full-pane
   plotly view, plus a static legend bar showing per-model markers.

2. **Per-face PCA with model toggle**: takes the multi-scene raw plotly
   HTML output by the per-face PCA generator, exposes a model
   segmented-control via shared-forms, and re-themes the active scene
   in-place using the site's CSS custom-property palette.

Both wrappers re-paint plotly's grid/text/background colors from the
parent document's `data-theme` attribute (with a MutationObserver), so
the figures track the site's light/dark toggle live.

Usage:
    .venv/bin/python scripts/local/98_wrap_blog_3d_html.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
BLOG_ASSETS_DIR = (
    STUDY_ROOT.parent / "a9lim.github.io" / "blog-assets" / "introspection-via-kaomoji"
)
RAW_PROCRUSTES = STUDY_ROOT / "figures" / "local" / "fig_v3_quadrant_procrustes_3d.html"
RAW_PER_FACE = STUDY_ROOT / "figures" / "local" / "fig_v3_per_face_pca_3d.html"
RAW_WILD_FACES = STUDY_ROOT / "figures" / "harness" / "wild_faces_pca_3d.html"
OUT_PROCRUSTES = BLOG_ASSETS_DIR / "fig_v3_quadrant_procrustes_3d.html"
OUT_PER_FACE = BLOG_ASSETS_DIR / "fig_v3_per_face_pca_3d.html"
OUT_WILD_FACES = BLOG_ASSETS_DIR / "fig_wild_faces_pca_3d.html"


# Active models, in order. The procrustes file currently emits 5 per-model
# scenes (scene1..scene5) + 1 overlay (scene6).
MODELS = ("gemma", "qwen", "ministral", "gpt_oss_20b", "granite")
# Plotly 3D marker symbols used by 26_v3_quadrant_procrustes.py.
MODEL_MARKERS = {
    "gemma":       "circle",
    "qwen":        "diamond",
    "ministral":   "square",
    "gpt_oss_20b": "cross",
    "granite":     "x",
}


def _extract_plotly_block(raw: str) -> str:
    """Pull just the `<div>...plotly init...<script>...</script></div>` block
    out of a py-plotly-emitted full HTML file. Skips the surrounding
    `<html>`, `<head>`, `<body>` boilerplate so we can paste it inside our
    own shell."""
    m = re.search(r"<body>\s*(<div>.*</div>)\s*</body>", raw, re.DOTALL)
    if not m:
        raise SystemExit(f"could not find <body><div>...</div></body> block in raw plotly HTML")
    return m.group(1)


# Common shared head — fonts, tokens, base CSS. Forms only needed for the
# per-face wrapper but harmless on the procrustes one (it's a defer load).
_SHARED_HEAD = """<head>
<meta charset="utf-8" />
<script src="/shared-tokens.js"></script>
<link rel="stylesheet" href="/fonts/fonts.css">
<link rel="stylesheet" href="/shared-base.css">
<script defer src="/shared-forms.js"></script>
"""

# Common reset + plotly transparent-background overrides shared by both
# wrappers. The `_THEME_JS_PRE` block runs in both — we factor it once.
_BASE_CSS = """  html, body {
    margin: 0;
    padding: 0;
    background: transparent !important;
    overflow: hidden;
    height: 100%;
  }
  #caption-bar, #model-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 10px 16px 4px;
    flex-wrap: wrap;
    background: transparent;
    font-family: var(--font-mono);
    font-size: var(--font-sm);
    font-variation-settings: 'MONO' 1;
  }
  .legend-static {
    display: flex;
    gap: 14px;
    align-items: center;
    color: var(--text);
    flex-wrap: wrap;
  }
  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    letter-spacing: 0.04em;
    text-transform: lowercase;
    padding: 6px 0;
  }
  .legend-mark {
    display: inline-block;
    width: 9px;
    height: 9px;
    background: var(--text-muted);
  }
  .legend-mark.mark-circle  { border-radius: 50%; }
  .legend-mark.mark-diamond { transform: rotate(45deg); }
  .legend-mark.mark-square  {}
  .legend-mark.mark-cross   { background: none; position: relative; }
  .legend-mark.mark-cross::before, .legend-mark.mark-cross::after {
    content: ''; position: absolute; left: 50%; top: 50%; width: 9px; height: 2px;
    background: var(--text-muted); transform: translate(-50%, -50%);
  }
  .legend-mark.mark-cross::after { transform: translate(-50%, -50%) rotate(90deg); }
  .legend-mark.mark-x { background: none; position: relative; }
  .legend-mark.mark-x::before, .legend-mark.mark-x::after {
    content: ''; position: absolute; left: 50%; top: 50%; width: 11px; height: 2px;
    background: var(--text-muted); transform: translate(-50%, -50%) rotate(45deg);
  }
  .legend-mark.mark-x::after { transform: translate(-50%, -50%) rotate(-45deg); }
  .subtitle {
    color: var(--text-muted);
    letter-spacing: 0.02em;
  }
  #model-bar .mode-btn {
    font-family: var(--font-mono);
    font-size: var(--font-sm);
    font-variation-settings: 'MONO' 1;
    letter-spacing: 0.04em;
    text-transform: lowercase;
    padding: 6px 14px;
    flex: 0 0 auto;
  }
  .plotly-graph-div {
    width: 100% !important;
    height: calc(100vh - 50px) !important;
    background: transparent !important;
  }
  .plot-container, .svg-container, .main-svg { background: transparent !important; }"""


# Color helpers used by both wrappers.
_THEME_JS_HELPERS = """  function readTheme() {
    try {
      var t = window.parent && window.parent !== window
        ? window.parent.document.documentElement.dataset.theme
        : document.documentElement.dataset.theme;
      return t === 'dark' ? 'dark' : 'light';
    } catch (e) { return 'light'; }
  }

  // Pre-composite an 8-digit hex (#RRGGBBAA) over a known opaque background
  // so plotly's 3D grid gets a flat opaque color (it strips the alpha
  // channel off gridcolor in WebGL, which renders dark-mode grids as
  // bright white).
  function compositeOver(fgHex, bgHex) {
    if (!fgHex || fgHex.length < 7) return bgHex;
    var fg = fgHex.replace('#', '');
    var bg = bgHex.replace('#', '');
    var fr = parseInt(fg.slice(0, 2), 16);
    var fgr = parseInt(fg.slice(2, 4), 16);
    var fgb = parseInt(fg.slice(4, 6), 16);
    var a = fg.length >= 8 ? parseInt(fg.slice(6, 8), 16) / 255 : 1;
    var br = parseInt(bg.slice(0, 2), 16);
    var bgg = parseInt(bg.slice(2, 4), 16);
    var bbg = parseInt(bg.slice(4, 6), 16);
    var rr = Math.round(fr * a + br * (1 - a));
    var rg = Math.round(fgr * a + bgg * (1 - a));
    var rb = Math.round(fgb * a + bbg * (1 - a));
    var to2 = function (n) { return ('0' + n.toString(16)).slice(-2); };
    return '#' + to2(rr) + to2(rg) + to2(rb);
  }

  function readThemeColors() {
    var theme = readTheme();
    document.documentElement.dataset.theme = theme;
    var styles = getComputedStyle(document.documentElement);
    var canvas = styles.getPropertyValue('--bg-canvas').trim() || (theme === 'dark' ? '#06070B' : '#E6E8ED');
    var bgHover = styles.getPropertyValue('--bg-hover').trim();
    var muted = styles.getPropertyValue('--text-muted').trim();
    return {
      theme: theme,
      text:      styles.getPropertyValue('--text').trim(),
      muted:     muted,
      grid:      compositeOver(bgHover || muted, canvas),
      lineMuted: muted,
      elevated:  styles.getPropertyValue('--bg-elevated').trim(),
      accent:    styles.getPropertyValue('--accent').trim(),
    };
  }"""


def _legend_marker_html() -> str:
    """Static legend bar markup with per-model marker shapes."""
    items = []
    for m in MODELS:
        sym = MODEL_MARKERS[m]
        # Map plotly symbols to CSS classes.
        cls_map = {"circle": "mark-circle", "diamond": "mark-diamond",
                   "square": "mark-square", "cross": "mark-cross", "x": "mark-x"}
        cls = cls_map.get(sym, "mark-circle")
        label = m.replace("_", "-")
        items.append(
            f'    <span class="legend-item"><span class="legend-mark {cls}"></span>{label}</span>'
        )
    return "\n".join(items)


def wrap_procrustes(raw_path: Path, out_path: Path, overlay_scene: str = "scene6") -> None:
    raw = raw_path.read_text(encoding="utf-8")
    plotly_block = _extract_plotly_block(raw)

    legend_html = _legend_marker_html()

    out = f"""<html>
{_SHARED_HEAD}<style>
{_BASE_CSS}
</style></head>
<body>
<div id="caption-bar">
  <div class="legend-static">
{legend_html}
  </div>
  <span class="subtitle">all five models overlaid on gemma</span>
</div>
{plotly_block}
<script>
(function() {{
  var OVERLAY_SCENE = '{overlay_scene}';
  var div = null;
  var origData = null;
  var origLayout = null;

{_THEME_JS_HELPERS}

  function buildLayout(c) {{
    var origScene = origLayout[OVERLAY_SCENE] || origLayout.scene || {{}};
    var scene = JSON.parse(JSON.stringify(origScene));
    scene.bgcolor = 'rgba(0,0,0,0)';
    scene.domain = {{ x: [0, 1], y: [0, 1] }};
    scene.aspectmode = 'cube';
    scene.camera = {{ eye: {{ x: 1.5, y: 1.5, z: 1.35 }} }};
    ['xaxis', 'yaxis', 'zaxis'].forEach(function(ax) {{
      if (!scene[ax]) scene[ax] = {{}};
      scene[ax].gridcolor = c.grid;
      scene[ax].linecolor = c.lineMuted;
      scene[ax].zerolinecolor = c.lineMuted;
      scene[ax].color = c.text;
      scene[ax].showbackground = false;
      if (!scene[ax].tickfont) scene[ax].tickfont = {{}};
      scene[ax].tickfont.color = c.muted;
      if (typeof scene[ax].title === 'string') scene[ax].title = {{ text: scene[ax].title }};
      if (!scene[ax].title) scene[ax].title = {{}};
      if (!scene[ax].title.font) scene[ax].title.font = {{}};
      scene[ax].title.font.color = c.text;
    }});
    return {{
      scene: scene,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: {{ l: 0, r: 0, t: 0, b: 0 }},
      showlegend: false,
      autosize: true,
      font: {{ color: c.text, family: 'Recursive, ui-monospace, monospace' }},
    }};
  }}

  function render() {{
    if (!div || !origData || !window.Plotly) return;
    var c = readThemeColors();
    var traces = origData
      .filter(function (t) {{ return (t.scene || 'scene') === OVERLAY_SCENE; }})
      .map(function (t) {{
        var clone = JSON.parse(JSON.stringify(t));
        clone.scene = 'scene';
        if (clone.marker && clone.marker.line) {{
          clone.marker.line.color = c.elevated;
        }}
        return clone;
      }});
    Plotly.react(div, traces, buildLayout(c), {{ responsive: true, displayModeBar: false }});
    setTimeout(function() {{ Plotly.Plots.resize(div); }}, 50);
  }}

  function init() {{
    div = document.querySelector('.plotly-graph-div');
    if (!div || !div.data || !window.Plotly) {{
      return setTimeout(init, 50);
    }}
    origData = JSON.parse(JSON.stringify(div.data));
    origLayout = JSON.parse(JSON.stringify(div.layout));
    render();
    try {{
      if (window.parent && window.parent !== window) {{
        var obs = new MutationObserver(render);
        obs.observe(window.parent.document.documentElement, {{ attributes: true, attributeFilter: ['data-theme'] }});
      }}
    }} catch (e) {{}}
    window.addEventListener('resize', function() {{ if (window.Plotly && div) Plotly.Plots.resize(div); }});
  }}

  document.documentElement.dataset.theme = readTheme();
  window.addEventListener('load', function() {{ setTimeout(init, 100); }});
}})();
</script>
</body>
</html>
"""
    out_path.write_text(out, encoding="utf-8")
    print(f"  wrote {out_path}")


# Quadrant colors for the wild-faces legend bar. Pulled from
# llmoji_study.emotional_analysis.QUADRANT_COLORS so the static legend
# matches the per-face proportional-blend colors plotly draws.
WILD_QUADRANTS = ("HP", "LP", "HN-D", "HN-S", "LN", "NB")
WILD_QUADRANT_COLORS = {
    "HP":   "#998700",
    "LP":   "#009F68",
    "HN-D": "#DA534F",
    "HN-S": "#9769DC",
    "LN":   "#0091C9",
    "NB":   "#808696",
}
# Surface marker shapes match the plotly markers script 67 emits:
# circle = Claude Code journal only, diamond = any claude.ai export,
# square = neither (HF corpus only, from another contributor).
WILD_SURFACE_LEGEND = (
    ("circle",  "claude code"),
    ("diamond", "claude.ai"),
    ("square",  "other"),
)


def _wild_legend_html() -> str:
    """Static legend bar: 6 quadrant color swatches plus 3 surface-shape
    markers. Per-face plotly markers carry an RGB blend of these
    quadrant colors weighted by the BoL share, so the legend exposes
    the pure-quadrant reference even though no individual point lands
    exactly on it."""
    items = []
    for q in WILD_QUADRANTS:
        color = WILD_QUADRANT_COLORS[q]
        items.append(
            f'    <span class="legend-item">'
            f'<span class="legend-mark mark-circle" style="background: {color};"></span>'
            f'{q.lower()}</span>'
        )
    for shape, label in WILD_SURFACE_LEGEND:
        items.append(
            f'    <span class="legend-item">'
            f'<span class="legend-mark mark-{shape}"></span>'
            f'{label}</span>'
        )
    return "\n".join(items)


def wrap_wild_faces(raw_path: Path, out_path: Path) -> None:
    """Wrap the wild-faces PCA HTML keeping only the BoL-quadrant-by-
    surface scene (left, "scene"), dropping the KMeans cluster scene
    (right, "scene2"). Filters out the dummy legend-entry traces script
    67 emits with all-null coordinates for plotly's built-in legend
    since the wrapper renders a static HTML legend bar instead."""
    raw = raw_path.read_text(encoding="utf-8")
    plotly_block = _extract_plotly_block(raw)
    legend_html = _wild_legend_html()

    out = f"""<html>
{_SHARED_HEAD}<style>
{_BASE_CSS}
</style></head>
<body>
<div id="caption-bar">
  <div class="legend-static">
{legend_html}
  </div>
  <span class="subtitle">color blends per-face BoL shares · shape = deployment surface</span>
</div>
{plotly_block}
<script>
(function() {{
  var div = null;
  var origData = null;
  var origLayout = null;

{_THEME_JS_HELPERS}

  function buildLayout(c) {{
    var origScene = origLayout.scene || {{}};
    var scene = JSON.parse(JSON.stringify(origScene));
    scene.bgcolor = 'rgba(0,0,0,0)';
    scene.domain = {{ x: [0, 1], y: [0, 1] }};
    scene.aspectmode = 'cube';
    scene.camera = {{ eye: {{ x: 1.5, y: 1.5, z: 1.35 }} }};
    ['xaxis', 'yaxis', 'zaxis'].forEach(function(ax) {{
      if (!scene[ax]) scene[ax] = {{}};
      scene[ax].gridcolor = c.grid;
      scene[ax].linecolor = c.lineMuted;
      scene[ax].zerolinecolor = c.lineMuted;
      scene[ax].color = c.text;
      scene[ax].showbackground = false;
      if (!scene[ax].tickfont) scene[ax].tickfont = {{}};
      scene[ax].tickfont.color = c.muted;
      if (typeof scene[ax].title === 'string') scene[ax].title = {{ text: scene[ax].title }};
      if (!scene[ax].title) scene[ax].title = {{}};
      if (!scene[ax].title.font) scene[ax].title.font = {{}};
      scene[ax].title.font.color = c.text;
    }});
    return {{
      scene: scene,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: {{ l: 0, r: 0, t: 0, b: 0 }},
      showlegend: false,
      autosize: true,
      font: {{ color: c.text, family: 'Recursive, ui-monospace, monospace' }},
    }};
  }}

  function render() {{
    if (!div || !origData || !window.Plotly) return;
    var c = readThemeColors();
    // Keep only scene1 traces with real data. Excludes scene2 (KMeans
    // cluster scene we're dropping) and the dummy legend-entry traces
    // script 67 emits with all-null coords for plotly's built-in
    // legend, which we render as a static HTML bar instead. Filter on
    // legendgroup since recent plotly serializes large arrays as
    // {{bdata,dtype,shape}} objects rather than plain JS arrays — so
    // a t.x.length check rejects every real data trace too.
    var traces = origData
      .filter(function (t) {{ return (t.scene || 'scene') === 'scene'; }})
      .filter(function (t) {{
        return t.legendgroup && t.legendgroup.indexOf('left-data-') === 0;
      }})
      .map(function (t) {{
        var clone = JSON.parse(JSON.stringify(t));
        clone.scene = 'scene';
        if (clone.marker && clone.marker.line) {{
          clone.marker.line.color = c.elevated;
        }}
        return clone;
      }});
    Plotly.react(div, traces, buildLayout(c), {{ responsive: true, displayModeBar: false }});
    setTimeout(function() {{ Plotly.Plots.resize(div); }}, 50);
  }}

  function init() {{
    div = document.querySelector('.plotly-graph-div');
    if (!div || !div.data || !window.Plotly) {{
      return setTimeout(init, 50);
    }}
    origData = JSON.parse(JSON.stringify(div.data));
    origLayout = JSON.parse(JSON.stringify(div.layout));
    render();
    try {{
      if (window.parent && window.parent !== window) {{
        var obs = new MutationObserver(render);
        obs.observe(window.parent.document.documentElement, {{ attributes: true, attributeFilter: ['data-theme'] }});
      }}
    }} catch (e) {{}}
    window.addEventListener('resize', function() {{ if (window.Plotly && div) Plotly.Plots.resize(div); }});
  }}

  document.documentElement.dataset.theme = readTheme();
  window.addEventListener('load', function() {{ setTimeout(init, 100); }});
}})();
</script>
</body>
</html>
"""
    out_path.write_text(out, encoding="utf-8")
    print(f"  wrote {out_path}")


def wrap_per_face(raw_path: Path, out_path: Path, subtitles: list[str]) -> None:
    """Wrap the per-face PCA HTML with a model-toggle bar.

    The raw plotly is expected to have one scene per model in MODELS
    order (scene, scene2, ..., sceneN), each containing per-face markers
    in PC1×PC2×PC3.
    """
    raw = raw_path.read_text(encoding="utf-8")
    plotly_block = _extract_plotly_block(raw)

    toggle_buttons = []
    for i, m in enumerate(MODELS):
        active = " active" if i == 0 else ""
        label = m.replace("_", "-")
        toggle_buttons.append(
            f'    <button class="mode-btn{active}" data-model="{i}">{label}</button>'
        )
    toggle_html = "\n".join(toggle_buttons)
    subtitles_js = "[" + ", ".join(f'"{s}"' for s in subtitles) + "]"

    out = f"""<html>
{_SHARED_HEAD}<style>
{_BASE_CSS}
</style></head>
<body>
<div id="model-bar">
  <div class="mode-toggles" id="model-toggles">
{toggle_html}
  </div>
  <span class="subtitle" id="model-subtitle">{subtitles[0]}</span>
</div>
{plotly_block}
<script>
(function() {{
  var SUBTITLES = {subtitles_js};
  var SCENES = ['scene', 'scene2', 'scene3', 'scene4', 'scene5'];
  var activeModel = 0;
  var div = null;
  var origData = null;
  var origLayout = null;

{_THEME_JS_HELPERS}

  function buildLayout(idx, c) {{
    var sceneKey = SCENES[idx];
    var origScene = origLayout[sceneKey] || origLayout.scene || {{}};
    var scene = JSON.parse(JSON.stringify(origScene));
    scene.bgcolor = 'rgba(0,0,0,0)';
    scene.domain = {{ x: [0, 1], y: [0, 1] }};
    scene.aspectmode = 'cube';
    scene.camera = {{ eye: {{ x: 1.5, y: 1.5, z: 1.35 }} }};
    ['xaxis', 'yaxis', 'zaxis'].forEach(function(ax) {{
      if (!scene[ax]) scene[ax] = {{}};
      scene[ax].gridcolor = c.grid;
      scene[ax].linecolor = c.lineMuted;
      scene[ax].zerolinecolor = c.lineMuted;
      scene[ax].color = c.text;
      scene[ax].showbackground = false;
      if (!scene[ax].tickfont) scene[ax].tickfont = {{}};
      scene[ax].tickfont.color = c.muted;
      if (typeof scene[ax].title === 'string') scene[ax].title = {{ text: scene[ax].title }};
      if (!scene[ax].title) scene[ax].title = {{}};
      if (!scene[ax].title.font) scene[ax].title.font = {{}};
      scene[ax].title.font.color = c.text;
    }});
    return {{
      scene: scene,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: {{ l: 0, r: 0, t: 0, b: 0 }},
      showlegend: false,
      autosize: true,
      font: {{ color: c.text, family: 'Recursive, ui-monospace, monospace' }},
    }};
  }}

  function showModel(idx) {{
    activeModel = idx;
    document.getElementById('model-subtitle').textContent = SUBTITLES[idx];
    if (!div || !origData || !window.Plotly) return;
    var c = readThemeColors();
    var sceneKey = SCENES[idx];
    var traces = origData
      .filter(function (t) {{ return (t.scene || 'scene') === sceneKey; }})
      .map(function (t) {{
        var clone = JSON.parse(JSON.stringify(t));
        clone.scene = 'scene';
        if (!clone.marker) clone.marker = {{}};
        if (!clone.marker.line) clone.marker.line = {{}};
        clone.marker.line.color = c.elevated;
        clone.marker.line.width = 0.5;
        return clone;
      }});
    Plotly.react(div, traces, buildLayout(idx, c), {{ responsive: true, displayModeBar: false }});
    setTimeout(function() {{ Plotly.Plots.resize(div); }}, 50);
  }}

  function init() {{
    div = document.querySelector('.plotly-graph-div');
    var toggles = document.getElementById('model-toggles');
    if (!div || !div.data || !window.Plotly || !toggles || typeof _forms === 'undefined') {{
      return setTimeout(init, 50);
    }}
    origData = JSON.parse(JSON.stringify(div.data));
    origLayout = JSON.parse(JSON.stringify(div.layout));
    _forms.bindModeGroup(toggles, 'model', function (val) {{
      showModel(parseInt(val, 10));
    }});
    showModel(0);
    try {{
      if (window.parent && window.parent !== window) {{
        var obs = new MutationObserver(function() {{ showModel(activeModel); }});
        obs.observe(window.parent.document.documentElement, {{ attributes: true, attributeFilter: ['data-theme'] }});
      }}
    }} catch (e) {{}}
    window.addEventListener('resize', function() {{ if (window.Plotly && div) Plotly.Plots.resize(div); }});
  }}

  document.documentElement.dataset.theme = readTheme();
  window.addEventListener('load', function() {{ setTimeout(init, 100); }});
}})();
</script>
</body>
</html>
"""
    out_path.write_text(out, encoding="utf-8")
    print(f"  wrote {out_path}")


def main() -> None:
    if not BLOG_ASSETS_DIR.parent.exists():
        print(f"blog-assets parent does not exist: {BLOG_ASSETS_DIR.parent}", file=sys.stderr)
        sys.exit(1)
    BLOG_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== procrustes overlay ===")
    if not RAW_PROCRUSTES.exists():
        print(f"  missing {RAW_PROCRUSTES} — skipping", file=sys.stderr)
    else:
        wrap_procrustes(RAW_PROCRUSTES, OUT_PROCRUSTES, overlay_scene="scene6")

    print("\n=== per-face PCA toggle ===")
    if not RAW_PER_FACE.exists():
        print(f"  missing {RAW_PER_FACE} — generate it first via scripts/local/97_build_per_face_pca_3d.py")
    else:
        # Subtitles read off the per-face JSON sidecar written by the
        # generator; default fallback is generic.
        sidecar = RAW_PER_FACE.with_suffix(".meta.json")
        if sidecar.exists():
            import json
            meta = json.loads(sidecar.read_text())
            subtitles = [meta[m] for m in MODELS]
        else:
            subtitles = [f"{m} per-face centroids" for m in MODELS]
        wrap_per_face(RAW_PER_FACE, OUT_PER_FACE, subtitles)

    print("\n=== wild faces PCA (BoL quadrant by deployment surface) ===")
    if not RAW_WILD_FACES.exists():
        print(f"  missing {RAW_WILD_FACES} — generate via scripts/67_wild_residual.py", file=sys.stderr)
    else:
        wrap_wild_faces(RAW_WILD_FACES, OUT_WILD_FACES)


if __name__ == "__main__":
    main()
