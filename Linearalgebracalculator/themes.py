"""
themes.py
---------
Colour themes for the Linear Algebra Calculator, plus a tiny settings file
so the chosen theme and motion options are remembered between runs.

Every theme is a plain dict of colour roles. The rest of the app only ever
asks for a role ("accent", "kv", ...), never a hard-coded colour.
"""

import json
from pathlib import Path

SETTINGS_FILE = Path.home() / ".linear_algebra_calculator.json"

THEMES = {
    "Classic Light": dict(
        dark=False,
        bg="#eef1f6", panel="#ffffff", field="#ffffff", border="#e5e7eb",
        button="#e5e7eb", text="#1f2937", muted="#6b7280",
        accent="#2563eb", accent_fg="#ffffff", accent_hover="#1d4ed8",
        accent_disabled="#93c5fd", soft="#dbeafe",
        v="#2563eb", kv="#dc2626", span="#9ca3af", w="#059669", proj="#d97706", axis="#374151", grid="#9ca3af",
        plot_bg="#ffffff", result_bg="#fef2f2",
        status_bg="#e5e7eb", error_bg="#fee2e2", error_fg="#991b1b",
        header=("#111827", "#1e3a8a"), header_fg="#ffffff", header_sub="#cbd5e1",
        code_bg="#0f172a", code_fg="#e2e8f0", out_bg="#111827", out_fg="#86efac",
    ),
    "Midnight": dict(
        dark=True,
        bg="#0b1220", panel="#111a2e", field="#0b1220", border="#22304d",
        button="#1b2742", text="#e2e8f0", muted="#94a3b8",
        accent="#60a5fa", accent_fg="#0b1220", accent_hover="#93c5fd",
        accent_disabled="#1e3a5f", soft="#1e2d4d",
        v="#60a5fa", kv="#f87171", w="#34d399", proj="#fbbf24", span="#64748b", axis="#94a3b8", grid="#334155",
        plot_bg="#111a2e", result_bg="#2a1a24",
        status_bg="#0b1220", error_bg="#3f1d24", error_fg="#fecaca",
        header=("#020617", "#1e3a8a"), header_fg="#ffffff", header_sub="#94a3b8",
        code_bg="#0a0f1c", code_fg="#e2e8f0", out_bg="#060a14", out_fg="#86efac",
    ),
    "Forest": dict(
        dark=False,
        bg="#eef3ee", panel="#ffffff", field="#ffffff", border="#dbe5dc",
        button="#dfe9e1", text="#1c2a21", muted="#5f6f64",
        accent="#15803d", accent_fg="#ffffff", accent_hover="#166534",
        accent_disabled="#86c59a", soft="#dcfce7",
        v="#15803d", kv="#c2410c", w="#2563eb", proj="#a16207", span="#9aa89e", axis="#2f3d33", grid="#9aa89e",
        plot_bg="#ffffff", result_bg="#fff4ed",
        status_bg="#dbe5dc", error_bg="#fee2e2", error_fg="#991b1b",
        header=("#0f1f14", "#166534"), header_fg="#ffffff", header_sub="#bbd5c2",
        code_bg="#0f1a14", code_fg="#e2ece5", out_bg="#0b140f", out_fg="#86efac",
    ),
    "Sepia": dict(
        dark=False,
        bg="#f3ede2", panel="#fbf8f2", field="#fffdf8", border="#e4d9c6",
        button="#ebe1cf", text="#3b2f22", muted="#7c6a55",
        accent="#b45309", accent_fg="#ffffff", accent_hover="#92400e",
        accent_disabled="#e0b98a", soft="#f6e7cf",
        v="#2f5d8a", kv="#b91c1c", w="#3f7d3a", proj="#7c3aed", span="#a8987f", axis="#4a3b2a", grid="#a8987f",
        plot_bg="#fbf8f2", result_bg="#fbeee0",
        status_bg="#e9dfcd", error_bg="#fde2d8", error_fg="#8a1c1c",
        header=("#2b2116", "#7c4a12"), header_fg="#fffaf0", header_sub="#e7d6bd",
        code_bg="#241c14", code_fg="#f1e7d8", out_bg="#1b150f", out_fg="#d9f99d",
    ),
    "Violet Night": dict(
        dark=True,
        bg="#16121f", panel="#1f1a2e", field="#16121f", border="#352c4d",
        button="#2b2440", text="#ede9fe", muted="#a59bc4",
        accent="#a78bfa", accent_fg="#16121f", accent_hover="#c4b5fd",
        accent_disabled="#3f3566", soft="#2e2645",
        v="#a78bfa", kv="#f472b6", w="#34d399", proj="#fbbf24", span="#6b6390", axis="#a59bc4", grid="#3f3566",
        plot_bg="#1f1a2e", result_bg="#2d1b2e",
        status_bg="#16121f", error_bg="#3f1d2e", error_fg="#fbcfe8",
        header=("#0d0a14", "#4c1d95"), header_fg="#ffffff", header_sub="#c4b5fd",
        code_bg="#120e1b", code_fg="#ede9fe", out_bg="#0d0a14", out_fg="#86efac",
    ),
}

DEFAULT = "Classic Light"

# Settings ▸ Motion: multiplier for every animation duration (0 = no animation)
MOTION = {"Relaxed": 1.4, "Normal": 1.0, "Snappy": 0.6, "Off": 0.0}

DEFAULT_SETTINGS = {"theme": DEFAULT, "motion": "Normal", "splash": True}


def get(name):
    """Theme dict by name (with its name included); falls back to the default."""
    name = name if name in THEMES else DEFAULT
    return dict(THEMES[name], name=name)


def load_settings():
    settings = dict(DEFAULT_SETTINGS)
    try:
        saved = json.loads(SETTINGS_FILE.read_text("utf-8"))
        settings.update({k: v for k, v in saved.items() if k in settings})
    except (OSError, ValueError, AttributeError):
        pass
    if settings["theme"] not in THEMES:
        settings["theme"] = DEFAULT
    if settings["motion"] not in MOTION:
        settings["motion"] = "Normal"
    return settings


def save_settings(settings):
    try:
        SETTINGS_FILE.write_text(json.dumps(settings, indent=1), "utf-8")
    except OSError:
        pass                     # not being able to save a preference is harmless


def mix(c1, c2, t):
    """Blend two #rrggbb colours (t = 0 -> c1, t = 1 -> c2)."""
    t = max(0.0, min(1.0, t))
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))
