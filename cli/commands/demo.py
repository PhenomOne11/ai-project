"""
`neuralstream demo` — Launch the interactive dashboard.
"""

import sys
import subprocess
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_demo(args) -> int:
    profile = getattr(args, 'profile', None)
    port = getattr(args, 'port', 8501)
    app_path = getattr(args, 'app', None)

    project_root = Path(__file__).resolve().parent.parent.parent

    # Determine app to launch
    if app_path is None:
        # Prefer root app.py if it exists, otherwise the panel preview
        main_app = project_root / "app.py"
        preview_app = project_root / "dashboard" / "preview_panels.py"
        if main_app.is_file():
            target = main_app
        elif preview_app.is_file():
            target = preview_app
        else:
            print("No dashboard app found.", file=sys.stderr)
            return 1
    else:
        target = Path(app_path)
        if not target.is_file():
            print(f"App not found: {target}", file=sys.stderr)
            return 1

    # Pass the profile via environment variable so the app can pick it up
    import os
    env = dict(os.environ)
    if profile:
        env['NEURALSTREAM_PROFILE'] = profile

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(target),
        "--server.port", str(port),
    ]

    print(f"[demo] launching Streamlit: {target.name}")
    print(f"[demo] profile: {profile or 'default'}")
    print(f"[demo] URL:     http://localhost:{port}")
    print(f"[demo] (Ctrl+C to stop)")

    try:
        return subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        print("\n[demo] stopped")
        return 0