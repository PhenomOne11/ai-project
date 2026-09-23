"""
`neuralstream config` — Inspect / validate / dump configuration.

Sub-actions:
    show       Pretty-print the current config summary
    validate   Validate a profile against the schema
    dump       Dump the full config as YAML or JSON
"""

import sys
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_config(args) -> int:
    action = getattr(args, 'config_action', 'show')

    if action == 'show':
        return _action_show(args)
    elif action == 'validate':
        return _action_validate(args)
    elif action == 'dump':
        return _action_dump(args)
    else:
        print(f"Unknown config action: {action}", file=sys.stderr)
        return 2


def _action_show(args) -> int:
    from config.loader import load_config, summarize_config

    profile = getattr(args, 'profile', None)
    cfg = load_config(profile=profile)
    summary = summarize_config(cfg)

    print("=" * 68)
    print(f"  Config: {profile or 'default'}")
    print("=" * 68)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _action_validate(args) -> int:
    from config.loader import list_profiles, resolve_profile, load_yaml
    from config.schema import SystemConfig
    from pydantic import ValidationError

    profile = getattr(args, 'profile', None)
    targets = [profile] if profile else list_profiles()

    ok_count = 0
    fail_count = 0

    for name in targets:
        try:
            path = resolve_profile(name)
            if path is None:
                print(f"  ✗ {name:<24} not found")
                fail_count += 1
                continue

            data = load_yaml(str(path))
            cfg = SystemConfig(**data)
            print(f"  ✓ {name:<24} valid ({len(data)} top-level keys)")
            ok_count += 1
        except ValidationError as e:
            print(f"  ✗ {name:<24} validation error:")
            for err in e.errors():
                loc = ".".join(str(x) for x in err['loc'])
                print(f"        {loc}: {err['msg']}")
            fail_count += 1
        except Exception as e:
            print(f"  ✗ {name:<24} {type(e).__name__}: {e}")
            fail_count += 1

    print()
    print(f"  Result: {ok_count} valid, {fail_count} failed")
    return 0 if fail_count == 0 else 1


def _action_dump(args) -> int:
    from config.loader import load_config
    import yaml

    profile = getattr(args, 'profile', None)
    fmt = getattr(args, 'format', 'yaml')
    cfg = load_config(profile=profile)

    data = cfg.model_dump(mode='json')

    if fmt == 'json':
        print(json.dumps(data, indent=2, default=str))
    else:
        print(yaml.dump(data, default_flow_style=False, sort_keys=False))

    return 0