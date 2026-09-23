"""
CLI Command Registry
"""

from cli.commands.info import run_info
from cli.commands.config_cmd import run_config
from cli.commands.demo import run_demo
from cli.commands.bench import run_bench
from cli.commands.verify import run_verify

COMMANDS = {
    'info': run_info,
    'config': run_config,
    'demo': run_demo,
    'bench': run_bench,
    'verify': run_verify,
}

__all__ = [
    'COMMANDS',
    'run_info', 'run_config', 'run_demo', 'run_bench', 'run_verify',
]