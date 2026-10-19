"""Localhost-only entry to the model-card's unmodified pi05 policy factory."""
import argparse
import logging
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--openpi-root', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--port', type=int, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.openpi_root / 'src'))
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

logging.basicConfig(level=logging.INFO, force=True)
policy = create_trained_policy(get_config('pi05_ft_vlabench_primitive'), a.checkpoint)
WebsocketPolicyServer(policy, host='127.0.0.1', port=a.port, metadata=policy.metadata).serve_forever()
