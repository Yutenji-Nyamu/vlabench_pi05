"""Official evaluator + durable per-episode records and a post-action final frame.

Observation/action transforms, IK, task success, horizon and replan=5 come from
the model-card's pinned OpenPI branch and VLABench's native evaluator.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--openpi-root', type=Path, required=True)
    p.add_argument('--track', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=50)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    save(a.output / 'progress.json', {'time': time.time(), 'phase': 'importing'})
    import VLABench
    from VLABench.evaluation.evaluator import base
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    spec = importlib.util.spec_from_file_location('official_pi05_eval', a.openpi_root / 'examples/vlabench/eval.py')
    official = importlib.util.module_from_spec(spec)
    # dataclasses resolves the defining module through sys.modules.
    import sys
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    track_file = Path(VLABench.__file__).parent / 'configs/evaluation/tracks' / (a.track + '.json')
    configs = json.loads(track_file.read_text())
    assert a.task in configs
    assert 0 < a.count <= len(configs[a.task])
    random.seed(7)
    np.random.seed(7)
    native_load = base.load_env

    class Recorder(base.Evaluator):
        def load_env(self, *args, **kwargs):
            env = native_load(*args, **kwargs)
            self.active_env = env
            self.closed = False
            observation = env.get_observation
            close = env.close
            step = env.step

            def observed(*args, **kwargs):
                result = observation(*args, **kwargs)
                save(a.output / 'progress.json', {'time': time.time(), 'phase': 'acting',
                     'episode': self.episode, 'actions': self.actions})
                return result

            def stepped(action=None):
                value = step(action)
                if action is not None:
                    self.actions += 1
                return value

            def closed():
                if self.closed:
                    return
                try:
                    # Native code closes before saving its pre-action frames.
                    # Capture the terminal observation while physics is alive.
                    self.final_rgb = observation(require_pcd=False)['rgb']
                finally:
                    self.closed = True
                    close()

            env.get_observation = observed
            env.step = stepped
            env.close = closed
            return env

        def evaluate_single_episode(self, agent, task_name, episode_id, episode_config, **kwargs):
            self.episode, self.actions = episode_id, 0
            self.active_env, self.final_rgb, self.closed = None, None, False
            self.video_frames = None
            self.consecutive_errors = getattr(self, 'consecutive_errors', 0)
            start = time.time()
            base.load_env = self.load_env
            save(a.output / 'progress.json', {'time': start, 'phase': 'initializing', 'episode': episode_id})
            try:
                result = super().evaluate_single_episode(agent, task_name, episode_id, episode_config, **kwargs)
                result.update(episode_id=episode_id, track=a.track, elapsed_seconds=time.time()-start,
                              physical_actions=self.actions, video_frames=self.video_frames,
                              final_frame_saved=self.final_rgb is not None)
                save(a.output / f'episode-{episode_id:04d}.json', result)
                self.consecutive_errors = 0
                return result
            except Exception:
                self.consecutive_errors += 1
                save(a.output / f'error-{episode_id:04d}.json', {'episode_id': episode_id,
                     'track': a.track, 'task': a.task, 'elapsed_seconds': time.time()-start,
                     'error': traceback.format_exc()})
                if self.consecutive_errors >= 3:
                    raise SystemExit('Three consecutive infrastructure errors; skip remaining task configurations.')
                raise
            finally:
                base.load_env = native_load
                if self.active_env is not None and not self.closed:
                    self.active_env.close()

        def save_video(self, frames, save_dir):
            if self.final_rgb is not None:
                frames.append(self.final_rgb)
            self.video_frames = len(frames)
            super().save_video(frames, save_dir)
            if self.final_rgb is not None:
                from PIL import Image
                mosaic = np.vstack([np.hstack(self.final_rgb[:2]), np.hstack(self.final_rgb[2:4])])
                Image.fromarray(mosaic).save(Path(save_dir).with_suffix('.last.jpg'))

    client = WebsocketClientPolicy('127.0.0.1', a.port)
    policy = official.Pi0(client=client, replan_steps=5)
    evaluator = Recorder(tasks=[a.task], n_episodes=a.count, episode_config=configs,
                         max_substeps=1, save_dir=str(a.output), visulization=True,
                         metrics=['success_rate', 'intention_score', 'progress_score'])
    results = evaluator.evaluate(policy)
    records = sorted(a.output.glob('episode-*.json'))
    errors = sorted(a.output.glob('error-*.json'))
    save(a.output / 'complete.json', {'time': time.time(), 'track': a.track, 'task': a.task,
         'requested': a.count, 'completed': len(records), 'errors': len(errors), 'metrics': results})


if __name__ == '__main__':
    main()
