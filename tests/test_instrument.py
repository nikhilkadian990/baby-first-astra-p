"""CPU unit and integration tests. No empirical developmental success is asserted."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from baby_p.agents.predictive_agent import PredictiveAgent, CoveragePolicy
from baby_p.analysis.metrics import adaptation_curves, mean_interval, analyze
from baby_p.analysis.plots import make_plots
from baby_p.env.generator import appearance, FAMILIES
from baby_p.env.world import World
from baby_p.experiments.ablations import VARIANTS, variant_config
from baby_p.experiments.transfer import contact_probe, probe_suite
from baby_p.memory.replay_buffer import ReplayBuffer, fragment
from baby_p.training.common import (load_config, validate, OnlineStream, save_checkpoint,
                                    load_checkpoint, derived_seed, window_length)
from baby_p.training.evaluate import frozen_evaluation, evaluate_checkpoint
from baby_p.training.train import train

BASE = Path(__file__).resolve().parents[1] / 'baby_p/config/base.yaml'


def tiny_config(root=None):
    config = load_config(BASE)
    config['seeds'] = [7]
    config['model'].update(latent=8, hidden=12, horizons=[1, 2], rollout_horizon=2)
    config['world'].update(grid=7, objects=3, episode_steps=8)
    config['learning'].update(burn_in=1, unroll=2, batch_size=2, update_every=2)
    config['memory'].update(capacity=3, store_every=2)
    config['training'].update(interactions=16, checkpoints=[4, 8, 16], shift_at=12)
    config['evaluation'].update(budgets=[0, 4, 8], probes=2, horizons=[1, 2],
                                families=['swapped_roles', 'history_alias'], bootstrap_samples=40)
    if root:
        config['output'] = str(Path(root) / 'results')
        config['checkpoint_dir'] = str(Path(root) / 'checkpoints')
    return validate(config)


class WorldTests(unittest.TestCase):
    def test_pixel_interface_and_determinism(self):
        config = tiny_config()
        a, b = World(config['world'], 11), World(config['world'], 11)
        self.assertEqual(a.observe().dtype, np.uint8)
        self.assertEqual(a.observe().shape, (3, 21, 21))
        for action in [4, 4, 0, 2, 1, 3] * 5:
            np.testing.assert_array_equal(a.step(action), b.step(action))
            positions = [tuple(o['position']) for o in a.objects] + [tuple(a.agent)]
            self.assertEqual(len(set(positions)), len(positions))

    def test_push_block_and_identity_invariance(self):
        config = tiny_config()
        a = World(config['world'], 11)
        a.objects = [dict(position=[3, 3], role=1, color=0, shape=0, velocity=[0, 0], identity=123)]
        a.agent[:] = [2, 3]
        b = a.clone()
        b.objects[0]['role'] = 0
        np.testing.assert_array_equal(a.observe(), b.observe())
        a.step(4)
        b.step(4)
        self.assertEqual(a.objects[0]['position'], [4, 3])
        self.assertEqual(b.objects[0]['position'], [3, 3])
        c = a.clone()
        c.objects[0]['identity'] = 999
        np.testing.assert_array_equal(a.step(0), c.step(0))

    def test_split_roles_appearances(self):
        for seed in range(12):
            train_color = appearance(1, 'train', np.random.default_rng(seed))
            self.assertIn(train_color, [0, 2])
            self.assertEqual(appearance(1, 'swapped_roles', np.random.default_rng(seed)), 1)
            self.assertIn(appearance(0, 'new_appearance', np.random.default_rng(seed)), [3, 4, 5])
        self.assertNotEqual(derived_seed(7, 'development', 0), derived_seed(7, 'adaptation', 0))

    def test_all_contact_families_valid(self):
        config = tiny_config()
        for family in FAMILIES:
            for index in range(4):
                probe = contact_probe(config, 7, family, index)
                for actual, alternative in probe['candidates'].values():
                    self.assertFalse(np.array_equal(actual, alternative))

    def test_alias_has_equal_present_different_history(self):
        config = tiny_config()
        a, b = [contact_probe(config, 7, 'history_alias', i) for i in (0, 1)]
        np.testing.assert_array_equal(a['context_observations'][-1], b['context_observations'][-1])
        self.assertEqual(a['context_actions'], b['context_actions'])
        self.assertFalse(np.array_equal(a['context_observations'][0], b['context_observations'][0]))
        np.testing.assert_array_equal(a['candidates'][1][0], b['candidates'][1][1])


class LearningTests(unittest.TestCase):
    def test_bounded_fifo_original_evidence(self):
        memory = ReplayBuffer(2)
        obs = [np.zeros((3, 9, 9), dtype=np.uint8)] * 3
        item = fragment(obs, [0, 1], 1, 0, 0, 0)
        memory.add(item)
        item['observations'][:] = 255
        self.assertEqual(memory.items[0]['observations'].max(), 0)
        for seed in (2, 3):
            memory.add(fragment(obs, [0, 1], seed, 0, 0, 0))
        self.assertEqual([i['metadata'][0] for i in memory.items], [2, 3])
        self.assertGreater(memory.bytes, 0)
        zero = ReplayBuffer(0)
        zero.add(item)
        self.assertEqual(len(zero.items), 0)

    def test_state_ablations_action_conditioning_and_parameter_match(self):
        config = tiny_config()
        recurrent = PredictiveAgent(config, 7)
        for variant in ('no_history', 'reactive'):
            agent = PredictiveAgent(variant_config(config, variant), 7)
            z = torch.ones(1, config['model']['latent'])
            a = torch.tensor([0])
            h = torch.ones(1, config['model']['hidden'])
            torch.testing.assert_close(agent.state(z, a, h), agent.state(z, a, h * 2))
            counts = [sum(p.numel() for p in model.parameters_for_learning()) for model in (recurrent, agent)]
            self.assertLess(abs(counts[0] - counts[1]) / counts[0], 0.02)
        state = torch.zeros(1, config['model']['hidden'])
        self.assertFalse(torch.allclose(recurrent.predictor(state, torch.tensor([[0]]))[1],
                                        recurrent.predictor(state, torch.tensor([[4]]))[1]))

    def test_alias_memoryless_chance(self):
        config = tiny_config()
        for variant in ('no_history', 'reactive'):
            agent = PredictiveAgent(variant_config(config, variant), 7)
            rows = probe_suite(agent, config, 7, 'history_alias')
            for horizon in config['evaluation']['horizons']:
                self.assertEqual(np.mean([r['accuracy'] for r in rows if r['horizon'] == horizon]), 0.5)

    def test_updates_bounded_no_cross_episode_and_equal_streams(self):
        config = tiny_config()
        events = []
        for variant in VARIANTS:
            cfg = variant_config(config, variant)
            agent = PredictiveAgent(cfg, 7)
            stream = OnlineStream(cfg, 7)
            sequence = []
            for _ in range(16):
                event = stream.advance(agent)
                sequence.append((event['environment_seed'], event['action']))
            events.append(sequence)
            self.assertLessEqual(agent.stats['updates'], 8)
            self.assertGreater(agent.stats['updates'], 0)
            for item in agent.memory.items:
                self.assertGreaterEqual(item['metadata'][2], 0)
                self.assertLessEqual(item['metadata'][2] + len(item['actions']), 8)
            self.assertTrue(all(p.grad is None for p in agent.target.parameters()))
            self.assertTrue(all(torch.isfinite(p).all() for p in agent.parameters()))
        self.assertTrue(all(seq == events[0] for seq in events))

    def test_resume_exact_model_and_evidence(self):
        config = tiny_config()
        agent = PredictiveAgent(config, 7)
        stream = OnlineStream(config, 7)
        for _ in range(7):
            stream.advance(agent)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            save_checkpoint(path, agent, stream, 'early', 'p')
            expected = [stream.advance(agent) for _ in range(5)]
            payload = load_checkpoint(path)
            resumed = PredictiveAgent.from_snapshot(payload['agent'])
            actual = [payload['stream'].advance(resumed) for _ in range(5)]
            self.assertEqual(expected, actual)
            for key, value in agent.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(agent.stats, resumed.stats)
            self.assertEqual(agent.memory.bytes, resumed.memory.bytes)

    def test_frozen_evaluation_and_adaptation_reset(self):
        config = tiny_config()
        agent = PredictiveAgent(config, 7)
        stream = OnlineStream(config, 7)
        for _ in range(8):
            stream.advance(agent)
        before = deepcopy(agent.snapshot())
        frozen_evaluation(agent, config, 7, 'history_alias')
        for key, value in before['model'].items():
            torch.testing.assert_close(value, agent.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(before['stats'], agent.stats)
        torch.testing.assert_close(before['live_h'], agent.live_h)
        adapted = PredictiveAgent.from_snapshot(before, adaptation=True)
        self.assertEqual(len(adapted.memory.items), 0)
        self.assertEqual(len(adapted.optimizer.state), 0)
        self.assertIsNone(adapted.live_h)
        self.assertEqual(adapted.stats['updates'], 0)


class MetricsAndIntegrationTests(unittest.TestCase):
    def test_censoring_and_seed_uncertainty(self):
        common = dict(variant='p', seed=7, checkpoint='late', age=16, adaptation_family='swapped_roles',
                      horizon=1, task='contact_discrimination', evaluation_kind='transfer')
        rows = [dict(common, budget=x, accuracy=y) for x, y in [(0, 0.5), (4, 1.0), (8, 0.5)]]
        curve = adaptation_curves(rows, 0.75)[0]
        self.assertIsNone(curve['criterion_cost'])
        self.assertTrue(curve['censored'])
        self.assertEqual(curve['restricted_cost'], 8)
        self.assertIsNone(mean_interval([1])['low'])
        self.assertEqual(mean_interval([1, 1])['low'], 1)

    def test_config_rejects_invalid_horizons(self):
        config = tiny_config()
        config['model']['horizons'] = [3]
        with self.assertRaises(ValueError):
            validate(config)

    def test_local_train_evaluate_analyze_plot_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            config = tiny_config(directory)
            paths = train(config, 7)
            self.assertIn(0, paths)
            self.assertIn(16, paths)
            for path in paths.values():
                evaluate_checkpoint(path)
            summaries, curves = analyze(config['output'], config)
            self.assertTrue(summaries and curves)
            make_plots(config['output'])
            root = Path(config['output'])
            self.assertTrue((root / 'report.md').exists())
            self.assertTrue((root / 'plots/08_horizons.png').exists())
            self.assertTrue((root / 'tables/adaptation.csv').exists())


if __name__ == '__main__':
    unittest.main()
