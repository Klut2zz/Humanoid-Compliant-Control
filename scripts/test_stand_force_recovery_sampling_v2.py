#new change
"""CPU-only V2 regression tests; no simulation and no optimizer updates."""
import ast
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import stand_force_recovery_sampling as v1
import stand_force_recovery_sampling_v2 as v2


class RecoveryV2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6)
        self.ids = torch.arange(4096)
        self.events = self.ids % 13
        direction = torch.randn(4096, 3)
        magnitude = torch.linspace(2., 4., 4096)[:, None]
        self.peaks = direction / direction.norm(dim=-1, keepdim=True) * magnitude
        self.stiffness = torch.full((4096, 1), 200.)

    def transform(self, module, mode='mixed', ids=None, events=None, peaks=None):
        return module.transform_peaks(
            self.peaks if peaks is None else peaks, self.stiffness, .1,
            self.ids if ids is None else ids,
            self.events if events is None else events, 42, mode)

    def test_only_cohort_allocation_changes(self):
        # Implementation of event transformation must be exactly V1's.
        self.assertEqual(inspect.getsource(v1.transform_peaks), inspect.getsource(v2.transform_peaks))
        self.assertEqual(inspect.getsource(v1.rear_draw), inspect.getsource(v2.rear_draw))
        a = v1.curriculum_cohort(self.ids, 'mixed')
        b = v2.curriculum_cohort(self.ids, 'mixed')
        self.assertEqual(torch.bincount(a).tolist(), [1024, 2048, 1024])
        self.assertEqual(torch.bincount(b).tolist(), [1024, 1024, 2048])
        self.assertTrue(torch.equal(a == 0, b == 0))
        self.assertTrue(torch.equal(a != b, self.ids % 4 == 2))

    def test_existing_replay_and_hard_events_unchanged(self):
        a, ar = self.transform(v1)
        b, br = self.transform(v2)
        unchanged = self.ids % 4 != 2
        self.assertTrue(torch.equal(a[unchanged], b[unchanged]))
        self.assertTrue(torch.equal(ar[unchanged], br[unchanged]))
        cohort = v2.curriculum_cohort(self.ids, 'mixed')
        self.assertEqual(int(torch.count_nonzero(b[cohort == 0])), 0)
        self.assertTrue(torch.equal(b[cohort == 1], self.peaks[cohort == 1]))
        desired = 1.5*self.peaks[cohort == 2].norm(dim=-1)+1
        torch.testing.assert_close(b[cohort == 2].norm(dim=-1), desired)
        self.assertTrue(.45 < float(br[cohort == 2].float().mean()) < .55)
        torch.testing.assert_close(b[br, 2]/b[br].norm(dim=-1),
                                   self.peaks[br, 2]/self.peaks[br].norm(dim=-1))
        self.assertTrue(torch.all(b[br, 0] < 0))
        self.assertTrue(torch.all(torch.atan2(b[br, 1], -b[br, 0]).abs() <= torch.pi/6+1e-6))

    def test_pure_evaluation_is_bit_identical(self):
        for mode in ('zero', 'low', 'medium'):
            a, ar = self.transform(v1, mode)
            b, br = self.transform(v2, mode)
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(ar, br))
            self.assertFalse(br.any())

    def test_no_rng_consumption_and_independent_resets(self):
        before = torch.get_rng_state().clone()
        a, ar = self.transform(v2)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        events = self.events.clone(); events[2] += 17
        b, br = self.transform(v2, events=events)
        keep = self.ids != 2
        self.assertTrue(torch.equal(a[keep], b[keep]))
        self.assertTrue(torch.equal(ar[keep], br[keep]))
        order = torch.arange(4095, -1, -1)
        b, br = self.transform(v2, ids=self.ids[order], events=self.events[order], peaks=self.peaks[order])
        self.assertTrue(torch.equal(a[order], b))
        self.assertTrue(torch.equal(ar[order], br))

    def test_displacement_cap_preserved(self):
        peaks = torch.tensor([[2., 0., 0.], [0., 4., 0.]])
        ids = torch.arange(2); events = torch.zeros(2, dtype=torch.long)
        for module in (v1, v2):
            force, _ = module.transform_peaks(peaks, torch.full((2, 1), 50.), .1, ids, events, 7, 'medium')
            torch.testing.assert_close(force.norm(dim=-1), torch.tensor([4., 5.]))

    def test_checkpoint_names_count_completed_updates(self):
        # Execute only the save class AST; importing the entry would launch Isaac.
        script = Path(__file__).with_name('train_stand_force_recovery_mix_v2.py')
        tree = ast.parse(script.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RecoveryRunner')
        namespace = dict(OnPolicyRunner=object, torch=torch, SOURCE=Path('/source-v1/model_50.pt'))
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(script), 'exec'), namespace)
        runner = namespace['RecoveryRunner']()
        runner.alg = SimpleNamespace(save=lambda: {'actor_state_dict': {'example': torch.zeros(1)}})
        with tempfile.TemporaryDirectory(prefix='recovery-v2-save-') as tmp:
            folder = Path(tmp)
            runner.save(folder/'disabled_base_schedule.pt')
            self.assertFalse((folder/'disabled_base_schedule.pt').exists())
            for count in (0, 10, 20, 30, 40, 50):
                runner.save_completed(folder, count)
                name = 'model_initial.pt' if count == 0 else f'model_{count}.pt'
                saved = torch.load(folder/name, weights_only=True)
                self.assertEqual(saved['iter'], count)
                self.assertEqual(saved['infos']['source_iteration'], 50)
                self.assertEqual(saved['infos']['completed_branch_updates'], count)
            with self.assertRaises(AssertionError):
                runner.save_completed(folder, 50)


if __name__ == '__main__':
    unittest.main()
