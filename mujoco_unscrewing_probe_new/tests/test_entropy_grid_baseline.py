from __future__ import annotations

import unittest

import numpy as np

from src.entropy_grid_baseline import (
    EntropyGridBelief,
    EntropyGridConfig,
    certify_disk_grid_coverage,
    disk_grid,
    edge_loaded_from_wrench,
    torque_category_from_wrench,
    weighted_mass_radius,
)


class TestEntropyGridBaseline(unittest.TestCase):
    def test_disk_grid_respects_declared_visual_support(self) -> None:
        center = np.array([0.001, -0.002])
        radius = 0.004
        grid = disk_grid(center, radius, 0.0005)
        self.assertGreater(len(grid), 100)
        self.assertLessEqual(
            float(np.max(np.linalg.norm(grid - center[None, :], axis=1))),
            radius + 1.0e-12,
        )

    def test_action_grid_padding_certifies_the_complete_visual_disk(self) -> None:
        capture_radius = 0.00025
        spacing = 0.95 * np.sqrt(2.0) * capture_radius
        belief = EntropyGridBelief(
            np.zeros(2),
            visual_sigma_m=0.002,
            visual_bound_m=0.004895303872079853,
            support_inner_radius_m=0.00325,
            support_outer_radius_m=0.00650,
            probe_radius_m=0.006 / np.sqrt(3.0),
            config=EntropyGridConfig(action_spacing_m=spacing),
        )
        self.assertLessEqual(
            float(np.max(np.linalg.norm(belief.candidate_xy_m, axis=1))),
            belief.visual_bound_m + 1.0e-12,
        )
        self.assertGreater(
            float(np.max(np.linalg.norm(belief.action_xy_m, axis=1))),
            belief.visual_bound_m,
        )
        self.assertAlmostEqual(belief.action_padding_m, capture_radius, places=12)
        self.assertTrue(belief.action_grid_coverage_certified)

        certificate = belief.certify_action_coverage(
            dense_subdivisions_per_cell=6,
            boundary_sample_count=2048,
        )
        self.assertTrue(certificate.complete_required_lattice)
        self.assertTrue(certificate.certified)
        self.assertLessEqual(
            certificate.analytic_worst_case_gap_m,
            capture_radius + 1.0e-12,
        )
        self.assertLessEqual(
            certificate.observed_worst_gap_m,
            capture_radius + 1.0e-12,
        )

    def test_clipped_action_disk_does_not_receive_a_false_certificate(self) -> None:
        radius = 0.004895303872079853
        capture_radius = 0.00025
        spacing = 0.95 * np.sqrt(2.0) * capture_radius
        clipped = disk_grid(np.zeros(2), radius, spacing)
        certificate = certify_disk_grid_coverage(
            clipped,
            np.zeros(2),
            support_radius_m=radius,
            lattice_spacing_m=spacing,
            effective_capture_radius_m=capture_radius,
            dense_subdivisions_per_cell=4,
            boundary_sample_count=1024,
        )
        self.assertFalse(certificate.complete_required_lattice)
        self.assertFalse(certificate.certified)

    def test_credible_radius_is_exact_weighted_mass_about_posterior_mean(self) -> None:
        points = np.array([[0.0, 0.0], [0.001, 0.0], [0.002, 0.0]])
        weights = np.array([0.94, 0.02, 0.04])
        mean = np.sum(weights[:, None] * points, axis=0)
        self.assertAlmostEqual(
            weighted_mass_radius(points, weights, mean, mass=0.95),
            0.0009,
            places=12,
        )

        belief = EntropyGridBelief(
            np.zeros(2), 0.002, 0.004, 0.00325, 0.00650, 0.003464
        )
        custom = np.zeros_like(belief.weights)
        selected = np.argsort(belief.candidate_xy_m[:, 0])[[0, len(belief.weights) // 2, -1]]
        custom[selected] = weights
        belief.weights = custom
        expected = weighted_mass_radius(
            belief.candidate_xy_m,
            belief.weights,
            belief.posterior_mean_xy_m,
            mass=0.95,
        )
        self.assertAlmostEqual(belief.credible_radius_95_m, expected, places=12)

    def test_entropy_action_and_update_do_not_require_true_center(self) -> None:
        belief = EntropyGridBelief(
            np.array([0.002, -0.001]),
            visual_sigma_m=0.002,
            visual_bound_m=0.004895,
            support_inner_radius_m=0.00325,
            support_outer_radius_m=0.00650,
            probe_radius_m=0.003464,
        )
        before = belief.entropy_nats
        action = belief.select_action(np.array([0.002, -0.001]))
        self.assertGreater(action.expected_information_gain_nats, 0.0)
        update = belief.update(action, observation_category=1)
        self.assertAlmostEqual(float(np.sum(belief.weights)), 1.0, places=12)
        self.assertLess(update.entropy_nats, before)
        self.assertEqual(len(belief.history), 1)

    def test_same_action_cannot_be_reused(self) -> None:
        belief = EntropyGridBelief(
            np.zeros(2), 0.002, 0.004895, 0.00325, 0.00650, 0.003464,
            EntropyGridConfig(max_probes=2),
        )
        action = belief.select_action(np.zeros(2))
        belief.update(action, observation_category=0)
        with self.assertRaises(ValueError):
            belief.update(action, observation_category=0)

    def test_ft_edge_cue_is_force_scale_normalized(self) -> None:
        weak = np.array([0.0, 0.0, 1.5, 0.0010, 0.0, 0.0])
        strong = 2.0 * weak
        weak_edge, weak_lever = edge_loaded_from_wrench(
            weak, lever_threshold_m=0.0005
        )
        strong_edge, strong_lever = edge_loaded_from_wrench(
            strong, lever_threshold_m=0.0005
        )
        self.assertTrue(weak_edge)
        self.assertTrue(strong_edge)
        self.assertAlmostEqual(weak_lever, strong_lever, places=12)

    def test_signed_torque_keeps_opposite_edges_distinct(self) -> None:
        inner = np.array([0.0, 0.0, 3.0, 0.0, -0.003, 0.0])
        outer = np.array([0.0, 0.0, 3.0, 0.0, +0.003, 0.0])
        inner_category, _ = torque_category_from_wrench(
            inner, low_lever_threshold_m=0.00025, direction_bins=8
        )
        outer_category, _ = torque_category_from_wrench(
            outer, low_lever_threshold_m=0.00025, direction_bins=8
        )
        self.assertNotEqual(inner_category, outer_category)

    def test_upstream_samples_initialize_a_nonuniform_joint_prior(self) -> None:
        belief = EntropyGridBelief(
            np.zeros(2),
            visual_sigma_m=0.002,
            visual_bound_m=0.004895,
            support_inner_radius_m=0.00325,
            support_outer_radius_m=0.00650,
            probe_radius_m=0.003464,
        )
        samples = np.repeat(np.array([[0.0015, -0.0005]]), 256, axis=0)
        belief.set_prior_from_samples(samples, bandwidth_m=0.00030)
        self.assertLess(
            np.linalg.norm(belief.posterior_mean_xy_m - samples[0]),
            0.00020,
        )
        drawn = belief.sample_posterior(512, np.random.default_rng(7))
        self.assertLess(
            np.linalg.norm(np.mean(drawn, axis=0) - samples[0]),
            0.00025,
        )

    def test_value_gate_penalizes_long_contact_actions(self) -> None:
        belief = EntropyGridBelief(
            np.zeros(2),
            visual_sigma_m=0.002,
            visual_bound_m=0.004895,
            support_inner_radius_m=0.00325,
            support_outer_radius_m=0.00650,
            probe_radius_m=0.003464,
        )
        action = belief.select_action(
            np.array([0.004, 0.0]),
            fixed_action_cost_nats=0.08,
            travel_penalty_nats_per_m=20.0,
        )
        self.assertIsNotNone(action.net_utility_nats)
        self.assertAlmostEqual(
            float(action.net_utility_nats),
            action.expected_information_gain_nats
            - 0.08
            - 20.0 * action.travel_distance_m,
        )

    def test_task_aware_action_combines_capture_information_and_cost(self) -> None:
        belief = EntropyGridBelief(
            np.zeros(2),
            visual_sigma_m=0.002,
            visual_bound_m=0.004895,
            support_inner_radius_m=0.00325,
            support_outer_radius_m=0.00650,
            probe_radius_m=0.003464,
        )
        action = belief.select_task_aware_action(
            np.array([0.002, -0.001]),
            effective_capture_radius_m=0.00025,
            information_gain_weight=0.35,
            capture_probability_reward_nats=1.5,
            fixed_action_cost_nats=0.08,
            travel_penalty_nats_per_m=20.0,
        )
        self.assertGreaterEqual(action.segment_capture_probability, 0.0)
        self.assertLessEqual(action.segment_capture_probability, 1.0)
        self.assertAlmostEqual(
            float(action.net_utility_nats),
            0.35 * action.expected_information_gain_nats
            + 1.5 * action.segment_capture_probability
            - 0.08
            - 20.0 * action.travel_distance_m,
        )

    def test_no_capture_softly_reduces_swept_probability_mass(self) -> None:
        belief = EntropyGridBelief(
            np.zeros(2),
            visual_sigma_m=0.002,
            visual_bound_m=0.004895,
            support_inner_radius_m=0.00325,
            support_outer_radius_m=0.00650,
            probe_radius_m=0.003464,
        )
        start = np.array([-0.002, 0.0])
        end = np.array([0.002, 0.0])
        before = belief.segment_capture_probability(start, end, 0.00025)
        update = belief.apply_no_capture_segment(
            start,
            end,
            effective_capture_radius_m=0.00025,
            miss_inside_likelihood=0.20,
            miss_outside_likelihood=0.995,
        )
        after = belief.segment_capture_probability(start, end, 0.00025)
        self.assertAlmostEqual(update.covered_probability_before, before)
        self.assertLess(after, before)
        self.assertGreater(after, 0.0)
        self.assertAlmostEqual(float(np.sum(belief.weights)), 1.0, places=12)

    def test_zero_temperature_ignores_an_untrusted_ft_category(self) -> None:
        belief = EntropyGridBelief(
            np.zeros(2), 0.002, 0.004895, 0.00325, 0.00650, 0.003464
        )
        action = belief.select_action(np.zeros(2))
        before = belief.weights.copy()
        update = belief.update(
            action,
            observation_category=1,
            likelihood_temperature=0.0,
        )
        np.testing.assert_allclose(belief.weights, before)
        self.assertAlmostEqual(update.likelihood_temperature, 0.0)


if __name__ == "__main__":
    unittest.main()
