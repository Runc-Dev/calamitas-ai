"""Unit tests for afetsonar.routing.*"""

from __future__ import annotations

import math

import pytest


class TestPriority:
    def test_compute_priority_zero_for_no_damage(self):
        from afetsonar.routing import compute_priority
        score = compute_priority(damage_class=1, area_m2=500.0)
        assert score == 0.0

    def test_compute_priority_higher_for_destroyed(self):
        from afetsonar.routing import compute_priority
        p_minor = compute_priority(2, 100.0)
        p_dest = compute_priority(4, 100.0)
        assert p_dest > p_minor

    def test_compute_survival_destroyed_low(self):
        from afetsonar.routing import compute_survival_probability
        s = compute_survival_probability(damage_class=4, elapsed_hours=6.0)
        assert 0.0 <= s <= 1.0
        assert s < 0.5

    def test_compute_survival_no_damage_full(self):
        from afetsonar.routing import compute_survival_probability
        s = compute_survival_probability(damage_class=1)
        assert s == 1.0

    def test_score_buildings_sorted(self, dummy_buildings):
        from afetsonar.routing import score_buildings
        scored = score_buildings(dummy_buildings)
        scores = [b["priority_score"] for b in scored]
        assert scores == sorted(scores, reverse=True)


class TestTeamAssignment:
    def test_all_buildings_assigned(self, dummy_buildings):
        from afetsonar.routing import assign_teams
        buildings, teams = assign_teams(dummy_buildings, n_teams=3)
        assert all("team_id" in b for b in buildings)

    def test_n_teams_correct(self, dummy_buildings):
        from afetsonar.routing import assign_teams
        _, teams = assign_teams(dummy_buildings, n_teams=3)
        assert len(teams) <= 3  # some clusters may be empty

    def test_hospital_assignment(self, dummy_buildings):
        from afetsonar.routing import assign_teams, assign_hospitals
        hospitals = [
            {"name": "Hospital A", "lat": 41.005, "lon": 28.977},
            {"name": "Hospital B", "lat": 41.010, "lon": 28.980},
        ]
        _, teams = assign_teams(dummy_buildings, n_teams=2)
        teams = assign_hospitals(teams, hospitals)
        for team in teams:
            assert "assigned_hospital" in team
            assert team["assigned_hospital"] in ("Hospital A", "Hospital B")


class TestTSP:
    def test_all_buildings_in_order(self, dummy_buildings):
        from afetsonar.routing import nearest_neighbor_tsp
        order = nearest_neighbor_tsp(
            start_lat=41.005, start_lon=28.977,
            buildings=dummy_buildings,
        )
        bids = {b["building_id"] for b in dummy_buildings}
        assert set(order) == bids

    def test_distance_finite(self, dummy_buildings):
        from afetsonar.routing import nearest_neighbor_tsp, tsp_total_distance
        order = nearest_neighbor_tsp(41.005, 28.977, dummy_buildings)
        dist = tsp_total_distance(41.005, 28.977, dummy_buildings, order)
        assert math.isfinite(dist) and dist > 0


class TestHelicopter:
    def test_eta_positive(self):
        from afetsonar.routing import compute_heli_eta
        eta = compute_heli_eta(41.0, 28.9, 41.01, 28.99)
        assert eta > 0

    def test_rank_landing_zones(self):
        from afetsonar.routing import rank_landing_zones
        lz_candidates = [
            {"lz_id": i, "name": f"Park_{i}", "lat": 41.005 + i * 0.002,
             "lon": 28.977, "area_m2": 1000.0 + i * 100}
            for i in range(5)
        ]
        building = {"building_id": 0, "lat": 41.006, "lon": 28.977}
        ranked = rank_landing_zones(building, lz_candidates, top_k=3)
        assert len(ranked) <= 3
        assert ranked[0]["rank"] == 1


class TestGeoUtils:
    def test_haversine_symmetry(self):
        from afetsonar.geo import haversine_distance
        d1 = haversine_distance(41.0, 28.9, 41.01, 28.91)
        d2 = haversine_distance(41.01, 28.91, 41.0, 28.9)
        assert abs(d1 - d2) < 1e-6

    def test_haversine_zero(self):
        from afetsonar.geo import haversine_distance
        assert haversine_distance(41.0, 28.9, 41.0, 28.9) == 0.0

    def test_pixel_to_geo_roundtrip(self):
        from afetsonar.geo import pixel_to_geo, geo_to_pixel
        transform = [0.5 / 1024, 0, 28.97, 0, -0.5 / 1024, 41.01]
        x, y = 512.0, 256.0
        gx, gy = pixel_to_geo(x, y, transform)
        rx, ry = geo_to_pixel(gx, gy, transform)
        assert abs(rx - x) < 1e-8 and abs(ry - y) < 1e-8
