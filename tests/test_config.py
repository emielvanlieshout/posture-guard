from __future__ import annotations

import json

import pytest

from posture_guard.config import Config, app_dir, in_quiet_hours


class TestValidation:
    def test_defaults_are_valid(self):
        Config().validate()

    def test_view_must_be_known(self):
        with pytest.raises(ValueError, match="view must be"):
            Config(view="diagonal").validate()

    def test_exit_below_enter(self):
        with pytest.raises(ValueError, match="exit < enter"):
            Config(enter=0.3, exit=0.5).validate()

    def test_dim_is_capped_so_the_screen_stays_usable(self):
        with pytest.raises(ValueError, match="max_dim"):
            Config(max_dim=0.99).validate()
        Config(max_dim=0.85).validate()

    def test_frame_rate_is_bounded(self):
        with pytest.raises(ValueError):
            Config(fps=0).validate()
        with pytest.raises(ValueError):
            Config(fps=120).validate()


class TestPersistence:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "config.json"
        Config(view="frontal", max_dim=0.4).save(path)
        loaded = Config.load(path)
        assert loaded.view == "frontal"
        assert loaded.max_dim == 0.4

    def test_a_missing_file_gives_defaults(self, tmp_path):
        assert Config.load(tmp_path / "nope.json").view == Config().view

    def test_a_typo_in_the_file_is_reported_not_ignored(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"maxdim": 0.4}))
        with pytest.raises(ValueError, match="unknown settings"):
            Config.load(path)

    def test_an_invalid_saved_value_is_caught_on_load(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"max_dim": 0.99}))
        with pytest.raises(ValueError):
            Config.load(path)


class TestSetFromString:
    @pytest.mark.parametrize(
        "key,value,expected",
        [
            ("fps", "10", 10.0),
            ("camera_index", "2", 2),
            ("ratchet_enabled", "false", False),
            ("view", "frontal", "frontal"),
            ("alerters", "dim,notify", ["dim", "notify"]),
            ("quiet_start", "22:30", "22:30"),
            ("quiet_start", "none", None),
        ],
    )
    def test_parsing(self, key, value, expected):
        cfg = Config()
        cfg.set_from_string(key, value)
        assert getattr(cfg, key) == expected

    def test_unknown_key_is_refused(self):
        with pytest.raises(ValueError, match="unknown setting"):
            Config().set_from_string("colour", "red")

    def test_an_invalid_value_is_refused(self):
        with pytest.raises(ValueError):
            Config().set_from_string("max_dim", "0.99")

    def test_a_malformed_time_is_refused(self):
        with pytest.raises(ValueError, match="HH:MM"):
            Config().set_from_string("quiet_start", "half past ten")


class TestQuietHours:
    def test_off_by_default(self):
        assert not in_quiet_hours(Config(), 3, 0)

    def test_a_daytime_window(self):
        cfg = Config(quiet_start="09:00", quiet_end="10:00")
        assert in_quiet_hours(cfg, 9, 30)
        assert not in_quiet_hours(cfg, 10, 0), "end is exclusive"
        assert not in_quiet_hours(cfg, 8, 59)

    def test_a_window_across_midnight(self):
        cfg = Config(quiet_start="22:00", quiet_end="07:00")
        assert in_quiet_hours(cfg, 23, 0)
        assert in_quiet_hours(cfg, 3, 0)
        assert not in_quiet_hours(cfg, 12, 0)

    def test_one_sided_configuration_is_ignored(self):
        assert not in_quiet_hours(Config(quiet_start="22:00"), 23, 0)


def test_home_can_be_redirected(monkeypatch, tmp_path):
    monkeypatch.setenv("POSTURE_GUARD_HOME", str(tmp_path / "pg"))
    assert app_dir() == tmp_path / "pg"
