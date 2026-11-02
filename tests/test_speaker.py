import os

from gabai.speaker import AlertSpeaker, all_clips, clip_for, distance_label, priority_of


def test_distance_is_rounded_to_half_metres_and_capped():
    assert distance_label(0.2) == "0_5" and distance_label(0.74) == "0_5" and distance_label(0.76) == "1_0"
    assert distance_label(1.2) == "1_0" and distance_label(1.3) == "1_5"
    assert distance_label(2.9) == "3_0" and distance_label(8.0) == "3_0"


def test_clip_names_cover_the_fixed_vocabulary():
    assert clip_for("ahead", 1.1) == "ahead_1_0"
    assert clip_for("drop_ahead", None) == "drop_ahead"
    names = all_clips()
    assert "left_0_5" in names and "right_3_0" in names and "gabai_stopped" in names and "reduced_accuracy" in names
    assert len(names) == len(set(names)) == 23


def test_priorities_follow_the_spec_order():
    assert priority_of("gabai_stopped", None) < priority_of("reduced_accuracy", None) < priority_of("drop_ahead", None)
    assert priority_of("ahead", 0.8) < priority_of("ahead", 2.0)       # within 1 m outranks 1 to 3 m
    assert priority_of("drop_ahead", None) <= priority_of("ahead", 0.8)


def make(played):
    return AlertSpeaker(clips_dir="x", player=lambda path: played.append(os.path.basename(path)), must_exist=False)


def test_repeats_are_suppressed_unless_the_obstacle_got_closer():
    played = []
    s = make(played)
    s.say("ahead", 2.0, now=0.0)
    s.say("ahead", 2.0, now=1.0)           # same alert within 3 s
    s.say("ahead", 1.0, now=1.5)           # closer: repeats
    s.say("ahead", 1.0, now=5.0)           # 3 s later: repeats
    assert played == ["ahead_2_0.wav", "ahead_1_0.wav", "ahead_1_0.wav"]


def test_a_system_alert_cuts_in_and_blocks_lower_ones_while_active():
    played = []
    s = make(played)
    s.say("left", 2.0, now=0.0)
    s.say("gabai_stopped", None, now=0.5)
    s.say("right", 2.0, now=1.0)           # lower priority while the system alert is still active: dropped
    s.say("right", 2.0, now=3.0)           # active window over: allowed
    assert played == ["left_2_0.wav", "gabai_stopped.wav", "right_2_0.wav"]


def test_missing_clip_is_reported_not_crashed():
    s = AlertSpeaker(clips_dir="definitely_missing", player=lambda p: None, must_exist=True)
    assert s.say("ahead", 1.0, now=0.0) is None
    assert s.missing == ["ahead_1_0"]
