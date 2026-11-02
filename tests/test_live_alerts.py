from gabai.live_alerts import AlertController
from gabai.speaker import AlertSpeaker


def make(obstacle_audio=True):
    played = []
    sp = AlertSpeaker(clips_dir="x", player=lambda p: played.append(p.replace("\\", "/").split("/")[-1][:-4]), must_exist=False)
    return AlertController(sp, obstacle_audio=obstacle_audio), played


def test_obstacle_is_spoken_with_rounded_distance():
    c, played = make()
    c.on_frame(0.0, ("ahead", 1.0), "steady", 0.1)
    assert played == ["ahead_1_0"]


def test_obstacle_speech_is_off_without_a_floor_model_but_system_alerts_still_work():
    c, played = make(obstacle_audio=False)
    c.on_frame(0.0, ("ahead", 1.0), "steady", 0.1)
    assert played == []
    c.on_idle(2.0)
    assert played == ["gabai_stopped"]


def test_stopped_stream_is_announced_after_one_second_and_repeats_every_five():
    c, played = make()
    c.on_idle(0.5)                                   # nothing ever arrived yet: stay quiet
    c.on_frame(1.0, None, "steady", 0.1)
    c.on_idle(1.8)
    assert played == []                              # under a second of silence
    c.on_idle(2.2)
    c.on_idle(4.0)                                   # repeat only every 5 s
    c.on_idle(7.3)
    assert played == ["gabai_stopped", "gabai_stopped"]


def test_old_frames_announce_tracking_delayed():
    c, played = make()
    c.on_frame(0.0, None, "steady", 0.45)
    assert played == ["tracking_delayed"]


def test_fast_turning_says_reduced_accuracy_once_per_ten_seconds():
    c, played = make()
    c.on_frame(0.0, None, "fast", 0.1)
    c.on_frame(1.0, None, "fast", 0.1)
    c.on_frame(12.0, None, "fast", 0.1)
    assert played == ["reduced_accuracy", "reduced_accuracy"]


def test_spoken_phrases_are_returned_for_the_telemetry_log():
    c, _ = make()
    said = c.on_frame(0.0, ("left", 2.0), "steady", 0.1)
    assert said == ["left, about two metres"]


def test_a_one_frame_flicker_is_not_spoken_but_a_repeat_is():
    c, played = make()
    c.on_frame(0.0, ("ahead", 1.0), "steady", 0.1)       # first sighting: wait for confirmation? no: obstacles speak at once
    assert played == ["ahead_1_0"]
    c2, played2 = make()
    c2.on_frame(0.0, ("drop_ahead", None), "steady", 0.1)     # a drop or 'cannot see' needs two frames in a row
    assert played2 == []
    c2.on_frame(0.07, ("drop_ahead", None), "steady", 0.1)
    assert played2 == ["drop_ahead"]


def test_alternating_results_do_not_chatter():
    c, played = make()
    for i in range(10):
        c.on_frame(i * 0.07, ("drop_ahead", None) if i % 2 == 0 else None, "steady", 0.1)
    assert played == []


def test_every_silence_has_a_stated_reason():
    from gabai.live_alerts import AlertController
    from gabai.speaker import AlertSpeaker
    sp = AlertSpeaker(player=lambda p: None, must_exist=False)
    c = AlertController(sp, obstacle_audio=True)
    c.on_frame(10.0, ("ahead", 1.0), "steady", 0.05)
    assert c.decision["spoken"] and c.decision["not_spoken"] == []
    c.on_frame(10.2, ("ahead", 1.0), "steady", 0.05)                   # same alert again inside the repeat window
    assert c.decision["spoken"] == [] and "already said" in c.decision["not_spoken"][0]["reason"]
    c.on_frame(10.4, ("cant_see_ahead", None), "steady", 0.05)         # a system-level alert needs two frames in a row
    assert "two frames" in c.decision["not_spoken"][0]["reason"]
    off = AlertController(AlertSpeaker(player=lambda p: None, must_exist=False), obstacle_audio=False)
    off.on_frame(1.0, ("left", 0.5), "steady", 0.05)
    assert "--cam-h" in off.decision["not_spoken"][0]["reason"]


def test_a_missing_clip_is_reported_with_the_command_that_fixes_it():
    from gabai.speaker import AlertSpeaker
    sp = AlertSpeaker(clips_dir="no_such_dir", player=lambda p: None)
    assert sp.say("ahead", 1.0, 5.0) is None
    assert "make_alert_audio" in sp.last_reason


def test_lower_priority_while_a_higher_one_is_active_is_explained():
    from gabai.speaker import AlertSpeaker
    sp = AlertSpeaker(player=lambda p: None, must_exist=False)
    sp.say("gabai_stopped", None, 1.0)
    assert sp.say("left", 2.0, 1.5) is None and "lower priority" in sp.last_reason
