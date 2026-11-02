from gabai.pi.lidar_sender import checksum_ok, parse

FRAME = bytes.fromhex("aa00090161ae00017a023e")


def test_frame():
    assert checksum_ok(FRAME)
    assert not checksum_ok(FRAME[:-1] + b"")
    assert parse(FRAME) == 634
