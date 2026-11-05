import json
import socket

from gabai.relay import discovery


def test_reply_only_to_gabai_query():
    assert json.loads(discovery.reply_for(b"GABAI?\n", 8090)) == {"gabai": 1, "relay_port": 8090}
    assert discovery.reply_for(b"hello", 8090) is None
    assert discovery.reply_for(b"", 8090) is None


def test_udp_round_trip_on_loopback():
    srv = discovery.serve(8090, udp_port=48091)
    try:
        c = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        c.settimeout(2)
        c.sendto(b"noise", ("127.0.0.1", 48091))      # ignored, must not break the loop
        c.sendto(discovery.QUERY, ("127.0.0.1", 48091))
        data, addr = c.recvfrom(256)
        assert json.loads(data)["relay_port"] == 8090
        assert addr[1] == 48091
    finally:
        srv.close()
