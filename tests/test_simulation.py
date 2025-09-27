from spatial_hud.simulation import offline_feature_stream


def test_offline_stream_generates_events():
    stream = offline_feature_stream(duration_s=0.2, seed=42)
    packets = [next(stream) for _ in range(3)]
    assert len(packets) == 3
    assert all(-90.0 <= pkt.azimuth_deg <= 90.0 for pkt in packets)
