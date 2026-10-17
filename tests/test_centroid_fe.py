import importlib
import importlib.util
import json
import numpy as np
import pytest


def backend():
    assert importlib.util.find_spec("efu.centroid_fe") is not None, (
        "compact transport missing"
    )
    return importlib.import_module("efu.centroid_fe")


def round_fixture():
    mod = backend()
    cfg = mod.Config.create(
        [2, 3], dimension=4, kappa=3, precision=2, max_abs=2, round_id=7
    )
    cfg = mod.Config.from_bytes(cfg.to_bytes())
    clients = [mod.Participant(cfg, i) for i in range(2)]
    peers = {i: client.public_key() for i, client in enumerate(clients)}
    for client in clients:
        client.set_peers(peers)
    maps = [np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0])]
    messages = [
        clients[0].encrypt([-1.25, 0.5], maps[0]),
        clients[1].encrypt([1.5, 0.25], maps[1]),
    ]
    server = mod.Aggregator(cfg, peers)
    return mod, cfg, clients, server, maps, messages


def test_compact_signed_weighted_round_and_local_state_restore():
    mod, cfg, clients, server, maps, messages = round_fixture()
    clients = [mod.Participant.from_private(c.export_private()) for c in clients]
    request = server.request(messages, maps)
    shares = [c.authorize(request) for c in clients]
    result = server.aggregate(messages, request, shares)
    np.testing.assert_allclose(result, [-0.35, 1.1, -0.35, 1.1], atol=1e-12)
    assert server.last_costs["ciphertext_points"] == 6
    assert server.last_costs["authorized_functions"] == 2
    assert server.last_costs["authorization_bytes"] == sum(map(len, shares))
    assert server.last_costs["request_bytes"] == len(request)
    with pytest.raises(ValueError, match="already|replay"):
        server.aggregate(messages, request, shares)
    with pytest.raises(ValueError, match="already|once"):
        clients[0].encrypt([0, 1], maps[0])


def test_authorization_rejects_mapping_or_ciphertext_commitment_changes():
    _, _, clients, server, maps, messages = round_fixture()
    request = server.request(messages, maps)
    bad = json.loads(request)
    bad["ciphertexts"][0] = "0" * 64
    with pytest.raises(ValueError, match="ciphertext|commitment"):
        clients[0].authorize(json.dumps(bad).encode())
    changed = [maps[0][::-1], maps[1]]
    with pytest.raises(ValueError, match="mapping"):
        server.request(messages, changed)
    shares = clients[0].authorize(request)
    assert clients[0].authorize(request) == shares  # identical network retry
    bad = json.loads(request)
    bad["ciphertexts"][1] = "1" * 64
    with pytest.raises(ValueError):
        clients[0].authorize(json.dumps(bad).encode())


def test_signatures_cohort_and_configuration_reject_tampering():
    mod, cfg, clients, server, maps, messages = round_fixture()
    bad = json.loads(messages[0])
    bad["body"]["points"][0] = "00"
    with pytest.raises(ValueError, match="signature"):
        server.request([json.dumps(bad).encode(), messages[1]], maps)
    with pytest.raises(ValueError, match="participants|cohort"):
        server.request([messages[0], messages[0]], maps)
    with pytest.raises(ValueError):
        mod.Config.create([1, True], dimension=2, kappa=2)
    with pytest.raises(ValueError):
        mod.Config.create([1, 1], dimension=2, kappa=2, round_id=1.0)
    with pytest.raises(ValueError):
        mod.Config.from_bytes(b'{"version":1,"version":2}')


def test_cross_session_shares_rejected_and_no_centroid_difference_cancellation():
    _, _, clients, server, maps, messages = round_fixture()
    from efu.crypto import decode_point, G

    points = json.loads(messages[0])["body"]["points"]
    assert decode_point(points[1]) - decode_point(points[0]) != 175 * G
    request = server.request(messages, maps)
    shares = [c.authorize(request) for c in clients]
    other = round_fixture()
    request2 = other[3].request(other[5], other[4])
    shares2 = [c.authorize(request2) for c in other[2]]
    with pytest.raises(ValueError):
        server.aggregate(messages, request, [shares[0], shares2[1]])


def test_large_common_sample_factor_does_not_expand_discrete_log_range():
    mod = backend()
    cfg = mod.Config.create(
        [2**60, 2**60], dimension=1, kappa=1, precision=0, max_abs=2
    )
    clients = [mod.Participant(cfg, i) for i in range(2)]
    peers = {i: c.public_key() for i, c in enumerate(clients)}
    for c in clients:
        c.set_peers(peers)
    messages = [clients[0].encrypt([-2], [0]), clients[1].encrypt([2], [0])]
    server = mod.Aggregator(cfg, peers)
    request = server.request(messages, [[0], [0]])
    np.testing.assert_array_equal(
        server.aggregate(messages, request, [c.authorize(request) for c in clients]),
        [0],
    )


def test_public_registration_separates_signing_and_exchange_keys():
    mod = backend()
    cfg = mod.Config.create([1, 1], dimension=1, kappa=1)
    client = mod.Participant(cfg, 0)
    raw = client.public_key()
    assert raw.startswith(b"{"), "Separate public key bundle is required"
    public = json.loads(raw)
    assert public["exchange"] != public["signing"]


def test_validly_signed_malformed_curve_point_is_rejected():
    mod, _, clients, server, maps, messages = round_fixture()
    from Crypto.Hash import SHA256
    from Crypto.PublicKey import ECC
    from Crypto.Signature import DSS

    body = json.loads(messages[0])["body"]
    body["points"][0] = "02" + "f" * 64
    state, _ = mod._private_decode(clients[0].export_private())
    key = ECC.construct(curve="P-256", d=int(state["signing"], 16))
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    signature = DSS.new(key, "deterministic-rfc6979").sign(SHA256.new(canonical)).hex()
    bad = json.dumps({"body": body, "signature": signature}).encode()
    with pytest.raises(ValueError, match="point|Point"):
        server.request([bad, messages[1]], maps)
