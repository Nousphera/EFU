import json
import zlib
import numpy as np
import pytest
from test_centroid_fe import round_fixture
from efu import centroid_fe as mod


def authorized_chunks(clients, server, messages, maps, size=2):
    for request in server.request_chunks(messages, maps, chunk_size=size):
        yield request, [client.authorize(request) for client in clients]


def test_streamed_chunks_cover_every_coordinate_and_count_actual_bytes():
    _, cfg, clients, server, maps, messages = round_fixture()
    observed = []

    def stream():
        for pair in authorized_chunks(clients, server, messages, maps):
            observed.append(pair)
            yield pair

    result = server.aggregate_chunks(messages, stream())
    np.testing.assert_allclose(result, [-0.35, 1.1, -0.35, 1.1], atol=1e-12)
    assert all(c.authorized_complete for c in clients)
    assert server.last_costs["authorization_chunks"] == 2
    assert server.last_costs["authorized_functions"] == 4
    assert server.last_costs["ciphertext_bytes"] == sum(map(len, messages))
    assert server.last_costs["request_bytes"] == sum(len(req) for req, _ in observed)
    assert server.last_costs["authorization_bytes"] == sum(
        len(s) for _, shares in observed for s in shares
    )
    with pytest.raises(ValueError, match="replay|already"):
        server.aggregate_chunks(messages, iter(observed))


def test_chunk_retries_restore_state_and_reject_changes_or_overlap():
    _, cfg, clients, server, maps, messages = round_fixture()
    requests = list(server.request_chunks(messages, maps, chunk_size=2))
    share = clients[0].authorize(requests[0])
    assert not clients[0].authorized_complete
    restored = mod.Participant.from_private(clients[0].export_private())
    assert restored.authorize(requests[0]) == share
    changed = json.loads(requests[0])
    changed["ciphertexts"][1] = "1" * 64
    with pytest.raises(ValueError, match="different|authorized"):
        restored.authorize(json.dumps(changed).encode())
    overlap = json.loads(requests[1])
    overlap["start"] = 1
    overlap["stop"] = 3
    overlap["mappings"] = [
        mod._b64(np.asarray(m[1:3], dtype=cfg.mapping_dtype).tobytes()) for m in maps
    ]
    with pytest.raises(ValueError, match="overlap|order|coverage"):
        restored.authorize(json.dumps(overlap).encode())
    restored.authorize(requests[1])
    assert mod.Participant.from_private(restored.export_private()).authorized_complete


@pytest.mark.parametrize("selection", [[0], [1], [1, 0], [0, 0]])
def test_aggregate_rejects_missing_reordered_or_duplicate_chunks(selection):
    _, _, clients, server, maps, messages = round_fixture()
    pairs = list(authorized_chunks(clients, server, messages, maps))
    with pytest.raises(ValueError, match="coverage|order|gap|overlap"):
        server.aggregate_chunks(messages, (pairs[i] for i in selection))


def test_chunk_own_mapping_and_full_mapping_commitment_are_bound():
    _, cfg, clients, server, maps, messages = round_fixture()
    request = next(server.request_chunks(messages, maps, chunk_size=2))
    bad = json.loads(request)
    bad["mappings"][0] = mod._b64(bytes([1, 0]))
    with pytest.raises(ValueError, match="mapping"):
        clients[0].authorize(json.dumps(bad).encode())
    changed = [maps[0].copy(), maps[1]]
    changed[0][-1] = 0
    with pytest.raises(ValueError, match="mapping"):
        next(server.request_chunks(messages, changed, chunk_size=2))


def test_default_model_mapping_sizes_fit_chunk_and_private_bounds():
    # ResNet18 and ViT-B have millions of float coordinates: the prior all-map
    # base64-u4 request exceeded 256 MiB even before signatures/JSON metadata.
    for dimension in (11_200_000, 86_000_000):
        cfg = mod.Config.create([1] * 10, dimension=dimension, kappa=128)
        assert 10 * dimension * 4 * 4 // 3 > mod.MAX_WIRE_BYTES
        assert cfg.mapping_bytes == dimension
        assert (
            cfg.mapping_bytes + mod.MAX_PRIVATE_HEADER_BYTES + 12 < mod.MAX_WIRE_BYTES
        )
        assert 10 * 16384 * cfg.mapping_dtype.itemsize * 4 // 3 < mod.MAX_WIRE_BYTES
    assert mod.Config.create([1, 1], dimension=4, kappa=257).mapping_dtype.itemsize == 2


def test_private_envelope_is_bounded_and_rejects_compression_expansion():
    _, _, clients, _, _, _ = round_fixture()
    raw = clients[0].export_private()
    assert raw.startswith(mod.PRIVATE_MAGIC)
    length = int.from_bytes(raw[4:12], "big")
    header = json.loads(raw[12 : 12 + length])
    header["mapping"] = {"codec": "zlib", "length": 4}
    encoded = mod._dump(header)
    bomb = (
        mod.PRIVATE_MAGIC
        + len(encoded).to_bytes(8, "big")
        + encoded
        + zlib.compress(bytes(10000))
    )
    with pytest.raises(ValueError, match="mapping|length|bounded"):
        mod.Participant.from_private(bomb)


def test_chunk_shares_cannot_be_replayed_for_another_range():
    _, _, clients, server, maps, messages = round_fixture()
    pairs = list(authorized_chunks(clients, server, messages, maps))
    with pytest.raises(ValueError, match="transcript"):
        server.aggregate_chunks(messages, [pairs[0], (pairs[1][0], pairs[0][1])])


def test_full_signed_mapping_digest_checked_after_all_chunks():
    _, _, clients, server, maps, messages = round_fixture()
    pairs = list(authorized_chunks(clients, server, messages, maps))
    # Even forged authorization by all registered signers must not bypass the
    # signed full-map commitment. Reordering preserves the selection tuple set.
    request = json.loads(pairs[1][0])
    request["mappings"] = [mod._b64(bytes([1, 0])), mod._b64(bytes([0, 1]))]
    raw = mod._dump(request)
    digest = mod.sha256(raw).hexdigest()
    shares = []
    for client, old in zip(clients, pairs[1][1]):
        body = json.loads(old)["body"]
        body["request"] = digest
        shares.append(mod._sign(body, client._signing_key))
    with pytest.raises(ValueError, match="mapping commitment"):
        server.aggregate_chunks(messages, [pairs[0], (raw, shares)])


def test_default_chunk_size_streams_more_than_one_wire_window():
    cfg = mod.Config.create([1, 2], dimension=65537, kappa=1, precision=1, max_abs=2)
    clients = [mod.Participant(cfg, i) for i in range(2)]
    peers = {i: c.public_key() for i, c in enumerate(clients)}
    maps = [np.zeros(cfg.dimension, dtype=np.uint8) for _ in clients]
    for c in clients:
        c.set_peers(peers)
    messages = [
        c.encrypt([value], mapping)
        for c, value, mapping in zip(clients, [-1.0, 0.5], maps)
    ]
    server = mod.Aggregator(cfg, peers)
    with pytest.raises(ValueError, match="request_chunks"):
        server.request(messages, maps)
    requests = []

    def stream():
        for request in server.request_chunks(messages, maps):
            requests.append(len(request))
            yield request, [c.authorize(request) for c in clients]

    np.testing.assert_array_equal(
        server.aggregate_chunks(messages, stream()), np.zeros(cfg.dimension)
    )
    assert len(requests) == 5 and max(requests) < 50000
    assert all(c.authorized_complete for c in clients)
