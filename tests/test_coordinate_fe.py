import numpy as np
import pytest
from efu.coordinate_fe import setup


def test_weighted_signed_aggregate_and_no_server_private_keys():
    clients, server = setup([2, 3], dimension=3, precision=2, max_abs=2)
    a = clients[0].encrypt([-1.25, .5, 1], round_id=1)
    b = clients[1].encrypt([.25, 1.5, -1], round_id=1)
    assert not hasattr(server, "clients") and not hasattr(server, "key")
    assert not hasattr(server.public, "msk")
    np.testing.assert_allclose(server.aggregate([b, a], round_id=1), [-.35, 1.1, -.2], atol=1e-10)
    assert a.ciphertexts[0].tag != a.ciphertexts[1].tag


def test_rejects_reuse_missing_participant_and_cross_round():
    clients, server = setup([1, 1], dimension=2, precision=1, max_abs=2)
    a = clients[0].encrypt([.1, .2], round_id=1)
    with pytest.raises(ValueError, match="already"):
        clients[0].encrypt([.2, .3], round_id=1)
    b = clients[1].encrypt([.2, .3], round_id=2)
    with pytest.raises(ValueError, match="participants"):
        server.aggregate([a], round_id=1)
    with pytest.raises(ValueError, match="round"):
        server.aggregate([a,b], round_id=1)


def test_public_bounds_and_client_shapes_checked_before_encryption():
    clients, _ = setup([1, 2], dimension=2, precision=1, max_abs=2)
    for bad in ([3, 0], [float("nan"), 0], [0]):
        with pytest.raises(ValueError):
            clients[0].encrypt(bad, round_id=1)
    with pytest.raises(ValueError):
        setup([1,0], dimension=2)


def test_distinct_coordinate_labels_prevent_plaintext_difference_cancellation():
    clients, server = setup([1, 1], dimension=2, precision=0, max_abs=8)
    message = clients[0].encrypt([2,5], round_id=1)
    difference = message.ciphertexts[1].c[0] - message.ciphertexts[0].c[0]
    assert difference != 3 * server.public.g


def test_authorized_function_and_coordinate_order_are_checked():
    from dataclasses import replace
    clients, server = setup([1,2],dimension=2,precision=1,max_abs=2)
    a=clients[0].encrypt([.1,.2],round_id=1)
    b=clients[1].encrypt([.2,.3],round_id=1)
    with pytest.raises(ValueError,match='label'):
        server.aggregate([replace(a,ciphertexts=tuple(reversed(a.ciphertexts))),b],round_id=1)
    a.share.y=[[1],[1]]
    with pytest.raises(ValueError,match='function'):
        server.aggregate([a,b],round_id=1)
