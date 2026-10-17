"""Known answers from RFC 9380 J.1.1; bounded arithmetic regressions."""

import importlib.util
import pytest


def test_hash_to_curve_known_answers():
    assert importlib.util.find_spec("efu.crypto") is not None, (
        "RFC 9380 backend missing"
    )
    from efu.crypto import hash_to_curve

    dst = b"QUUX-V01-CS02-with-P256_XMD:SHA-256_SSWU_RO_"
    vectors = [
        (
            b"",
            "2c15230b26dbc6fc9a37051158c95b79656e17a1a920b11394ca91c44247d3e4",
            "8a7a74985cc5c776cdfe4b1f19884970453912e9d31528c060be9ab5c43e8415",
        ),
        (
            b"abc",
            "0bb8b87485551aa43ed54f009230450b492fead5f1cc91658775dac4a3388a0f",
            "5c41b3d0731a27a7b14bc0bf0ccded2d8751f83493404c84a88e71ffd424212e",
        ),
        (
            b"abcdef0123456789",
            "65038ac8f2b1def042a5df0b33b1f4eca6bff7cb0f9c6c1526811864e544ed80",
            "cad44d40a656e7aff4002a8de287abc8ae0482b5ae825822bb870d6df9b56ca3",
        ),
        (
            b"q128_" + b"q" * 128,
            "4be61ee205094282ba8a2042bcb48d88dfbb609301c49aa8b078533dc65a0b5d",
            "98f8df449a072c4721d241a3b1236d3caccba603f916ca680f4539d2bfb3c29e",
        ),
        (
            b"a512_" + b"a" * 512,
            "457ae2981f70ca85d8e24c308b14db22f3e3862c5ea0f652ca38b5e49cd64bc5",
            "ecb9f0eadc9aeed232dabc53235368c1394c78de05dd96893eefa62b0f4757dc",
        ),
    ]
    for message, x, y in vectors:
        point = hash_to_curve(message, dst)
        assert (point.x, point.y) == (int(x, 16), int(y, 16))


def test_signed_discrete_log_endpoints_zero_and_failure():
    assert importlib.util.find_spec("efu.crypto") is not None, "bounded decoder missing"
    from efu.crypto import bounded_log, G

    for value in (-99, -1, 0, 1, 100):
        assert bounded_log(value * G, (-99, 100)) == value
    with pytest.raises(ValueError, match="bound"):
        bounded_log(101 * G, (-99, 100))


def test_reused_discrete_log_plan_with_non_square_table_size():
    from efu.crypto import bounded_log, G

    for value in [-99, -1, 0, 1, 100]:
        assert bounded_log(value * G, (-99, 100), table_size=5) == value
