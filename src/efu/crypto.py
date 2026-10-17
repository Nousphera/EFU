"""P-256 arithmetic for the research DMCFE transport.

Hash-to-curve implements RFC 9380 sections 5.3.1, 6.6.2 and 8.2,
P256_XMD:SHA-256_SSWU_RO_ (tested against all J.1.1 known answers).
The linear decentralized shares follow Abdalla et al., PKC 2019, section 3,
and the vendored PyMIFE interface, with group-valued label hashing and
canonical field arithmetic. This Python implementation is not a claim of
side-channel resistance or an independently audited cryptosystem.
"""

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256, shake_256
from math import isqrt
from numbers import Integral
from secrets import randbelow

from Crypto.PublicKey import ECC
from Crypto.Protocol.DH import key_agreement
from fastecdsa.curve import P256
from fastecdsa.point import Point

G = P256.G
Q = P256.q
P = P256.p
ZERO = 0 * G
MAX_LOG_WIDTH = 10**10


def integer(value, name, low=0, high=2**64 - 1):
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not low <= int(value) <= high
    ):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _expand_xmd(message, dst, length):
    if len(dst) > 255:
        dst = sha256(b"H2C-OVERSIZE-DST-" + dst).digest()
    if not 0 < length <= min(65535, 255 * 32):
        raise ValueError("Invalid hash expansion length")
    dst = dst + bytes([len(dst)])
    first = sha256(
        bytes(64) + message + length.to_bytes(2, "big") + b"\0" + dst
    ).digest()
    previous = sha256(first + b"\1" + dst).digest()
    output = previous
    for counter in range(2, (length + 31) // 32 + 1):
        previous = sha256(
            bytes(a ^ b for a, b in zip(first, previous)) + bytes([counter]) + dst
        ).digest()
        output += previous
    return output[:length]


def _swu(u):
    # RFC 9380 6.6.2. All branches depend only on public label material.
    a, b, z = P256.a % P, P256.b, -10 % P
    zu2 = z * u * u % P
    denominator = (zu2 * zu2 + zu2) % P
    inverse = pow(denominator, P - 2, P)  # inv0(0) = 0
    x = (-b * pow(a, -1, P) * (1 + inverse)) % P
    if denominator == 0:
        x = b * pow(z * a % P, -1, P) % P
    gx = (x * x * x + a * x + b) % P
    y = pow(gx, (P + 1) // 4, P)
    if y * y % P != gx:
        x = zu2 * x % P
        y = pow((x * x * x + a * x + b) % P, (P + 1) // 4, P)
    if (u & 1) != (y & 1):
        y = -y % P
    return Point(x, y, curve=P256)


def hash_to_curve(message, dst=b"CLUSTER-FE-V1-P256_XMD:SHA-256_SSWU_RO_"):
    if not isinstance(message, bytes) or not isinstance(dst, bytes) or not dst:
        raise ValueError("Hash input and nonempty domain must be bytes")
    uniform = _expand_xmd(message, dst, 96)  # m=1, count=2, L=48, h_eff=1
    return _swu(int.from_bytes(uniform[:48], "big") % P) + _swu(
        int.from_bytes(uniform[48:], "big") % P
    )


@lru_cache(maxsize=128)
def hash_pair(tag):
    return (
        hash_to_curve(tag, b"CLUSTER-FE-V1-H1-P256_XMD:SHA-256_SSWU_RO_"),
        hash_to_curve(tag, b"CLUSTER-FE-V1-H2-P256_XMD:SHA-256_SSWU_RO_"),
    )


def point_key(point):
    return (point.x, point.y)


def encode_point(point):
    if point == ZERO:
        return "00"
    if point.curve != P256:
        raise ValueError("Expected P-256 point")
    return (bytes([2 + (point.y & 1)]) + point.x.to_bytes(32, "big")).hex()


def decode_point(encoded, allow_zero=True):
    if not isinstance(encoded, str):
        raise ValueError("Point encoding must be hex text")
    try:
        raw = bytes.fromhex(encoded)
    except ValueError as exc:
        raise ValueError("Invalid point encoding") from exc
    if raw.hex() != encoded:
        raise ValueError("Noncanonical point encoding")
    if raw == b"\0" and allow_zero:
        return ZERO
    if len(raw) != 33 or raw[0] not in (2, 3):
        raise ValueError("Invalid compressed P-256 point")
    x = int.from_bytes(raw[1:], "big")
    if x >= P:
        raise ValueError("Invalid point coordinate")
    gx = (x * x * x + P256.a * x + P256.b) % P
    y = pow(gx, (P + 1) // 4, P)
    if y * y % P != gx:
        raise ValueError("Point is not on P-256")
    if (y & 1) != (raw[0] & 1):
        y = -y % P
    # P-256 has cofactor one: every nonidentity curve point is in the subgroup.
    return Point(x, y, curve=P256)


@lru_cache(maxsize=4)
def _baby_table(step):
    table = {}
    current = ZERO
    for j in range(step):
        table[point_key(current)] = j
        current = current + G
    return table, -(step * G)


@lru_cache(maxsize=4)
def _log_plan(lower, step):
    table, jump = _baby_table(step)
    return table, jump, lower * G


def bounded_log(point, bounds, table_size=None):
    """Deterministic BSGS with cached baby table and inclusive signed bounds."""
    lower = integer(bounds[0], "lower bound", -Q // 2, Q // 2)
    upper = integer(bounds[1], "upper bound", lower, Q // 2)
    width = upper - lower
    if width >= Q or width > MAX_LOG_WIDTH:
        raise ValueError("Discrete-log bound exceeds the configured work limit")
    step = (
        isqrt(width) + 1
        if table_size is None
        else integer(table_size, "table size", 1, 100000)
    )
    table, jump, offset = _log_plan(lower, step)
    current = point - offset
    for i in range(width // step + 1):
        j = table.get(point_key(current))
        if j is not None and i * step + j <= width:
            return lower + i * step + j
        current = current + jump
    raise ValueError("Plaintext is outside the agreed discrete-log bound")


@dataclass
class Ciphertext:
    tag: bytes
    c: tuple


@dataclass
class FunctionShare:
    y: list
    d: tuple


class Public:
    def __init__(self, n, m, context):
        self.n, self.m, self.context = n, m, context
        self.g = G
        self.hash = hash_pair

    def generate_party(self, index):
        return Party(self, integer(index, "client ID", 0, self.n - 1))


class Party:
    def __init__(self, public, index):
        self.pub, self.index = public, index
        self.sk = [(randbelow(Q), randbelow(Q)) for _ in range(public.m)]
        self.exc_priv_key = ECC.generate(curve="P-256")
        self.exchange_key = {}
        self.share = None

    def get_exc_public_key(self):
        return self.exc_priv_key.public_key()

    def exchange(self, index, public_key):
        index = integer(index, "peer ID", 0, self.pub.n - 1)
        if (
            index == self.index
            or public_key.has_private()
            or public_key.curve != "NIST P-256"
        ):
            raise ValueError("Expected a different peer with a P-256 public key")
        self.exchange_key[index] = key_agreement(
            static_priv=self.exc_priv_key,
            static_pub=public_key,
            kdf=lambda value: sha256(value).digest(),
        )

    def generate_share(self):
        if set(self.exchange_key) != set(range(self.pub.n)) - {self.index}:
            raise ValueError("Every peer must exchange a public key")
        context = b"CLUSTER-FE-V1-ZERO-SHARE\0" + self.pub.context
        self.share = []
        for owner in range(self.pub.n):
            row = []
            for slot in range(self.pub.m):
                pair = []
                for component in range(2):
                    label = (
                        context
                        + owner.to_bytes(8, "big")
                        + slot.to_bytes(8, "big")
                        + bytes([component])
                    )
                    total = 0
                    for peer, secret in self.exchange_key.items():
                        value = (
                            int.from_bytes(shake_256(secret + label).digest(48), "big")
                            % Q
                        )
                        total += value if peer > self.index else -value
                    pair.append(total % Q)
                row.append(tuple(pair))
            self.share.append(row)


class Scheme:
    @staticmethod
    def generate(n, m, context=b""):
        return Public(
            integer(n, "client count", 2), integer(m, "vector size", 1), context
        )

    @staticmethod
    def encrypt(x, tag, key):
        if len(x) != key.pub.m:
            raise ValueError("Ciphertext dimension differs")
        u1, u2 = hash_pair(tag)
        return Ciphertext(
            tag,
            tuple(
                s1 * u1 + s2 * u2 + int(value) * G for value, (s1, s2) in zip(x, key.sk)
            ),
        )

    @staticmethod
    def keygen(y, key):
        if (
            len(y) != key.pub.n
            or any(len(row) != key.pub.m for row in y)
            or key.share is None
        ):
            raise ValueError("Function dimensions or exchanged peer state differ")
        for row in y:
            for value in row:
                integer(value, "function coefficient", -Q // 2, Q // 2)
        parts = []
        for part in range(2):
            total = sum(
                int(v) * secret[part] for v, secret in zip(y[key.index], key.sk)
            )
            total += sum(
                int(y[i][j]) * key.share[i][j][part]
                for i in range(key.pub.n)
                for j in range(key.pub.m)
            )
            parts.append(total % Q)
        return FunctionShare([list(row) for row in y], tuple(parts))

    @staticmethod
    def decrypt(ciphertexts, tag, public, shares, bound):
        if len(ciphertexts) != public.n or len(shares) != public.n:
            raise ValueError("Exactly the configured participants are required")
        y = shares[0].y
        if any(s.y != y for s in shares) or len(y) != public.n:
            raise ValueError("Authorized function differs")
        u1, u2 = hash_pair(tag)
        total = ZERO
        for cipher, row in zip(ciphertexts, y):
            if cipher.tag != tag or len(cipher.c) != public.m or len(row) != public.m:
                raise ValueError("Ciphertext label or dimension differs")
            for point, coefficient in zip(cipher.c, row):
                total = total + int(coefficient) * point
        total = (
            total
            - (sum(s.d[0] for s in shares) % Q) * u1
            - (sum(s.d[1] for s in shares) % Q) * u2
        )
        return bounded_log(total, bound)
