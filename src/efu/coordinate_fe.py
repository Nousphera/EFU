"""Coordinate-labelled decentralized FE using the upstream PyMIFE primitive.

Each expanded clustered vector uses d ciphertexts for d model coordinates.
See README.md for the protocol and numerical representation.
Keys use OS entropy; the training seed never seeds cryptographic key generation.
"""
from dataclasses import dataclass
import math
from numbers import Integral
import secrets

import numpy as np
from fastecdsa.curve import P256
from mife.data.fastecdsa_wrapper import WrapCurve
from mife.multiclient.decentralized.ddh import FeDDHMultiClientDec as Scheme


@dataclass(frozen=True)
class Message:
    client_id: int
    round_id: int
    ciphertexts: tuple
    share: object


def _tag(session, round_id, position):
    return session + int(round_id).to_bytes(8, "big") + int(position).to_bytes(8, "big")


class Participant:
    """Client state. Only this object retains its own private key."""
    def __init__(self, key, sizes, dimension, precision, max_abs, session):
        self._key = key
        self._sizes = tuple(sizes)
        self._dimension = dimension
        self._scale = 10 ** precision
        self._max_abs = max_abs
        self._session = session
        self._used_rounds = set()

    def encrypt(self, vector, *, round_id):
        if not isinstance(round_id, Integral) or not 0 < round_id < 2**64:
            raise ValueError("round_id must be a positive uint64")
        if round_id in self._used_rounds:
            raise ValueError("This client has already encrypted this round")
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (self._dimension,) or not np.isfinite(values).all():
            raise ValueError("Expected a finite vector with the configured dimension")
        if np.max(np.abs(values)) > self._max_abs:
            raise ValueError("Weight exceeds the agreed public max_abs bound")
        encoded = np.rint(values * self._scale).astype(np.int64)
        # Claim the label before encryption so an interrupted call cannot reuse it.
        self._used_rounds.add(round_id)
        ciphertexts = tuple(Scheme.encrypt([int(x)], _tag(self._session, round_id, j), self._key)
                            for j, x in enumerate(encoded))
        share = Scheme.keygen([[int(n)] for n in self._sizes], self._key)
        return Message(self._key.index, round_id, ciphertexts, share)


class Aggregator:
    """Server state: public parameters and approved function, no client secrets.

    Shape/label checks are application validation, not authenticated transport.
    The simulator does not isolate client objects into separate processes.
    """
    def __init__(self, public, sizes, dimension, precision, max_abs, session):
        self.public = public
        self._sizes = tuple(sizes)
        self._dimension = dimension
        self._scale = 10 ** precision
        self._session = session
        limit = math.ceil(max_abs * self._scale + 0.5) * sum(sizes)
        self._bound = (-limit, limit)

    def aggregate(self, messages, *, round_id):
        if len(messages) != len(self._sizes) or sorted(m.client_id for m in messages) != list(range(len(self._sizes))):
            raise ValueError("Exactly the configured participants must contribute once")
        messages = sorted(messages, key=lambda m: m.client_id)
        expected_function = [[int(n)] for n in self._sizes]
        for message in messages:
            if message.round_id != round_id:
                raise ValueError("Mixed round identifiers")
            if len(message.ciphertexts) != self._dimension or message.share.y != expected_function:
                raise ValueError("Ciphertext dimension or authorized function differs")
            if any(c.tag != _tag(self._session, round_id, j) for j, c in enumerate(message.ciphertexts)):
                raise ValueError("Ciphertext round/coordinate label differs")
        shares = [m.share for m in messages]
        return np.array([
            Scheme.decrypt([m.ciphertexts[j] for m in messages], _tag(self._session, round_id, j),
                           self.public, shares, self._bound)
            / (self._scale * sum(self._sizes))
            for j in range(self._dimension)
        ], dtype=np.float64)


def setup(sizes, *, dimension, precision=3, max_abs=8.0):
    """Simulate public setup and authenticated pairwise client key exchange.

    Return independent client states and a server with public parameters only.
    Every active participant must be fixed before setup; make a new session when
    participation changes. The public bound is checked before any encryption.
    """
    sizes = tuple(sizes)
    if len(sizes) < 2 or any(not isinstance(n, Integral) or n <= 0 for n in sizes):
        raise ValueError("At least two clients with positive integer sample counts are required")
    if not isinstance(dimension, Integral) or dimension <= 0:
        raise ValueError("dimension must be positive")
    if not isinstance(precision, Integral) or not 0 <= precision <= 6:
        raise ValueError("precision must be an integer between 0 and 6")
    if not math.isfinite(max_abs) or max_abs <= 0 or max_abs * 10**precision >= 2**62:
        raise ValueError("max_abs must be finite, positive and safely encodable")
    public = Scheme.generate(len(sizes), 1, WrapCurve(P256))
    keys = [public.generate_party(i) for i in range(len(sizes))]
    for i, key in enumerate(keys):
        for j, other in enumerate(keys):
            if i != j:
                key.exchange(j, other.get_exc_public_key())
    for key in keys:
        key.generate_share()
    session = secrets.token_bytes(32)
    clients = [Participant(key, sizes, dimension, precision, max_abs, session) for key in keys]
    return clients, Aggregator(public, sizes, dimension, precision, max_abs, session)
