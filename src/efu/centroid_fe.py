"""Compact centroid-vector FE with explicit, mapping-dependent authorization.

One fresh Config is used for one round. Ciphertexts contain kappa points,
independently masked by slot. Every distinct tuple of centroid selections
requires a functional share from every client. Requests include all mappings;
these and key-share traffic must be counted in communication measurements.

Signatures detect transport modification relative to the registered peer keys.
Deployments must authenticate that registration. This protocol does not force
an adversarial server to publish its result or erase an earlier model.
The aggregate and public mappings can reveal centroid differences through
linear equations; encryption does not hide information in authorized outputs.
Private state serialization is ONLY for local client storage, never transport.
"""

import base64
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import secrets
import zlib

import numpy as np
from Crypto.Hash import SHA256
from Crypto.PublicKey import ECC
from Crypto.Signature import DSS

from efu.crypto import (
    Q,
    ZERO,
    MAX_LOG_WIDTH,
    Scheme,
    integer,
    hash_pair,
    bounded_log,
    encode_point,
    decode_point,
    point_key,
)

VERSION = "cluster-fe-v2"
MAX_WIRE_BYTES = 256 * 1024 * 1024
MAX_PRIVATE_HEADER_BYTES = 32 * 1024 * 1024
MAX_CHUNK_COORDINATES = 65536
MAX_AUTHORIZATION_CHUNKS = 65536
MAX_DECODE_CACHE = 100000
PRIVATE_MAGIC = b"CFP2"


def _dump(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _load(raw):
    if not isinstance(raw, bytes) or len(raw) > MAX_WIRE_BYTES:
        raise ValueError("Expected a bounded byte message")

    def pairs(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError("Duplicate JSON field")
            result[name] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("Nonfinite JSON")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid JSON message") from exc
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError("Message fields differ from protocol schema")


def _hex(value, length):
    if not isinstance(value, str) or len(value) != length * 2:
        raise ValueError("Invalid hexadecimal value")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("Invalid hexadecimal value") from exc
    if raw.hex() != value:
        raise ValueError("Noncanonical hexadecimal value")
    return raw


def _b64(raw):
    return base64.b64encode(raw).decode("ascii")


def _unb64(value):
    if not isinstance(value, str):
        raise ValueError("Expected base64 text")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("Invalid base64 data") from exc
    if _b64(raw) != value:
        raise ValueError("Noncanonical base64 data")
    return raw


@dataclass(frozen=True)
class Config:
    sizes: tuple
    dimension: int
    kappa: int
    precision: int
    max_abs: float
    round_id: int
    session: str
    layout_digest: str
    version: str = VERSION

    def __post_init__(self):
        sizes = tuple(integer(n, "sample count", 1) for n in self.sizes)
        if not 2 <= len(sizes) <= 1024:
            raise ValueError("Require 2 to 1024 configured participants")
        object.__setattr__(self, "sizes", sizes)
        for name, low, high in [
            ("dimension", 1, 100_000_000),
            ("kappa", 1, 65536),
            ("precision", 0, 6),
            ("round_id", 1, 2**64 - 1),
        ]:
            object.__setattr__(
                self, name, integer(getattr(self, name), name, low, high)
            )
        if (
            isinstance(self.max_abs, bool)
            or not isinstance(self.max_abs, (int, float))
            or not math.isfinite(self.max_abs)
            or self.max_abs <= 0
        ):
            raise ValueError("max_abs must be finite and positive")
        object.__setattr__(self, "max_abs", float(self.max_abs))
        if self.max_abs * self.scale >= 2**52 or 2 * self.limit > MAX_LOG_WIDTH:
            raise ValueError(
                "Public bound exceeds exact encoding or discrete-log work limit"
            )
        _hex(self.session, 32)
        if (
            not isinstance(self.layout_digest, str)
            or len(self.layout_digest) > 128
            or self.version != VERSION
        ):
            raise ValueError("Invalid layout or protocol version")

    @classmethod
    def create(
        cls,
        sizes,
        *,
        dimension,
        kappa,
        precision=3,
        max_abs=8.0,
        round_id=1,
        session=None,
        layout_digest="",
    ):
        if isinstance(session, bytes):
            session = session.hex()
        return cls(
            tuple(sizes),
            dimension,
            kappa,
            precision,
            max_abs,
            round_id,
            session or secrets.token_hex(32),
            layout_digest,
        )

    @property
    def mapping_dtype(self):
        return np.dtype(
            "u1" if self.kappa <= 256 else "<u2" if self.kappa <= 65536 else "<u4"
        )

    @property
    def mapping_bytes(self):
        return self.dimension * self.mapping_dtype.itemsize

    @property
    def scale(self):
        return 10**self.precision

    @property
    def weights(self):
        divisor = math.gcd(*self.sizes)
        return tuple(n // divisor for n in self.sizes)

    @property
    def limit(self):
        return math.ceil(self.max_abs * self.scale + 0.5) * sum(self.weights)

    def to_bytes(self):
        return _dump(asdict(self))

    @classmethod
    def from_bytes(cls, raw):
        data = _load(raw)
        _fields(
            data,
            "sizes dimension kappa precision max_abs round_id session layout_digest version",
        )
        if not isinstance(data["sizes"], list):
            raise ValueError("Sample counts must be a list")
        return cls(**data)

    @property
    def digest(self):
        return sha256(self.to_bytes()).hexdigest()

    @property
    def tag(self):
        return b"CLUSTER-FE-V2-ROUND\0" + bytes.fromhex(self.digest)


def _mapping(value, cfg, count=None):
    array = np.asarray(value)
    if array.shape != (cfg.dimension,) or array.dtype.kind not in "iu":
        raise ValueError("Expected an integer mapping for every model coordinate")
    maximum = cfg.kappa if count is None else count
    if np.any(array < 0) or np.any(array >= maximum):
        raise ValueError("Mapping has an invalid centroid slot")
    return np.asarray(array, dtype=cfg.mapping_dtype).tobytes()


def _read_mapping(encoded, cfg, start, stop):
    raw = _unb64(encoded)
    if len(raw) != (stop - start) * cfg.mapping_dtype.itemsize:
        raise ValueError("Mapping chunk dimension differs")
    result = np.frombuffer(raw, dtype=cfg.mapping_dtype)
    if np.any(result >= cfg.kappa):
        raise ValueError("Mapping has an invalid centroid slot")
    return result


def _public_bundle(raw):
    data = _load(raw)
    _fields(data, "exchange signing")
    result = {}
    for kind, value in data.items():
        try:
            encoded = bytes.fromhex(value)
            key = ECC.import_key(encoded)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid peer public key") from exc
        if (
            key.has_private()
            or key.curve != "NIST P-256"
            or key.public_key().export_key(format="DER") != encoded
        ):
            raise ValueError("Only canonical P-256 public keys may be exchanged")
        result[kind] = key
    if result["exchange"] == result["signing"]:
        raise ValueError("Signing and exchange keys must be independent")
    if _dump(data) != raw:
        raise ValueError("Noncanonical public key bundle")
    return result


def _peers(peer_keys, cfg):
    if not isinstance(peer_keys, dict) or set(peer_keys) != set(range(len(cfg.sizes))):
        raise ValueError("Exactly the configured peer cohort is required")
    raw = {}
    keys = {}
    exchanges = set()
    signers = set()
    for index, value in peer_keys.items():
        integer(index, "peer ID", 0, len(cfg.sizes) - 1)
        if not isinstance(value, bytes) or len(value) > 4096:
            raise ValueError("Invalid peer public key encoding")
        bundle = _public_bundle(value)
        raw[index] = value
        keys[index] = bundle["signing"]
        exchanges.add(bundle["exchange"].export_key(format="DER"))
        signers.add(bundle["signing"].export_key(format="DER"))
    if len(exchanges) != len(raw) or len(signers) != len(raw):
        raise ValueError("Every participant must have independent public keys")
    digest = sha256(_dump([raw[i].hex() for i in range(len(raw))])).hexdigest()
    return raw, keys, digest


def _sign(body, key):
    signature = DSS.new(key, "deterministic-rfc6979").sign(SHA256.new(_dump(body)))
    return _dump({"body": body, "signature": signature.hex()})


def _verify(raw, keys, cfg, peer_digest, kind):
    message = _load(raw)
    _fields(message, "body signature")
    body = message["body"]
    if not isinstance(body, dict):
        raise ValueError("Invalid signed body")
    index = integer(body.get("client_id"), "client ID", 0, len(cfg.sizes) - 1)
    try:
        DSS.new(keys[index], "deterministic-rfc6979").verify(
            SHA256.new(_dump(body)), _hex(message["signature"], 64)
        )
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid message signature") from exc
    if (
        body.get("context") != cfg.digest
        or body.get("peers") != peer_digest
        or body.get("kind") != kind
    ):
        raise ValueError("Message session, cohort or kind differs")
    return body


def _submissions(messages, cfg, keys, peer_digest):
    if len(messages) != len(cfg.sizes):
        raise ValueError("Exactly the configured participants must contribute")
    output = {}
    for raw in messages:
        body = _verify(raw, keys, cfg, peer_digest, "ciphertext")
        _fields(body, "client_id context peers kind points mapping centroid_count")
        index = body["client_id"]
        if index in output:
            raise ValueError("Duplicate participants in ciphertext cohort")
        integer(body["centroid_count"], "centroid count", 1, cfg.kappa)
        _hex(body["mapping"], 32)
        if not isinstance(body["points"], list) or len(body["points"]) != cfg.kappa:
            raise ValueError("Ciphertext dimension differs")
        points = tuple(decode_point(value) for value in body["points"])
        output[index] = (raw, body, points)
    return [output[i] for i in range(len(cfg.sizes))]


def _request(raw, cfg, peer_digest):
    data = _load(raw)
    _fields(data, "version context peers ciphertexts mappings start stop")
    if (
        data["version"] != VERSION
        or data["context"] != cfg.digest
        or data["peers"] != peer_digest
    ):
        raise ValueError("Authorization session or cohort differs")
    start = integer(data["start"], "chunk start", 0, cfg.dimension - 1)
    stop = integer(data["stop"], "chunk stop", start + 1, cfg.dimension)
    if stop - start > MAX_CHUNK_COORDINATES:
        raise ValueError("Authorization chunk exceeds the bounded coordinate limit")
    if (
        not isinstance(data["ciphertexts"], list)
        or not isinstance(data["mappings"], list)
        or len(data["ciphertexts"]) != len(cfg.sizes)
        or len(data["mappings"]) != len(cfg.sizes)
    ):
        raise ValueError("Authorization cohort differs")
    for digest in data["ciphertexts"]:
        _hex(digest, 32)
    maps = [_read_mapping(value, cfg, start, stop) for value in data["mappings"]]
    return data, maps


def _private_decode(raw):
    if (
        not isinstance(raw, bytes)
        or not raw.startswith(PRIVATE_MAGIC)
        or len(raw) < 12
        or len(raw) > MAX_WIRE_BYTES
    ):
        raise ValueError("Expected a bounded private-state envelope")
    length = int.from_bytes(raw[4:12], "big")
    if not 0 < length <= MAX_PRIVATE_HEADER_BYTES or 12 + length > len(raw):
        raise ValueError("Invalid private header length")
    return _load(raw[12 : 12 + length]), raw[12 + length :]


def _private_mapping(info, payload, cfg):
    if info is None:
        if payload:
            raise ValueError("Unexpected private mapping payload")
        return None
    _fields(info, "codec length")
    if type(info["length"]) != int or info["length"] != cfg.mapping_bytes:
        raise ValueError("Private mapping length differs")
    if info["codec"] == "raw":
        raw = payload
    elif info["codec"] == "zlib":
        try:
            decoder = zlib.decompressobj()
            raw = decoder.decompress(payload, cfg.mapping_bytes + 1)
            if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
                raise ValueError("Private mapping exceeds its bounded length")
        except zlib.error as exc:
            raise ValueError("Invalid compressed private mapping") from exc
    else:
        raise ValueError("Unknown private mapping codec")
    if len(raw) != cfg.mapping_bytes or np.any(
        np.frombuffer(raw, dtype=cfg.mapping_dtype) >= cfg.kappa
    ):
        raise ValueError("Invalid private mapping length or slots")
    return raw


def _functions(maps):
    # Stable lexicographic deduplication. Independent secrets require independent
    # function shares for different tuples, even within the same round.
    return np.unique(np.stack(maps, axis=1), axis=0, return_inverse=True)


class Participant:
    def __init__(self, config, index):
        self.config = config
        self.index = integer(index, "client ID", 0, len(config.sizes) - 1)
        self._key = Scheme.generate(
            len(config.sizes), config.kappa, context=bytes.fromhex(config.digest)
        ).generate_party(self.index)
        self._signing_key = ECC.generate(curve="P-256")
        self._peers = {}
        self._peer_keys = {}
        self._peer_digest = None
        self._used = False
        self._mapping = None
        self._message = None
        self._authorized = []
        self._mapping_storage = None

    def public_key(self):
        return _dump(
            {
                "exchange": self._key.get_exc_public_key()
                .export_key(format="DER")
                .hex(),
                "signing": self._signing_key.public_key()
                .export_key(format="DER")
                .hex(),
            }
        )

    def set_peers(self, peer_keys):
        raw, keys, digest = _peers(peer_keys, self.config)
        if raw[self.index] != self.public_key():
            raise ValueError("Own public key differs from registered cohort")
        if self._peers and self._peers != raw:
            raise ValueError("Cannot change a registered peer cohort")
        if self._peers:
            return
        self._peers, self._peer_keys, self._peer_digest = raw, keys, digest
        for index, key in keys.items():
            if index != self.index:
                self._key.exchange(index, _public_bundle(raw[index])["exchange"])
        self._key.generate_share()

    def encrypt(self, centroids, mapping):
        if self._used:
            raise ValueError(
                "Centroids may be encrypted only once per round; already used"
            )
        if not self._peers:
            raise ValueError("Exchange every peer public key before encryption")
        values = np.asarray(centroids, dtype=np.float64)
        if (
            values.ndim != 1
            or not 1 <= len(values) <= self.config.kappa
            or not np.isfinite(values).all()
            or np.max(np.abs(values)) > self.config.max_abs
        ):
            raise ValueError("Centroid shape, finiteness or public bound differs")
        map_raw = _mapping(mapping, self.config, len(values))
        encoded = np.rint(values * self.config.scale).astype(np.int64)
        encoded = np.pad(encoded, (0, self.config.kappa - len(encoded)))
        self._used = True
        self._mapping = map_raw
        cipher = Scheme.encrypt(encoded, self.config.tag, self._key)
        body = {
            "client_id": self.index,
            "context": self.config.digest,
            "peers": self._peer_digest,
            "kind": "ciphertext",
            "points": [encode_point(point) for point in cipher.c],
            "mapping": sha256(map_raw).hexdigest(),
            "centroid_count": len(values),
        }
        self._message = _sign(body, self._signing_key)
        return self._message

    @property
    def authorized_complete(self):
        return (
            bool(self._authorized) and self._authorized[-1][1] == self.config.dimension
        )

    def authorize(self, request):
        if self._message is None:
            raise ValueError("Encrypt a contribution before authorization")
        data, maps = _request(request, self.config, self._peer_digest)
        digest = sha256(_dump(data)).hexdigest()
        start, stop = data["start"], data["stop"]
        if data["ciphertexts"][self.index] != sha256(self._message).hexdigest():
            raise ValueError("Own ciphertext commitment differs")
        width = self.config.mapping_dtype.itemsize
        if maps[self.index].tobytes() != self._mapping[start * width : stop * width]:
            raise ValueError("Own authorized mapping differs")
        repeated = False
        for previous_start, previous_stop, previous_digest in self._authorized:
            if (start, stop) == (previous_start, previous_stop):
                if digest != previous_digest:
                    raise ValueError(
                        "A different function transcript was already authorized"
                    )
                repeated = True
                break
            if start < previous_stop and previous_start < stop:
                raise ValueError("Authorization chunks overlap")
        if not repeated:
            expected = self._authorized[-1][1] if self._authorized else 0
            if start != expected:
                raise ValueError("Authorization chunk order leaves a coverage gap")
            if len(self._authorized) >= MAX_AUTHORIZATION_CHUNKS:
                raise ValueError("Too many authorization chunks")
        tuples, _ = _functions(maps)
        sizes = self.config.weights
        shares = []
        for selection in tuples:
            pair = []
            for part in range(2):
                total = (
                    sizes[self.index] * self._key.sk[int(selection[self.index])][part]
                )
                total += sum(
                    sizes[i] * self._key.share[i][int(slot)][part]
                    for i, slot in enumerate(selection)
                )
                pair.append(f"{total % Q:064x}")
            shares.append(pair)
        body = {
            "client_id": self.index,
            "context": self.config.digest,
            "peers": self._peer_digest,
            "kind": "authorization",
            "request": digest,
            "shares": shares,
        }
        result = _sign(body, self._signing_key)
        if len(result) > MAX_WIRE_BYTES:
            raise ValueError("Authorization exceeds bounded wire size")
        if not repeated:
            self._authorized.append((start, stop, digest))
        # Deterministic signatures make exact retries byte-identical without
        # retaining every potentially large share payload in client state.
        return result

    def export_private(self):
        """Opaque bounded bytes, ONLY for trusted local client storage.

        The mapping uses its smallest unsigned dtype. Compression is cached and
        decoded with an exact output bound; neither pickle nor executable data
        is accepted. Authorization state stores digests rather than all shares.
        """
        mapping = None
        payload = b""
        if self._mapping is not None:
            if self._mapping_storage is None:
                compressed = zlib.compress(self._mapping, 1)
                self._mapping_storage = (
                    ("zlib", compressed)
                    if len(compressed) < len(self._mapping)
                    else ("raw", self._mapping)
                )
            codec, payload = self._mapping_storage
            mapping = {"codec": codec, "length": len(self._mapping)}
        header = _dump(
            {
                "config": asdict(self.config),
                "index": self.index,
                "secret": [[f"{x:064x}" for x in pair] for pair in self._key.sk],
                "identity": f"{int(self._key.exc_priv_key.d):064x}",
                "signing": f"{int(self._signing_key.d):064x}",
                "peers": [self._peers[i].hex() for i in range(len(self._peers))],
                "used": self._used,
                "mapping": mapping,
                "message": None if self._message is None else _b64(self._message),
                "authorized": self._authorized,
            }
        )
        if (
            len(header) > MAX_PRIVATE_HEADER_BYTES
            or 12 + len(header) + len(payload) > MAX_WIRE_BYTES
        ):
            raise ValueError("Private state exceeds its bounded serialization size")
        return PRIVATE_MAGIC + len(header).to_bytes(8, "big") + header + payload

    @classmethod
    def from_private(cls, raw):
        data, payload = _private_decode(raw)
        _fields(
            data,
            "config index secret identity signing peers used mapping message authorized",
        )
        cfg = Config.from_bytes(_dump(data["config"]))
        obj = cls(cfg, data["index"])
        if not isinstance(data["secret"], list) or len(data["secret"]) != cfg.kappa:
            raise ValueError("Private slot count differs")
        secret = []
        for pair in data["secret"]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Invalid private slot")
            secret.append(
                tuple(
                    integer(
                        int.from_bytes(_hex(x, 32), "big"), "secret scalar", 0, Q - 1
                    )
                    for x in pair
                )
            )
        scalar = integer(
            int.from_bytes(_hex(data["identity"], 32), "big"),
            "identity scalar",
            1,
            Q - 1,
        )
        obj._key.sk = secret
        obj._key.exc_priv_key = ECC.construct(curve="P-256", d=scalar)
        signing = integer(
            int.from_bytes(_hex(data["signing"], 32), "big"), "signing scalar", 1, Q - 1
        )
        obj._signing_key = ECC.construct(curve="P-256", d=signing)
        if not isinstance(data["peers"], list):
            raise ValueError("Invalid private peer state")
        if data["peers"]:
            try:
                peers = {
                    i: bytes.fromhex(value) for i, value in enumerate(data["peers"])
                }
            except (ValueError, TypeError) as exc:
                raise ValueError("Invalid private peer encoding") from exc
            obj.set_peers(peers)
        if not isinstance(data["used"], bool):
            raise ValueError("Invalid private round state")
        obj._used = data["used"]
        obj._mapping = _private_mapping(data["mapping"], payload, cfg)
        obj._mapping_storage = (
            None if data["mapping"] is None else (data["mapping"]["codec"], payload)
        )
        obj._message = None if data["message"] is None else _unb64(data["message"])
        if obj._message is not None:
            if not obj._used or obj._mapping is None or not obj._peers:
                raise ValueError("Inconsistent private encryption state")
            body = _verify(
                obj._message, obj._peer_keys, cfg, obj._peer_digest, "ciphertext"
            )
            if (
                body["client_id"] != obj.index
                or body["mapping"] != sha256(obj._mapping).hexdigest()
            ):
                raise ValueError("Private mapping commitment differs")
        if (
            not isinstance(data["authorized"], list)
            or len(data["authorized"]) > MAX_AUTHORIZATION_CHUNKS
        ):
            raise ValueError("Invalid private authorization ledger")
        expected = 0
        for entry in data["authorized"]:
            if not isinstance(entry, list) or len(entry) != 3:
                raise ValueError("Invalid private authorization record")
            start = integer(entry[0], "private chunk start", 0, cfg.dimension - 1)
            stop = integer(entry[1], "private chunk stop", start + 1, cfg.dimension)
            _hex(entry[2], 32)
            if (
                start != expected
                or stop - start > MAX_CHUNK_COORDINATES
                or obj._message is None
            ):
                raise ValueError("Invalid private authorization coverage")
            obj._authorized.append((start, stop, entry[2]))
            expected = stop
        return obj


class Aggregator:
    def __init__(self, config, peer_keys):
        self.config = config
        self._peers, self._peer_keys, self._peer_digest = _peers(peer_keys, config)
        self._ciphertext_digests = None
        self._done = False
        self.last_costs = {}

    def _bind(self, ciphertext_messages):
        submissions = _submissions(
            ciphertext_messages, self.config, self._peer_keys, self._peer_digest
        )
        digests = tuple(sha256(raw).hexdigest() for raw, _, _ in submissions)
        if self._ciphertext_digests is not None and self._ciphertext_digests != digests:
            raise ValueError("A different ciphertext cohort was already bound")
        self._ciphertext_digests = digests
        return submissions

    def request_chunks(self, ciphertext_messages, mappings, chunk_size=16384):
        """Yield bounded requests over the complete signed mapping, in order."""
        chunk_size = integer(chunk_size, "chunk size", 1, MAX_CHUNK_COORDINATES)
        if math.ceil(self.config.dimension / chunk_size) > MAX_AUTHORIZATION_CHUNKS:
            raise ValueError("Chunk size would exceed the bounded authorization count")
        submissions = self._bind(ciphertext_messages)
        if len(mappings) != len(self.config.sizes):
            raise ValueError("Mapping cohort differs")
        raw_maps = []
        for mapping, (_, body, _) in zip(mappings, submissions):
            raw = _mapping(mapping, self.config, body["centroid_count"])
            if sha256(raw).hexdigest() != body["mapping"]:
                raise ValueError("Submitted full mapping commitment differs")
            raw_maps.append(raw)
        width = self.config.mapping_dtype.itemsize
        for start in range(0, self.config.dimension, chunk_size):
            stop = min(start + chunk_size, self.config.dimension)
            request = _dump(
                {
                    "version": VERSION,
                    "context": self.config.digest,
                    "peers": self._peer_digest,
                    "ciphertexts": self._ciphertext_digests,
                    "start": start,
                    "stop": stop,
                    "mappings": [
                        _b64(raw[start * width : stop * width]) for raw in raw_maps
                    ],
                }
            )
            if len(request) > MAX_WIRE_BYTES:
                raise ValueError("Authorization request exceeds bounded wire size")
            yield request

    def request(self, ciphertext_messages, mappings):
        """Convenience wrapper for a single small range; use chunks for models."""
        if self.config.dimension > MAX_CHUNK_COORDINATES:
            raise ValueError("Use request_chunks for this model dimension")
        return next(
            self.request_chunks(ciphertext_messages, mappings, self.config.dimension)
        )

    def _shares(self, raw_messages, digest, count):
        if len(raw_messages) != len(self.config.sizes):
            raise ValueError("Every configured participant must authorize")
        shares = {}
        for raw in raw_messages:
            body = _verify(
                raw, self._peer_keys, self.config, self._peer_digest, "authorization"
            )
            _fields(body, "client_id context peers kind request shares")
            if body["request"] != digest or body["client_id"] in shares:
                raise ValueError("Authorization transcript or participant differs")
            if not isinstance(body["shares"], list) or len(body["shares"]) != count:
                raise ValueError("Authorized function count differs")
            decoded = []
            for pair in body["shares"]:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("Invalid function share pair")
                decoded.append(
                    tuple(
                        integer(
                            int.from_bytes(_hex(x, 32), "big"),
                            "function share",
                            0,
                            Q - 1,
                        )
                        for x in pair
                    )
                )
            shares[body["client_id"]] = decoded
        return shares

    def aggregate_chunks(self, ciphertext_messages, chunks):
        """Consume (request, full-cohort shares) incrementally, return only after
        ordered complete coverage and all full mapping commitments are checked.
        """
        if self._done:
            raise ValueError("Round already aggregated; replay rejected")
        submissions = self._bind(ciphertext_messages)
        cfg = self.config
        cohort = len(cfg.sizes)
        u1, u2 = hash_pair(cfg.tag)
        weights = cfg.weights
        weighted = [
            [weights[i] * point for point in item[2]]
            for i, item in enumerate(submissions)
        ]
        hashers = [sha256() for _ in range(cohort)]
        result = np.empty(cfg.dimension, dtype=np.float64)
        decoded = {}
        function_cache = {}
        decode_overflow = False
        costs = {
            "ciphertext_points": cohort * cfg.kappa,
            "authorized_functions": 0,
            "authorization_chunks": 0,
            "function_share_scalars": 0,
            "ciphertext_bytes": sum(len(raw) for raw in ciphertext_messages),
            "authorization_bytes": 0,
            "request_bytes": 0,
            "total_request_delivery_bytes": 0,
            "public_key_bytes": sum(len(raw) for raw in self._peers.values()),
            "dlog_table_entries": 0,
            "dlog_decodes": 0,
        }
        expected_start = 0
        for request, share_messages in chunks:
            if costs["authorization_chunks"] >= MAX_AUTHORIZATION_CHUNKS:
                raise ValueError("Too many authorization chunks")
            data, maps = _request(request, cfg, self._peer_digest)
            start, stop = data["start"], data["stop"]
            if start != expected_start:
                raise ValueError("Chunk coverage has a gap, overlap or invalid order")
            if tuple(data["ciphertexts"]) != self._ciphertext_digests:
                raise ValueError("Chunk ciphertext commitments differ")
            for i, mapping in enumerate(maps):
                if np.any(mapping >= submissions[i][1]["centroid_count"]):
                    raise ValueError("Mapping references an unavailable centroid")
                hashers[i].update(mapping.tobytes())
            tuples, inverse = _functions(maps)
            digest = sha256(_dump(data)).hexdigest()
            shares = self._shares(share_messages, digest, len(tuples))
            table_size = min(
                100000, math.isqrt(max(1, 2 * cfg.limit) * max(1, len(tuples))) + 1
            )
            values = []
            for number, selection in enumerate(tuples):
                d0 = sum(shares[i][number][0] for i in range(cohort)) % Q
                d1 = sum(shares[i][number][1] for i in range(cohort)) % Q
                cache_key = (tuple(int(slot) for slot in selection), d0, d1)
                value = function_cache.get(cache_key)
                if value is None:
                    total = ZERO
                    for i, slot in enumerate(selection):
                        total = total + weighted[i][int(slot)]
                    total = total - d0 * u1 - d1 * u2
                    point = point_key(total)
                    value = decoded.get(point)
                    if value is None:
                        value = bounded_log(
                            total, (-cfg.limit, cfg.limit), table_size=table_size
                        )
                        costs["dlog_decodes"] += 1
                        if len(decoded) < MAX_DECODE_CACHE:
                            decoded[point] = value
                        else:
                            decode_overflow = True
                    if len(function_cache) < MAX_DECODE_CACHE:
                        function_cache[cache_key] = value
                values.append(value)
            result[start:stop] = np.asarray(values, dtype=np.float64)[inverse] / (
                cfg.scale * sum(weights)
            )
            expected_start = stop
            costs["authorization_chunks"] += 1
            costs["authorized_functions"] += len(tuples)
            costs["function_share_scalars"] += 2 * cohort * len(tuples)
            costs["authorization_bytes"] += sum(map(len, share_messages))
            costs["request_bytes"] += len(request)
            costs["total_request_delivery_bytes"] += len(request) * cohort
            costs["dlog_table_entries"] = max(costs["dlog_table_entries"], table_size)
        if expected_start != cfg.dimension:
            raise ValueError("Incomplete coordinate coverage")
        if any(
            h.hexdigest() != item[1]["mapping"] for h, item in zip(hashers, submissions)
        ):
            raise ValueError(
                "Complete mapping commitment differs from signed submission"
            )
        costs["distinct_plaintext_sums"] = None if decode_overflow else len(decoded)
        self.last_costs = costs
        self._done = True
        return result

    def aggregate(self, ciphertext_messages, request, share_messages):
        return self.aggregate_chunks(ciphertext_messages, [(request, share_messages)])
