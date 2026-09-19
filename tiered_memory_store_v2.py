"""Disk-backed page storage for Natural Memory v2.

The neural router never reads this module directly.  ``PagedMemoryBankV2``
uses it as an owned storage tier when a deployment needs more records than
RAM can comfortably retain.  Keys and token ids are stored as compact binary
blobs; SQLite is used only for durable metadata, page locality and recovery.
The backend is deliberately dependency-free beyond PyTorch and the Python
standard library.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch import Tensor


def _pack_tensor(value: Optional[Tensor], *, dtype: str) -> Optional[bytes]:
    if value is None:
        return None
    tensor = value.detach().cpu().contiguous()
    if dtype == "float32":
        tensor = tensor.float()
        raw = tensor.numpy().tobytes()
    elif dtype == "int32":
        tensor = tensor.to(torch.int32)
        raw = tensor.numpy().tobytes()
    elif dtype == "bool":
        raw = tensor.bool().numpy().tobytes()
    else:
        raise ValueError(f"unsupported tensor dtype: {dtype}")
    return struct.pack("<I", int(tensor.numel())) + raw


def _unpack_tensor(value: Optional[bytes], *, dtype: str) -> Optional[Tensor]:
    if value is None:
        return None
    if len(value) < 4:
        raise ValueError("corrupt tensor blob")
    count = struct.unpack("<I", value[:4])[0]
    payload = value[4:]
    if dtype == "float32":
        item_size = 4
        tensor = torch.frombuffer(bytearray(payload), dtype=torch.float32).clone()
    elif dtype == "int32":
        item_size = 4
        tensor = torch.frombuffer(bytearray(payload), dtype=torch.int32).clone().to(torch.long)
    elif dtype == "bool":
        item_size = 1
        tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone().bool()
    else:
        raise ValueError(f"unsupported tensor dtype: {dtype}")
    if len(payload) != count * item_size or tensor.numel() != count:
        raise ValueError("corrupt tensor blob length")
    return tensor


class TieredMemoryStoreV2:
    """Recoverable page/record store used by the warm and cold tiers."""

    def __init__(
        self,
        path: str | Path,
        *,
        key_dim: int,
        page_capacity: int = 32,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key_dim = int(key_dim)
        self.page_capacity = int(page_capacity)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,
        )
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS store_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pages (
                    page_id TEXT PRIMARY KEY,
                    tier TEXT NOT NULL,
                    capacity INTEGER NOT NULL,
                    record_ids TEXT NOT NULL,
                    key BLOB,
                    summary BLOB,
                    importance REAL NOT NULL,
                    created_at INTEGER NOT NULL,
                    last_access INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS records (
                    record_id TEXT PRIMARY KEY,
                    page_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    key BLOB NOT NULL,
                    summary BLOB NOT NULL,
                    memory_type TEXT NOT NULL,
                    entity TEXT NOT NULL,
                    attribute TEXT NOT NULL,
                    value TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    supersedes TEXT NOT NULL,
                    related_ids TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    slot_index INTEGER NOT NULL,
                    token_ids BLOB,
                    token_mask BLOB,
                    access_count INTEGER NOT NULL,
                    last_access INTEGER NOT NULL,
                    origin TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(page_id) REFERENCES pages(page_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS records_page_idx ON records(page_id);
                CREATE INDEX IF NOT EXISTS records_status_idx ON records(status);
                CREATE INDEX IF NOT EXISTS records_text_idx ON records(text COLLATE NOCASE);
                CREATE INDEX IF NOT EXISTS records_conflict_idx
                    ON records(entity COLLATE NOCASE, attribute COLLATE NOCASE, status);
                CREATE TABLE IF NOT EXISTS quarantine (
                    record_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS coarse_buckets (
                    signature INTEGER NOT NULL,
                    page_id TEXT NOT NULL,
                    PRIMARY KEY(signature, page_id),
                    FOREIGN KEY(page_id) REFERENCES pages(page_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS coarse_page_idx ON coarse_buckets(page_id);
                """
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO store_meta(key, value) VALUES('format_version', '2')"
            )
            self.connection.execute(
                "INSERT OR REPLACE INTO store_meta(key, value) VALUES('key_dim', ?)",
                (str(self.key_dim),),
            )
            self.connection.execute(
                "INSERT OR REPLACE INTO store_meta(key, value) VALUES('page_capacity', ?)",
                (str(self.page_capacity),),
            )
            # ``CREATE TABLE IF NOT EXISTS`` leaves an existing store untouched, so a
            # database written before ``origin`` existed has to be widened in place.
            # Without this the new column would only exist in fresh stores and every
            # insert into an old one would fail with "no such column".
            columns = {
                str(row[1]) for row in self.connection.execute("PRAGMA table_info(records)")
            }
            if "origin" not in columns:
                self.connection.execute(
                    "ALTER TABLE records ADD COLUMN origin TEXT NOT NULL DEFAULT ''"
                )

    @staticmethod
    def _record_values(record: Any) -> tuple[Any, ...]:
        return (
            record.record_id,
            record.page_id,
            record.text,
            _pack_tensor(record.key, dtype="float32"),
            _pack_tensor(record.summary, dtype="float32"),
            record.memory_type,
            record.entity,
            record.attribute,
            record.value,
            int(record.timestamp),
            float(record.importance),
            float(record.confidence),
            record.source,
            record.status,
            int(record.version),
            record.supersedes,
            json.dumps(record.related_ids, ensure_ascii=False, separators=(",", ":")),
            json.dumps(record.evidence, ensure_ascii=False, separators=(",", ":")),
            int(record.slot_index),
            _pack_tensor(record.token_ids, dtype="int32"),
            _pack_tensor(record.token_mask, dtype="bool"),
            int(record.access_count),
            int(record.last_access),
            str(getattr(record, "origin", "") or ""),
        )

    def upsert_page(self, page: Any) -> None:
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO pages(page_id, tier, capacity, record_ids, key, summary,
                                   importance, created_at, last_access)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(page_id) DO UPDATE SET
                    tier=excluded.tier,
                    capacity=excluded.capacity,
                    record_ids=excluded.record_ids,
                    key=excluded.key,
                    summary=excluded.summary,
                    importance=excluded.importance,
                    last_access=excluded.last_access
                """,
                (
                    page.page_id,
                    page.tier,
                    int(page.capacity),
                    json.dumps(page.record_ids, ensure_ascii=False, separators=(",", ":")),
                    _pack_tensor(page.key, dtype="float32"),
                    _pack_tensor(page.summary, dtype="float32"),
                    float(page.importance),
                    int(page.created_at),
                    int(page.last_access),
                ),
            )

    def upsert_record(self, record: Any) -> None:
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO records(
                    record_id, page_id, text, key, summary, memory_type, entity,
                    attribute, value, timestamp, importance, confidence, source,
                    status, version, supersedes, related_ids, evidence, slot_index,
                    token_ids, token_mask, access_count, last_access, origin
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET
                    page_id=excluded.page_id,
                    text=excluded.text,
                    key=excluded.key,
                    summary=excluded.summary,
                    memory_type=excluded.memory_type,
                    entity=excluded.entity,
                    attribute=excluded.attribute,
                    value=excluded.value,
                    timestamp=excluded.timestamp,
                    importance=excluded.importance,
                    confidence=excluded.confidence,
                    source=excluded.source,
                    status=excluded.status,
                    version=excluded.version,
                    supersedes=excluded.supersedes,
                    related_ids=excluded.related_ids,
                    evidence=excluded.evidence,
                    slot_index=excluded.slot_index,
                    token_ids=excluded.token_ids,
                    token_mask=excluded.token_mask,
                    access_count=excluded.access_count,
                    last_access=excluded.last_access,
                    origin=excluded.origin
                """,
                self._record_values(record),
            )

    @staticmethod
    def _quarantine_payload(record: Any) -> str:
        payload = {
            "record_id": record.record_id,
            "text": record.text,
            "key": record.key.detach().cpu().tolist(),
            "summary": record.summary.detach().cpu().tolist(),
            "memory_type": record.memory_type,
            "entity": record.entity,
            "attribute": record.attribute,
            "value": record.value,
            "timestamp": int(record.timestamp),
            "importance": float(record.importance),
            "confidence": float(record.confidence),
            "source": record.source,
            "status": record.status,
            "version": int(record.version),
            "page_id": record.page_id,
            "supersedes": record.supersedes,
            "related_ids": list(record.related_ids),
            "evidence": list(record.evidence),
            "slot_index": int(record.slot_index),
            "token_ids": record.token_ids.detach().cpu().tolist() if record.token_ids is not None else None,
            "token_mask": record.token_mask.detach().cpu().tolist() if record.token_mask is not None else None,
            "access_count": int(record.access_count),
            "last_access": int(record.last_access),
            "origin": str(getattr(record, "origin", "") or ""),
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def upsert_quarantine(self, record: Any) -> None:
        with self._lock:
            self.connection.execute(
                "INSERT INTO quarantine(record_id, payload) VALUES(?, ?) ON CONFLICT(record_id) DO UPDATE SET payload=excluded.payload",
                (record.record_id, self._quarantine_payload(record)),
            )

    def load_quarantine(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute("SELECT payload FROM quarantine ORDER BY record_id").fetchall()
        output = []
        for (payload,) in rows:
            item = json.loads(payload)
            item["key"] = torch.tensor(item["key"], dtype=torch.float32)
            item["summary"] = torch.tensor(item["summary"], dtype=torch.float32)
            if item.get("token_ids") is not None:
                item["token_ids"] = torch.tensor(item["token_ids"], dtype=torch.long)
            if item.get("token_mask") is not None:
                item["token_mask"] = torch.tensor(item["token_mask"], dtype=torch.bool)
            output.append(item)
        return output

    def delete_quarantine(self, record_id: str) -> None:
        with self._lock:
            self.connection.execute("DELETE FROM quarantine WHERE record_id=?", (record_id,))

    def upsert_page_with_records(self, page: Any, records: Iterable[Any]) -> None:
        with self._lock:
            self.connection.execute("BEGIN")
            try:
                self.upsert_page(page)
                for record in records:
                    self.upsert_record(record)
                self.connection.execute("COMMIT")
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise

    @contextmanager
    def transaction(self):
        with self._lock:
            self.connection.execute("BEGIN")
            try:
                yield self
                self.connection.execute("COMMIT")
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise

    def replace_page_buckets(self, page_id: str, signatures: Iterable[int]) -> None:
        with self._lock:
            self.connection.execute("DELETE FROM coarse_buckets WHERE page_id=?", (page_id,))
            self.connection.executemany(
                "INSERT OR IGNORE INTO coarse_buckets(signature, page_id) VALUES(?, ?)",
                [(int(signature), page_id) for signature in set(signatures)],
            )

    def clear_coarse_buckets(self) -> None:
        with self._lock:
            self.connection.execute("DELETE FROM coarse_buckets")

    def record_keys(self) -> list[tuple[str, Tensor]]:
        """Return compact record addresses for an explicit index rebuild."""

        with self._lock:
            rows = self.connection.execute("SELECT record_id, key FROM records").fetchall()
        return [
            (str(record_id), _unpack_tensor(blob, dtype="float32"))
            for record_id, blob in rows
        ]

    def page_headers(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT page_id, tier, capacity, record_ids, key, summary, importance, created_at, last_access FROM pages ORDER BY created_at, page_id"
            ).fetchall()
        return [
            {
                "page_id": row[0],
                "tier": row[1],
                "capacity": int(row[2]),
                "record_ids": list(json.loads(row[3])),
                "key": _unpack_tensor(row[4], dtype="float32"),
                "summary": _unpack_tensor(row[5], dtype="float32"),
                "importance": float(row[6]),
                "created_at": int(row[7]),
                "last_access": int(row[8]),
            }
            for row in rows
        ]

    def load_records(self, record_ids: Iterable[str]) -> list[dict[str, Any]]:
        ids = list(dict.fromkeys(str(item) for item in record_ids))
        if not ids:
            return []
        output: list[dict[str, Any]] = []
        with self._lock:
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = self.connection.execute(
                    f"SELECT record_id, page_id, text, key, summary, memory_type, entity, attribute, value, timestamp, importance, confidence, source, status, version, supersedes, related_ids, evidence, slot_index, token_ids, token_mask, access_count, last_access, origin FROM records WHERE record_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    output.append(
                        {
                            "record_id": row[0],
                            "page_id": row[1],
                            "text": row[2],
                            "key": _unpack_tensor(row[3], dtype="float32"),
                            "summary": _unpack_tensor(row[4], dtype="float32"),
                            "memory_type": row[5],
                            "entity": row[6],
                            "attribute": row[7],
                            "value": row[8],
                            "timestamp": int(row[9]),
                            "importance": float(row[10]),
                            "confidence": float(row[11]),
                            "source": row[12],
                            "status": row[13],
                            "version": int(row[14]),
                            "supersedes": row[15],
                            "related_ids": list(json.loads(row[16])),
                            "evidence": list(json.loads(row[17])),
                            "slot_index": int(row[18]),
                            "token_ids": _unpack_tensor(row[19], dtype="int32"),
                            "token_mask": _unpack_tensor(row[20], dtype="bool"),
                            "access_count": int(row[21]),
                            "last_access": int(row[22]),
                            "origin": str(row[23] or ""),
                        }
                    )
        return output

    def find_by_text(self, text: str, *, active_status: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self.connection.execute(
                "SELECT record_id FROM records WHERE text=? COLLATE NOCASE AND status=? LIMIT 1",
                (text.strip(), active_status),
            ).fetchone()
        if row is None:
            return None
        loaded = self.load_records([row[0]])
        return loaded[0] if loaded else None

    def find_by_conflict(
        self,
        entity: str,
        attribute: str,
        *,
        active_status: str,
    ) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self.connection.execute(
                "SELECT record_id FROM records WHERE entity=? COLLATE NOCASE AND attribute=? COLLATE NOCASE AND status=? ORDER BY version DESC LIMIT 1",
                (entity.strip(), attribute.strip(), active_status),
            ).fetchone()
        if row is None:
            return None
        loaded = self.load_records([row[0]])
        return loaded[0] if loaded else None

    def active_conflicts(self, *, active_status: str) -> dict[str, str]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT entity, attribute, record_id FROM records WHERE status=? AND entity<>'' AND attribute<>'' ORDER BY version DESC, last_access DESC",
                (active_status,),
            ).fetchall()
        output: dict[str, str] = {}
        for entity, attribute, record_id in rows:
            output.setdefault(f"{entity.strip().lower()}::{attribute.strip().lower()}", record_id)
        return output

    def candidate_page_ids(
        self,
        signatures: Iterable[int],
        *,
        hot_page_ids: Iterable[str] = (),
        limit: int = 4096,
    ) -> list[str]:
        values = list(dict.fromkeys(int(item) for item in signatures))
        selected: list[str] = []
        with self._lock:
            # Preserve the exact bucket before Hamming-neighbor probes.  A
            # single combined IN query is subtly unsafe: SQLite may return
            # neighbor pages first and truncate the exact match away.
            for index, signature in enumerate(values):
                remaining = int(limit) - len(selected)
                if index == 0:
                    rows = self.connection.execute(
                        "SELECT page_id FROM coarse_buckets WHERE signature=? ORDER BY page_id",
                        (int(signature),),
                    ).fetchall()
                elif remaining > 0:
                    rows = self.connection.execute(
                        "SELECT page_id FROM coarse_buckets WHERE signature=? ORDER BY page_id LIMIT ?",
                        (int(signature), remaining),
                    ).fetchall()
                else:
                    break
                selected.extend(row[0] for row in rows)
            selected.extend(str(item) for item in hot_page_ids)
            selected = list(dict.fromkeys(selected))[:limit]
            if not selected:
                rows = self.connection.execute(
                    "SELECT page_id FROM pages ORDER BY last_access DESC, page_id LIMIT ?",
                    (min(128, int(limit)),),
                ).fetchall()
                selected = [row[0] for row in rows]
        return selected

    def set_page_tier(self, page_id: str, tier: str) -> None:
        with self._lock:
            self.connection.execute("UPDATE pages SET tier=? WHERE page_id=?", (tier, page_id))

    def count(self) -> dict[str, int]:
        with self._lock:
            pages = int(self.connection.execute("SELECT COUNT(*) FROM pages").fetchone()[0])
            records = int(self.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])
            status_rows = self.connection.execute(
                "SELECT status, COUNT(*) FROM records GROUP BY status"
            ).fetchall()
            cold = int(self.connection.execute("SELECT COUNT(*) FROM pages WHERE tier='cold'").fetchone()[0])
            warm = int(self.connection.execute("SELECT COUNT(*) FROM pages WHERE tier='warm'").fetchone()[0])
            hot = int(self.connection.execute("SELECT COUNT(*) FROM pages WHERE tier='hot'").fetchone()[0])
            quarantined = int(self.connection.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0])
        output = {
            "pages": pages,
            "records": records,
            "hot_pages": hot,
            "warm_pages": warm,
            "cold_pages": cold,
            "quarantined": quarantined,
        }
        output.update({f"status_{status}": int(count) for status, count in status_rows})
        return output

    def coarse_bucket_count(self) -> int:
        with self._lock:
            return int(self.connection.execute("SELECT COUNT(DISTINCT signature) FROM coarse_buckets").fetchone()[0])

    def flush(self) -> None:
        with self._lock:
            self.connection.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def clear(self) -> None:
        """Clear durable memory records while preserving the store schema."""

        with self._lock:
            self.connection.execute("BEGIN")
            try:
                self.connection.execute("DELETE FROM coarse_buckets")
                self.connection.execute("DELETE FROM records")
                self.connection.execute("DELETE FROM pages")
                self.connection.execute("DELETE FROM quarantine")
                self.connection.execute("COMMIT")
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            self.flush()
            self.connection.close()

    def __enter__(self) -> "TieredMemoryStoreV2":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = ["TieredMemoryStoreV2"]
