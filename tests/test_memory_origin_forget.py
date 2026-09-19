"""Retraction must be turn-scoped, not just key-scoped.

The bug this covers was measured end-to-end, not reasoned about.  A user said
"remember: my emergency contact is Wang, extension 7781", which the agent stored as two
records (contact and extension).  The user then said "forget my emergency contact
information".  The router's auto-forget retracted only the record whose key matched --
the contact -- leaving the extension active, and on the next turn the model read that
leftover record back and answered "your emergency contact extension is 7781": a fact the
user had just revoked.

Records now carry the ``origin`` of the user turn they came from, and retraction can
retire a whole origin group, so the unit of forgetting is the unit of telling.
"""

from __future__ import annotations

import sqlite3
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from V2_dpskw.memory_os_v2 import (
    MemoryOSV2,
    MemoryRecordV2,
    MemoryRouterV2,
    PagedMemoryBankV2,
    STATUS_ACTIVE,
    STATUS_RETRACTED,
    memory_record_to_dict,
)
from V2_dpskw.qwen_integration import (
    compose_corrected_evidence,
    infer_memory_metadata,
    memory_fact_sentence,
    memory_origin,
)
from V2_dpskw.tiered_memory_store_v2 import TieredMemoryStoreV2, _pack_tensor

TURN = "记一下：我的紧急联系人是 王工，电话分机 7781。"
OTHER_TURN = "记一下：我的常用编辑器是 VSCode。"


def _bank(key_dim: int = 16, **kwargs) -> PagedMemoryBankV2:
    return PagedMemoryBankV2(
        key_dim,
        router=MemoryRouterV2(key_dim, router_dim=8, num_heads=2, max_hops=3),
        page_capacity=8,
        max_pages=512,
        hot_pages=4,
        top_k_pages=4,
        top_k_records=8,
        max_hops=3,
        coarse_index_bits=8,
        **kwargs,
    )


class MemoryOriginForgetTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.bank = _bank()

    def _write_turn(self, origin: str, **fields):
        return self.bank.write(
            key=torch.randn(16),
            origin=origin,
            confidence=0.95,
            **fields,
        )[0]

    def test_forget_retires_every_record_from_the_turn(self) -> None:
        """The measured leak: the sibling record must not survive the forget."""

        origin = memory_origin(TURN)
        contact = self._write_turn(
            origin, text="user 的 紧急联系人 是 王工。", entity="user",
            attribute="紧急联系人", value="王工",
        )
        extension = self._write_turn(
            origin, text="user 的 电话分机 是 7781。", entity="user",
            attribute="电话分机", value="7781",
        )
        self.assertEqual(contact.status, STATUS_ACTIVE)
        self.assertEqual(extension.status, STATUS_ACTIVE)

        # Forgetting "the emergency contact information" addresses the contact record.
        retracted = self.bank.retract_origin(contact.origin)

        self.assertIn(contact.record_id, retracted)
        self.assertIn(extension.record_id, retracted)
        self.assertEqual(contact.status, STATUS_RETRACTED)
        self.assertEqual(extension.status, STATUS_RETRACTED)
        self.assertEqual(
            [r for r in self.bank.records.values() if r.status == STATUS_ACTIVE], []
        )

    def test_retract_origin_spares_other_turns(self) -> None:
        target = self._write_turn(
            memory_origin(TURN), text="user 的 紧急联系人 是 王工。",
            entity="user", attribute="紧急联系人", value="王工",
        )
        bystander = self._write_turn(
            memory_origin(OTHER_TURN), text="user 的 常用编辑器 是 VSCode。",
            entity="user", attribute="常用编辑器", value="VSCode",
        )

        self.bank.retract_origin(target.origin)

        self.assertEqual(target.status, STATUS_RETRACTED)
        self.assertEqual(bystander.status, STATUS_ACTIVE)

    def test_empty_origin_never_matches_anything(self) -> None:
        """Records predating this field carry no origin and must be untouchable."""

        # A blank turn must not hash to a *shared* non-empty origin, or every
        # text-less record would form one group and forgetting one would take all.
        self.assertEqual(memory_origin(""), "")
        self.assertEqual(memory_origin("   "), "")
        self.assertEqual(memory_origin(None), "")

        legacy = self._write_turn("", text="旧记录", entity="user", attribute="旧", value="x")
        blank = self._write_turn(
            memory_origin(""), text="", entity="user", attribute="空", value="x"
        )

        self.assertEqual(self.bank.retract_origin(""), [])
        self.assertEqual(self.bank.retract_origin("   "), [])
        self.assertEqual(legacy.status, STATUS_ACTIVE)
        self.assertEqual(blank.status, STATUS_ACTIVE)
        # A real origin must not sweep up the origin-less record either.
        self.assertEqual(self.bank.retract_origin(memory_origin(TURN)), [])
        self.assertEqual(legacy.status, STATUS_ACTIVE)
        self.assertEqual(blank.status, STATUS_ACTIVE)

    def test_edit_record_inherits_origin(self) -> None:
        origin = memory_origin(TURN)
        original = self._write_turn(
            origin, text="user 的 紧急联系人 是 王工。", entity="user",
            attribute="紧急联系人", value="王工",
        )

        edited = self.bank.edit_record(original.record_id, value="李工")

        self.assertEqual(edited.origin, origin)
        self.assertEqual(edited.value, "李工")
        # The group still retracts as a unit after versioning.
        self.assertEqual(len(self.bank.retract_origin(origin)), 1)
        self.assertEqual(edited.status, STATUS_RETRACTED)

    def test_record_view_exposes_origin(self) -> None:
        record = self._write_turn(
            memory_origin(TURN), text="user 的 紧急联系人 是 王工。", entity="user",
            attribute="紧急联系人", value="王工",
        )
        self.assertEqual(memory_record_to_dict(record)["origin"], memory_origin(TURN))

    def test_payload_round_trip_and_legacy_payloads(self) -> None:
        """``asdict`` is what export serialises; ``from_payload`` splats the dict back."""

        record = self._write_turn(
            memory_origin(TURN), text="user 的 紧急联系人 是 王工。", entity="user",
            attribute="紧急联系人", value="王工",
        )
        payload = asdict(record)
        self.assertEqual(MemoryRecordV2(**payload).origin, memory_origin(TURN))

        # A payload written before the field existed must still load.
        payload.pop("origin")
        self.assertEqual(MemoryRecordV2(**payload).origin, "")

    def test_origin_survives_the_tier_store(self) -> None:
        """The SQLite store keeps its own column list, so this is the real risk."""

        origin = memory_origin(TURN)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "store.db"
            store = TieredMemoryStoreV2(path, key_dim=16, page_capacity=8)
            bank = _bank(tier_store=store)
            first = bank.write(
                text="user 的 紧急联系人 是 王工。", key=torch.randn(16), origin=origin,
                entity="user", attribute="紧急联系人", value="王工", confidence=0.95,
            )[0]
            second = bank.write(
                text="user 的 电话分机 是 7781。", key=torch.randn(16), origin=origin,
                entity="user", attribute="电话分机", value="7781", confidence=0.95,
            )[0]
            store.close()

            # A fresh bank over the same file must see the origin it never had in RAM.
            reopened_store = TieredMemoryStoreV2(path, key_dim=16, page_capacity=8)
            reloaded = _bank(tier_store=reopened_store)
            reloaded._hydrate_records([first.record_id, second.record_id])
            for record_id in (first.record_id, second.record_id):
                self.assertEqual(
                    reloaded.records[record_id].origin, origin,
                    "origin was lost on the way through the store",
                )
            retracted = reloaded.retract_origin(origin)
            self.assertEqual(len(retracted), 2)
            reopened_store.close()

    def test_store_migrates_a_database_written_without_origin(self) -> None:
        """An existing store has no ``origin`` column; opening it must widen it."""

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.db"
            # Reproduce a pre-change store: same schema, no origin column.
            connection = sqlite3.connect(str(path))
            connection.executescript(
                """
                CREATE TABLE store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE records (
                    record_id TEXT PRIMARY KEY, page_id TEXT NOT NULL, text TEXT NOT NULL,
                    key BLOB NOT NULL, summary BLOB NOT NULL, memory_type TEXT NOT NULL,
                    entity TEXT NOT NULL, attribute TEXT NOT NULL, value TEXT NOT NULL,
                    timestamp INTEGER NOT NULL, importance REAL NOT NULL,
                    confidence REAL NOT NULL, source TEXT NOT NULL, status TEXT NOT NULL,
                    version INTEGER NOT NULL, supersedes TEXT NOT NULL,
                    related_ids TEXT NOT NULL, evidence TEXT NOT NULL,
                    slot_index INTEGER NOT NULL, token_ids BLOB, token_mask BLOB,
                    access_count INTEGER NOT NULL, last_access INTEGER NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT INTO records(record_id, page_id, text, key, summary, memory_type,"
                " entity, attribute, value, timestamp, importance, confidence, source,"
                " status, version, supersedes, related_ids, evidence, slot_index,"
                " token_ids, token_mask, access_count, last_access) VALUES"
                "('legacy', 'page_00000001', '旧记录', ?, ?, 'fact', 'user', '旧',"
                " 'x', 1, 0.5, 0.5, 'user', 'active', 0, '', '[]', '[]', -1, NULL, NULL, 0, 1)",
                (
                    _pack_tensor(torch.zeros(16), dtype="float32"),
                    _pack_tensor(torch.zeros(16), dtype="float32"),
                ),
            )
            connection.commit()
            connection.close()

            store = TieredMemoryStoreV2(path, key_dim=16, page_capacity=8)
            columns = {str(row[1]) for row in store.connection.execute("PRAGMA table_info(records)")}
            self.assertIn("origin", columns)
            # The row written before the column existed reads back as origin-less, and
            # the widened table still accepts ordinary writes.
            self.assertEqual(store.load_records(["legacy"])[0]["origin"], "")
            bank = _bank(tier_store=store)
            record = bank.write(
                text="新记录", key=torch.randn(16), origin=memory_origin(TURN),
                entity="user", attribute="新", value="y", confidence=0.95,
            )[0]
            self.assertEqual(store.load_records([record.record_id])[0]["origin"], memory_origin(TURN))
            store.close()

    def test_structured_write_absorbs_the_turns_unidentified_record(self) -> None:
        """The auto layer writes before the model generates, so it cannot know the turn's
        identity.  A structured write for the same turn must retire that raw record."""

        origin = memory_origin(TURN)
        # The automatic layer's record: whole sentence, no entity/attribute inferred.
        raw = self._write_turn(
            origin, text="【长期记忆证据】 事实：我叫Wpy", entity="", attribute="", value="",
        )
        self.assertEqual(raw.conflict_key(), "")
        self.assertEqual(raw.status, STATUS_ACTIVE)

        # The agent's structured write for the same turn.
        structured = self._write_turn(
            origin, text="我的名字是 Wpy", entity="user", attribute="名字", value="Wpy",
        )

        self.assertEqual(structured.status, STATUS_ACTIVE)
        self.assertEqual(raw.status, "superseded")
        active = [r for r in self.bank.records.values() if r.status == STATUS_ACTIVE]
        self.assertEqual([r.record_id for r in active], [structured.record_id])

    def test_absorption_is_confined_to_the_same_turn(self) -> None:
        """This runs on every structured write, so it must not reach other turns."""

        raw_other = self._write_turn(
            memory_origin(OTHER_TURN), text="【长期记忆证据】 事实：我住在上海",
            entity="", attribute="", value="",
        )
        structured = self._write_turn(
            memory_origin(TURN), text="我的名字是 Wpy", entity="user", attribute="名字", value="Wpy",
        )

        self.assertEqual(structured.status, STATUS_ACTIVE)
        self.assertEqual(raw_other.status, STATUS_ACTIVE)

    def test_absorption_spares_identified_siblings(self) -> None:
        """One turn can carry two facts; the second structured write must not eat the first."""

        origin = memory_origin(TURN)
        contact = self._write_turn(
            origin, text="紧急联系人是 王工", entity="user", attribute="紧急联系人", value="王工",
        )
        extension = self._write_turn(
            origin, text="电话分机是 7781", entity="user", attribute="电话分机", value="7781",
        )

        self.assertEqual(contact.status, STATUS_ACTIVE)
        self.assertEqual(extension.status, STATUS_ACTIVE)

    def test_unidentified_write_absorbs_nothing(self) -> None:
        """Only a record that *has* an identity may absorb, or the raw records would
        supersede each other and the turn's fact would be lost."""

        origin = memory_origin(TURN)
        first = self._write_turn(origin, text="事实：我叫Wpy", entity="", attribute="", value="")
        second = self._write_turn(origin, text="事实：我是开发者", entity="", attribute="", value="")

        self.assertEqual(first.status, STATUS_ACTIVE)
        self.assertEqual(second.status, STATUS_ACTIVE)

    def test_absorption_then_correction_leaves_one_active_record(self) -> None:
        """The end-to-end shape of the bug: after absorption a correction has a key to
        version, so the stale value cannot survive next to the new one."""

        origin = memory_origin(TURN)
        self._write_turn(origin, text="事实：我叫Wpy", entity="", attribute="", value="")
        structured = self._write_turn(
            origin, text="我的名字是 Wpy", entity="user", attribute="名字", value="Wpy",
        )
        corrected = self.bank.edit_record(structured.record_id, value="王五")

        active = [r for r in self.bank.records.values() if r.status == STATUS_ACTIVE]
        self.assertEqual([r.record_id for r in active], [corrected.record_id])
        self.assertEqual(corrected.value, "王五")
        self.assertEqual(corrected.version, 1)
        # The stale value is gone from the active set: the raw record that carried it was
        # absorbed, so nothing active still answers "Wpy".
        self.assertEqual([r.value for r in active], ["王五"])
        # Note for future readers: ``edit_record(value=...)`` keeps the old ``text`` on
        # purpose, so an edit that only changes the value leaves a stale sentence behind --
        # and the memory layer quotes ``text`` back to the model as the 事实 line.  Callers
        # that need the sentence to follow the value pass ``text`` as well; the agent's
        # nm2_write tool does exactly that, which is why the TUI shows the corrected sentence.
        self.assertEqual(corrected.text, "我的名字是 Wpy")

    def test_memory_os_v2_forwards_retract_origin(self) -> None:
        """The model-facing facade is what the integration actually calls."""

        os_v2 = MemoryOSV2(16, router=self.bank.router, bank=self.bank)
        origin = memory_origin(TURN)
        for attribute, value in (("紧急联系人", "王工"), ("电话分机", "7781")):
            os_v2.write(
                text=f"user 的 {attribute} 是 {value}。",
                key=torch.randn(16),
                entity="user",
                attribute=attribute,
                value=value,
                confidence=0.95,
                origin=origin,
            )
        retracted = os_v2.retract_origin(origin)
        self.assertEqual(len(retracted), 2)
        self.assertEqual(os_v2.stats()["active_records"], 0)


class CorrectedEvidenceTest(unittest.TestCase):
    """A correction must reach the thing the model actually reads.

    The model reads the record's stored evidence card, not its structured fields.  A
    correction that supplied only a new value therefore used to report success, bump the
    version and supersede the old record while every injected card still said
    ``已确认值：<old>`` -- and the model kept answering the old value.
    """

    CARD = (
        "【长期记忆证据】\n"
        "事实：我的名字是 Wpy\n"
        "实体：user\n属性：名字\n已确认值：Wpy\n"
        "可直接复述的关键短语：user；名字；Wpy\n"
        "回答要求：第一句先直接回答问题，并原样复述上面的关键短语。"
    )

    def test_value_correction_removes_the_old_value_everywhere(self) -> None:
        rebuilt = compose_corrected_evidence(
            {"entity": "user", "attribute": "名字", "value": "Wpy", "text": self.CARD},
            value="王五",
        )
        self.assertIsNotNone(rebuilt)
        assert rebuilt is not None
        self.assertNotIn("Wpy", rebuilt)
        self.assertIn("事实：我的名字是 王五", rebuilt)
        self.assertIn("已确认值：王五", rebuilt)
        self.assertIn("关键短语：user；名字；王五", rebuilt)

    def test_the_sentence_is_reused_and_natural(self) -> None:
        """Substituting in place keeps the original phrasing rather than emitting labels."""

        rebuilt = compose_corrected_evidence(
            {"entity": "user", "attribute": "名字", "value": "Wpy", "text": self.CARD},
            value="王五",
        )
        assert rebuilt is not None
        self.assertIn("事实：我的名字是 王五", rebuilt)

    def test_value_absent_from_the_sentence_falls_back(self) -> None:
        rebuilt = compose_corrected_evidence(
            {"entity": "user", "attribute": "分机", "value": "7781", "text": "记一下：分机号码是它。"},
            value="9999",
        )
        assert rebuilt is not None
        self.assertNotIn("7781", rebuilt)
        self.assertIn("9999", rebuilt)

    def test_nothing_to_rebuild_returns_none(self) -> None:
        """No field supplied means the caller is not correcting a fact -- leave it alone."""

        self.assertIsNone(compose_corrected_evidence(
            {"entity": "user", "attribute": "名字", "value": "Wpy", "text": self.CARD},
        ))
        # A record with no structured identity cannot be rendered better than it already is.
        self.assertIsNone(compose_corrected_evidence({"entity": "", "attribute": "", "value": "", "text": "x"}, value="y"))
        self.assertIsNone(compose_corrected_evidence({"entity": "user", "attribute": "名字", "value": "Wpy", "text": "x"}, value=""))

    def test_plain_sentence_records_are_handled(self) -> None:
        """Records written by the agent hold a plain sentence, not a card."""

        self.assertEqual(memory_fact_sentence("记一下：我的名字是 王五。"), "记一下：我的名字是 王五。")
        rebuilt = compose_corrected_evidence(
            {"entity": "user", "attribute": "名字", "value": "Wpy", "text": "我的名字是 Wpy。"},
            value="王五",
        )
        assert rebuilt is not None
        self.assertIn("事实：我的名字是 王五。", rebuilt)
        self.assertNotIn("Wpy", rebuilt.replace("回答要求", ""))


class AttributeExtractionTest(unittest.TestCase):
    """A drive-letter colon is not a key/value separator.

    Measured: 「记住：我的评测报告放在 E:\\deepseek\\artifacts。」 states a place with 放在
    rather than 是, so the parser fell through to the bare colon in its operator list and split
    on the drive letter, filing the fact under attribute 「评测报告放在 E」 with value
    「\\deepseek\\artifacts」.  The same fact written by the agent used 「评测报告路径」, so the
    two keys could never match, could never version each other, and the stale path stayed
    active and ranked above the corrected one.
    """

    def test_paths_are_not_split_on_the_drive_colon(self) -> None:
        for sentence in (
            "记住：我的评测报告放在 E:\\deepseek\\artifacts。",
            "改了：我的评测报告以后放到 H:\\Memory\\agent_lab\\runs。",
        ):
            meta = infer_memory_metadata(sentence)
            self.assertNotIn("E", str(meta.get("attribute")), sentence)
            self.assertNotIn("H", str(meta.get("attribute")), sentence)
            # No attribute is better than a wrong one: it stays an unstructured episode rather
            # than claiming a conflict key that nothing else can match.
            self.assertIsNone(meta.get("attribute"), sentence)

    def test_a_colon_after_cjk_is_still_a_separator(self) -> None:
        meta = infer_memory_metadata("我的密钥：abc")
        self.assertEqual(meta.get("attribute"), "密钥")
        self.assertEqual(meta.get("value"), "abc")

    def test_the_copula_still_wins_for_paths(self) -> None:
        meta = infer_memory_metadata("我的评测报告目录是 E:\\deepseek\\artifacts")
        self.assertEqual(meta.get("attribute"), "评测报告目录")
        self.assertEqual(meta.get("value"), "E:\\deepseek\\artifacts")

    def test_colons_inside_values_survive(self) -> None:
        self.assertEqual(infer_memory_metadata("我的时区是 UTC+8").get("value"), "UTC+8")
        self.assertEqual(infer_memory_metadata("我的密钥是 abc:def").get("value"), "abc:def")


if __name__ == "__main__":
    unittest.main()
