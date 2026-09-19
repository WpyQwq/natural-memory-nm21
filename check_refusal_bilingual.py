"""Spot-check bilingual refusal detection on real reply wordings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from V2_dpskw.eval_scoring import is_refusal

CASES = [
    ("zh", "关于您的 weekly 会议安排，当前长期记忆中未包含相关信息，无法回答。", True),
    ("zh", "您的协议编号未知。", True),
    ("zh", "您的培训周期是 32GB。", False),
    ("zh", "我固定每周三 22:00 做上线。", False),
    ("en", "I don't know when that happened.", True),
    ("en", "There is no mention of a charity race in the conversation.", True),
    ("en", "The date was not mentioned.", True),
    ("en", "I cannot determine that from the dialogue.", True),
    ("en", "She was unable to answer that question about her research.", True),
    ("en", "The conversation does not mention any support group.", True),
    ("en", "I'm not sure about that.", True),
    ("en", "She went to the LGBTQ support group on 7 May 2023.", False),
    ("en", "2022", False),
    ("en", "Caroline researched adoption agencies.", False),
    ("en", "Melanie painted a sunrise in 2022.", False),
]

bad = 0
for lang, reply, expected in CASES:
    got = is_refusal(reply)
    flag = "ok " if got == expected else "BAD"
    if got != expected:
        bad += 1
    print(f"{flag} [{lang}] expected={'REFUSE' if expected else 'assert'} got={'REFUSE' if got else 'assert'}  {reply[:64]}")
print(f"\nmismatches: {bad}/{len(CASES)}")
