"""Named style traits judged by Jev (TypeSafe's System One decision model), one 5-level bipolar Score each.

Level 2 is the neutral middle, and also the answer when a trait does not apply to the text's language or
situation (e.g. classical Chinese in an English line), so every trait is a scalar on the same 0..4 scale
with "no signal" at 2. Traits are allowed to overlap; each is meant to be readable on its own. Surface
counts (punctuation, particles, length) are computed by rule elsewhere, not judged.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

JEV_URL = "https://api.typesafe.ai/v1/systemone"
# Pinned: "jev-latest" moves, and labels from different versions must not be mixed silently. Labels made before
# 2026-11-03 used jev-latest, which resolved to jev-1.14.0 that day.
JEV_MODEL = "jev-1.14.0"
NOT_APPLICABLE = ("If this trait does not apply to this text's language or situation, or the text gives no "
                  "signal either way, choose the neutral middle level.")

# name: (what is judged, [level 0 .. level 4]); level 2 is always the neutral middle.
TRAITS: dict[str, tuple[str, list[str]]] = {
    "formality": ("Register of the wording: casual chat versus formal or official language (not the topic).", [
        "Very casual: slang, fillers, clipped chat (嗯嗯，好啦～ / yeah nah, gonna)",
        "Relaxed everyday speech",
        "Neutral register, or no signal",
        "Careful, standard, written-like phrasing",
        "Ceremonial or official: solemn, bureaucratic or ritual wording"]),
    "archaic": ("Modern versus archaic or classical language forms (文言 in Chinese, archaic diction in English).", [
        "Contemporary, even trendy or internet language",
        "Plain modern language",
        "Neutral, or no signal",
        "Literary flavour: a few classical words, set phrases, 成语",
        "Largely classical or archaic grammar and vocabulary (之乎者也, thee/thou)"]),
    "deference": ("How the speaker ranks themselves against the listener in address and self-reference.", [
        "Condescending or commanding: talks down, orders",
        "Blunt, unceremonious",
        "Peer-to-peer, or no signal",
        "Polite: 请, 您, softened requests, please",
        "Humble or self-lowering: 在下, 晚辈, 小的, your humble servant"]),
    "intensity": ("Emotional intensity of the delivery, whatever the emotion.", [
        "Flat, detached, emotionless",
        "Calm and composed",
        "Ordinary feeling, or no signal",
        "Clearly emotional: exclamations, strong words",
        "Outburst: very intense, broken or repeated exclamations"]),
    "valence": ("Emotional tone of the delivery: negative versus positive feeling.", [
        "Hostile, bitter or despairing",
        "Mildly negative: annoyed, worried, sad",
        "Neutral, or no signal",
        "Mildly positive: friendly, pleased",
        "Exuberant, delighted, gushing"]),
    "playfulness": ("Serious versus playful, teasing or joking delivery.", [
        "Grave, solemn",
        "Serious, matter-of-fact",
        "Neutral, or no signal",
        "Light and friendly, a little humour",
        "Clearly teasing, joking or mischievous"]),
    "age_voice": ("How old the voice sounds from its wording alone.", [
        "Elderly or venerable voice: sage-like, old-fashioned self-address",
        "Mature adult",
        "Neutral adult, or no signal",
        "Youthful, lively",
        "Childlike or cutesy: baby talk, reduplication, 人家, cute nicknames"]),
    "ornament": ("Plain versus rhetorically ornamented language.", [
        "Bare, functional, minimal words",
        "Plain",
        "Neutral, or no signal",
        "Some imagery, idioms or set phrases",
        "Heavily ornamented: parallelism, poetic imagery, metaphor, allusion"]),
    "spokenness": ("Composed written prose versus spontaneous speech.", [
        "Composed prose: complete, edited sentences",
        "Mostly composed",
        "Mixed, or no signal",
        "Speech-like: loose syntax, interjections",
        "Spontaneous speech: fillers (嗯, 呃, 啊), self-corrections, trailing off, fragments"]),
    "hedging": ("How directly the speaker commits to what they say.", [
        "Absolute, categorical assertions or commands",
        "Direct",
        "Neutral, or no signal",
        "Softened: 吧, 呢, 可能, maybe, I guess",
        "Evasive or very tentative: avoids committing (或许…吧, 我也说不准)"]),
    "verbosity": ("How many words the speaker spends relative to what they say.", [
        "Terse to the point of curtness",
        "Brief",
        "Ordinary, or no signal",
        "Elaborate, adds detail and qualifiers",
        "Long-winded, rambling or digressive"]),
    "irony": ("Sincere versus ironic or sarcastic delivery.", [
        "Earnest and heartfelt",
        "Sincere",
        "Neutral, or no signal",
        "Dry or wry",
        "Openly sarcastic or mocking"]),
    "swagger": ("How the speaker presents their own standing: self-effacing versus boastful or arrogant.", [
        "Self-deprecating, insecure",
        "Modest",
        "Neutral, or no signal",
        "Self-assured, confident",
        "Boastful or arrogant: 本大爷, 本座, bragging"]),
    "warmth": ("Cold or aloof versus warm and caring toward the listener.", [
        "Cold, dismissive",
        "Reserved, distant",
        "Neutral, or no signal",
        "Kind, considerate",
        "Very warm, affectionate, nurturing"]),
    "softness": ("Rough, hard delivery versus soft, gentle delivery.", [
        "Rough, coarse, gruff",
        "Firm, hard-edged",
        "Neutral, or no signal",
        "Gentle",
        "Very soft and tender: delicate particles, soothing phrasing"]),
    "theatricality": ("Understated versus dramatic, theatrical or grandiloquent delivery.", [
        "Deadpan, understated",
        "Low-key",
        "Neutral, or no signal",
        "Expressive, a little dramatic",
        "Theatrical or grandiloquent: declamation, stage-like flourish"]),
    "nonstandard": ("Standard language versus dialect, slang, internet speak or idiosyncratic speech habits.", [
        "Textbook-standard language",
        "Standard",
        "Neutral, or no signal",
        "Some slang, dialect words or personal verbal tics",
        "Strongly nonstandard: heavy dialect, slang, catchphrases or a distinctive verbal tic"]),
}


def questions(names: list[str] | None = None) -> dict[str, Any]:
    return {name: {"type": "score", "instructions": f"{what} {NOT_APPLICABLE}", "criteria": levels}
            for name, (what, levels) in TRAITS.items() if names is None or name in names}


def judge(text: str, api_key: str | None = None, names: list[str] | None = None, retries: int = 8) -> dict[str, Any]:
    """One Jev request: every trait scored on `text`. Returns {name: (score 0..4, confidence)} plus usage."""
    body = json.dumps({"state": text, "model": JEV_MODEL, "questions": questions(names)}).encode()
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + (api_key or os.environ["TYPESAFE_API_KEY"])}
    for attempt in range(retries):
        try:
            reply = json.load(urllib.request.urlopen(urllib.request.Request(JEV_URL, body, headers), timeout=60))
            return {"scores": {k: (v["score"], v["confidence"]) for k, v in reply["answers"].items()},
                    "input_tokens": reply.get("usage", {}).get("input_tokens", 0)}
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 529) or attempt == retries - 1:
                raise RuntimeError(f"Jev HTTP {error.code}: {error.read()[:300]!r}") from None
        except Exception:  # the link drops connections now and then (IncompleteRead, RemoteDisconnected)
            if attempt == retries - 1:
                raise
        time.sleep(min(60, 2 ** attempt))
    raise AssertionError("unreachable")
