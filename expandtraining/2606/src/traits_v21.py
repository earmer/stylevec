"""17-trait rubric v2.1: Level 2 = everyday plain baseline. Self-contained: NO clarifications; Jev and span-02 get byte-identical questions."""
TRAITS_V21 = {
"formality": ("Register of the wording, from heavy slang and dialect to ceremonial or academic language (not the topic).",
  ["Heavy slang, internet speak or dialect","Casual, relaxed chat","Standard everyday language","Careful formal written prose","Ceremonial, official or academic register"]),
"archaic": ("Modern versus archaic or classical language forms (文言 in Chinese, archaic diction in English).",
  ["Heavy internet, trendy or slangy language","Casual modern language","Standard modern prose, including common idioms and set phrases","Classical literary flavour: classical words, 成语 beyond everyday use","Largely Classical Chinese / archaic grammar and vocabulary (之乎者也, thee/thou)"]),
"deference": ("How the speaker ranks themselves against the listener, judged from address and self-reference. Humble self-terms count as humble even in archaic speech; asking a favour is polite, not commanding.",
  ["Condescending or commanding: talks down, orders","Blunt, unceremonious","Peer-to-peer, ordinary address","Polite: please, 请, 您, softened requests, asking a favour","Very humble: lowers themselves and raises the listener, self-effacing self-reference"]),
"intensity": ("Emotional intensity of the delivery, from flat to dramatic declamation, whatever the emotion.",
  ["Flat, deadpan, emotionless","Calm, restrained","Ordinary everyday emotional level","Clearly emotional, expressive","Intense outburst or dramatic declamation"]),
"valence": ("Emotional tone of the delivery: negative versus positive feeling.",
  ["Hostile, bitter or despairing","Mildly negative: annoyed, worried, sad","Neutral, even","Mildly positive: friendly, pleased","Strongly positive: delighted, approving, praising"]),
"playfulness": ("Serious versus playful delivery. Modest, formal or factual statements are serious.",
  ["Grave, solemn","Serious, matter-of-fact","Ordinary, neither serious nor playful","Light and friendly, a little humour, cute particles (喵, ~, 嘛)","Clearly teasing, mock-scolding, joking, nicknames or mischievous"]),
"age_voice": ("How old the voice sounds from its wording alone.",
  ["Elderly voice: old-fashioned wording","Mature adult","Ordinary adult","Youthful, lively","Childlike or cutesy: baby talk, reduplication, cute nicknames"]),
"ornament": ("Plain versus rhetorically ornamented language.",
  ["Bare, telegraphic, minimal words","Very plain and functional: a short line with no imagery or rhetoric","Ordinary everyday wording, perhaps one common idiom","Clear imagery, figurative language or set phrases","Heavily ornamented: parallelism, poetic imagery, metaphor, allusion"]),
"spokenness": ("Formal written prose versus spontaneous speech. A short, everyday spoken line is at least level 2, not composed prose.",
  ["Formal written prose: polished, edited sentences","Composed, written-like","Ordinary plain language that reads equally well spoken or written","Speech-like: loose syntax, interjections, tildes, ellipses","Spontaneous speech: fillers (嗯, 呃, 啊, um), self-corrections, trailing off, fragments"]),
"hedging": ("How directly the speaker commits to what they say.",
  ["Absolute, categorical assertions or commands","Emphatic, confident","Plain, direct statement as people ordinarily speak","Softened: 吧, 呢, 可能, maybe, I guess","Evasive or very tentative: avoids committing (或许…吧, 我也说不准)"]),
"verbosity": ("How many words the speaker spends relative to what they say.",
  ["Terse to the point of curtness","Very brief","Ordinary length for what is said","Elaborate, adds detail and qualifiers","Long-winded, rambling or digressive"]),
"irony": ("Literal sincerity versus ironic or sarcastic meaning (saying the opposite of what is meant; blunt criticism or direct complaints are literal, not ironic).",
  ["Deeply earnest, heartfelt, and sincere","Straightforward and literal (including direct praise or direct, blunt criticism)","Ordinary literal statement with no ironic signal","Dry wit, wry understatement, or subtle irony where intended meaning diverges from surface wording","Openly sarcastic or ironic: stating the opposite of what is meant (e.g., mock-praise)"]),
"swagger": ("How the speaker presents their own standing and competence relative to others: self-effacing versus superior or boastful (genuine self-deprecation is low; condescending remarks dismissing others' competence are high).",
  ["Self-deprecating, insecure, or openly admitting inferiority","Modest, unassuming, or genuinely self-critical","Ordinary, unremarkable self-presentation","Self-assured, confident, or acting superior / condescending toward others","Openly boastful, arrogant, self-aggrandizing, or disdainfully talking down to others"]),
"warmth": ("Rough or cold versus warm, gentle and caring toward the listener.",
  ["Rough, coarse, cold or hostile","Reserved, firm or distant","Everyday neutral, even-tempered","Kind, gentle, considerate; praising, welcoming, thanking or encouraging the listener","Very warm, tender, affectionate"]),
"syntactic_complexity": ("Syntactic complexity of the sentence structure.",
  ["Minimal fragments, telegraphic","Simple short clauses","Ordinary everyday sentence structure","Multi-clause with subordinate connectors and modifiers","Deeply nested syntax, long pre-nominal modifiers"]),
"abstraction": ("Concrete versus abstract content of the wording.",
  ["Purely concrete sensory or physical details and actions","Mostly concrete events and objects","Everyday balance of concrete and general","Analytical or conceptual focus","Dense abstract theory, nominalizations"]),
"ego_focus": ("Explicit speaker self-reference and personal framing in the wording (not whether the text states an opinion).",
  ["Purely impersonal, agentless, or third-person narration; zero speaker presence","Focused on external things, events, or third parties (including direct opinions/reviews stated without I/me/we/我)","Everyday balanced framing or general address","Explicit first-person perspective (I/we/我); the speaker frames statements through their own experience or actions","Strongly self-centered: the speaker's own feelings, status, or personal narrative dominate the text"]),
}
NOT_APPLICABLE = ("If this trait does not apply to this text's language or situation, or the text gives no "
                  "signal either way, choose the neutral middle level.")
def questions():
    return {n: {"type": "score", "criteria": c, "instructions": f"{w} {NOT_APPLICABLE}"} for n, (w, c) in TRAITS_V21.items()}


def calibrate(trait: str, span_score: float, path: str | None = None) -> float:
    """Map a span-02 score onto Jev's scale with the per-trait linear fit in traits_v21_calibration.json."""
    import json, os
    path = path or os.path.join(os.path.dirname(__file__), "traits_v21_calibration.json")
    c = json.load(open(path))["calibration"][trait]
    return min(4.0, max(0.0, c["a"] * span_score + c["b"]))
