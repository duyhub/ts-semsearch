# Mobility Track Briefing Recap — excerpts relevant to P7

**What this is:** verbatim excerpts from the sponsor's "Mobility Track by Tasco — Briefing
Recap" Google Doc ([source](https://docs.google.com/document/d/1VRrgfhpM3vW5Wmh_1nqovqN6iat2194b61Y7jg9bFns/edit),
retrieved 2026-07-08). The doc recaps Tasco's kickoff briefing for all 12 Mobility-track
challenges. Copied here: the sections that bear on our problem (P7, section 10 below), its
sibling challenge AI Search (section 9 — same judges, overlapping expectations), and the
track-wide Q&A. Questions go to the Tasco Discord channel: https://discord.gg/Xe8tzxgHE

**Why it matters:** section 10 is the sponsor's own one-paragraph definition of our
challenge; section 9 lists the query types and Vietnamese-language behaviors Tasco expects
any map search to handle — the Q&A repeats that list as the answer to "what should teams
pay attention to". Treat section 9's lists as baseline requirements for P7 too, since
section 10 positions our challenge as going *beyond* that baseline. The Q&A also confirms
(no longer just implies) a **private judging eval set** similar to the public one.

---

## Section 9 — AI Search for Tasco Map (sibling challenge, verbatim)

> Build an AI search experience for Tasco Map that understands real user queries.
>
> Users may search in many different ways:
> - By place name
> - By category, such as "cafe nearby"
> - By brand
> - By address
> - By nearby location
> - By direction
> - By full sentence
> - By latitude / longitude coordinates
>
> The AI search should handle Vietnamese language behavior, including:
> - Missing accents
> - Typos
> - Abbreviations
> - Slang
> - Informal language
> - Mixed languages
> - Incomplete queries
> - Ambiguous place names
>
> Builders will receive datasets including abbreviation dictionaries, user scenarios,
> public evaluation questions, and expected query handling.

## Section 10 — Semantic Search and Ranking (our challenge, verbatim)

> This is a more advanced search challenge.
>
> The focus is not only on classifying what type of query the user entered, but also
> understanding the meaning behind the query and ranking results more intelligently.

## Section 11 — Conversational AI Map Assistant (sibling, verbatim, partial)

> Build an AI-powered map assistant that lets users interact with the map through
> natural-language conversation.
>
> Instead of only typing keywords and manually refining searches, users should be able to
> ask questions or describe what they need in plain language.
>
> The transcript cuts off during this section, so the detailed requirements were not
> fully captured.

## What builders should focus on (verbatim, Maps-relevant items)

> - Read the detailed challenge page and dataset before building
> - Use the provided SDK / API documentation where relevant
> - Build something that can integrate with the Tasco / VETC / Tasco Map ecosystem
> - Use the public evaluation questions to test your solution
> - Focus on real user journeys, not just a generic AI demo
> - For Maps, handle Vietnamese search behavior properly
> - Prepare a working prototype, demo, or recorded walkthrough depending on the challenge
>   requirements
>
> Main takeaway: Tasco is looking for practical AI solutions that can improve internal
> productivity, increase VETC engagement, and create stronger map-based mobility experiences.

## Q&A / Clarifications (verbatim, general + Maps entries)

> **Q: Where can builders find the challenge resources?**
> A: Resources are available at the bottom of each challenge page. Depending on the
> challenge, these may include app links, SDK documentation, API documentation, datasets,
> and public evaluation questions.
>
> **Q: Are datasets provided?**
> A: Yes. Datasets are provided for the challenges covered in the briefing. These may
> include schemas, users, services, user activity, content, notification templates,
> permissions, public evaluation questions, and expected answer / handling examples
> depending on the challenge.
>
> **Q: Will the judging questions be exactly the same as the public evaluation questions?**
> A: No. The jury will use different questions or datasets, but they will be similar to the
> public evaluation questions so teams can understand the expected behavior.
>
> **Q: For Tasco Map search, what should teams pay attention to?**
> A: Vietnamese query behavior. The search system should handle missing accents, typos,
> abbreviations, slang, mixed language, incomplete queries, ambiguous place names, and
> coordinate-based searches.

(Omitted as irrelevant to P7: Tasco company overview, VETC/AI-Workspace challenge sections
4–8 and their Q&A entries.)

---

## How this maps to our requirements (analysis, not source text)

Confirmed by this doc:

- **Private eval set is sponsor-confirmed** — judging uses different-but-similar questions.
  PRD §1/NFR-6 (generalization, no fitting to public queries) is the right posture.
- Missing accents, abbreviations, slang, informal language, mixed languages → already
  required (PRD FR-1, FR-3).

Expectations this doc adds beyond the P7 problem statement — **all folded into
PRD FR-2/FR-3 and SPEC §3/§7 on 2026-07-08** (PRD §11 delta 7), except item 5 which stays
out of scope:

1. **Coordinate-based queries** — lat/lon typed into the query string resolves to a
   nearby-search anchor, not just accepted as API params → PRD FR-2, SPEC §7.
2. **Typo tolerance** — explicit mechanism: edit-distance ≤1 fuzzy match against the
   category/attribute/gazetteer vocabularies (folded BM25 alone won't absorb typos)
   → PRD FR-3, SPEC §3.
3. **Ambiguous place names** — anchor disambiguation policy: query city/district context →
   request lat/lon focus → most popular candidate → PRD FR-2, SPEC §7.
4. **Brand queries and incomplete queries** — named requirements with tests, no longer
   implicit (BM25 brand fields + popularity signal; semantic retrieval) → PRD FR-3,
   SPEC §11 tests.
5. **"By direction" queries** — routing territory; out of P7 scope (PRD non-goal), belongs
   to siblings P6/P8.

Resource lead: section 9 says builders receive **abbreviation dictionaries** — not in our
xlsx, so it likely lives on the AI Search (P6) challenge page. Grab it to seed the FR-1
abbreviation dict before hand-writing one.
