"""Prompt templates for the partisan classifier."""
BASE_SYSTEM_PROMPT = """
You classify US FOIA-released documents for partisan political content.
Treat Democrats and Republicans symmetrically. Base judgments only on the provided text.
Return STRICT JSON matching the requested schema.
""".strip()

CLASSIFICATION_TEMPLATE = """
You will receive the text of a document released under the US Freedom of Information Act (FOIA).
Carefully read the document text, then classify it per the schema.

Definitions (apply these equally to Democrats and Republicans):
1. Partisan actors include the Democratic and Republican parties, their candidates,
   elected officials, party committees, and campaign staff.
2. Wrongdoing includes allegations or evidence of illegal acts, corruption, serious ethics
   violations, or abuse of office. Policy disagreement alone is not wrongdoing.
3. Favorability reflects whether the document portrays a party positively or negatively.

Return JSON with keys:
- political_relevance: "none" | "low" | "high"
- main_partisan_targets: list of {name, party (D/R/mixed/unknown), role}
- wrongdoing_assessment: {overall_wrongdoing_probability, wrongdoing_by_party: {D, R}}
- favorability_assessment: {overall_valence_party: {D, R}, favorability_scores: {D, R}}
- notes: 1-3 sentences of rationale.

DOCUMENT_ID: {doc_id}
DOCUMENT_TEXT:
\"\"\"{doc_text}\"\"\"
""".strip()
