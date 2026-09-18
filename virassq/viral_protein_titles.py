r"""Assign simple, auditable labels to viral DIAMOND protein titles.

The rules deliberately recognize only words present in ``best_viral_title``::

    "RNA-dependent RNA polymerase [Virus X]"
                      |
                      +--> protein_role = "rdrp"
                      +--> matched_keyword = "rna-dependent rna polymerase"

Family-specific names such as VP3 and segment labels such as L/M/S are retained
as tokens but are not converted to a functional role.  Their meaning depends
on the virus. Add a new role only when its wording has a stable function.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

# Ordered, explicit rules. The first matched role becomes the primary label.
# More than one matched functional role is preserved as an ambiguity rather
# than hidden by precedence.
PROTEIN_ROLE_PATTERNS = (
    (
        "rdrp",
        re.compile(
            r"\b(?:rdrp|rna[- ]dependent rna polymerase|"
            r"rna[- ]directed rna polymerase)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "nucleoprotein",
        re.compile(r"\b(?:nucleoprotein|nucleocapsid(?: protein)?)\b", re.IGNORECASE),
    ),
    (
        "glycoprotein",
        re.compile(
            r"\b(?:glycoprotein|envelope glycoprotein|envelope protein)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "capsid",
        re.compile(r"\b(?:capsid protein|capsid|coat protein)\b", re.IGNORECASE),
    ),
    ("matrix", re.compile(r"\bmatrix protein\b", re.IGNORECASE)),
    ("helicase", re.compile(r"\bhelicase\b", re.IGNORECASE)),
    ("protease", re.compile(r"\bprotease\b", re.IGNORECASE)),
)

# These labels describe how a database record is named, not a specific
# molecular function. They are used only when no functional rule matched.
FALLBACK_ROLE_PATTERNS = (
    ("nonstructural", re.compile(r"\bnon[- ]?structural protein\b", re.IGNORECASE)),
    ("polyprotein", re.compile(r"\bpolyprotein\b", re.IGNORECASE)),
)

NAMED_PROTEIN_PATTERNS = (
    ("vp", re.compile(r"\bVP[\s_-]*(\d+[A-Z]?)\b", re.IGNORECASE)),
    ("ns", re.compile(r"\bNS([SM]|\d+)\b", re.IGNORECASE)),
    ("glycoprotein", re.compile(r"\bG([NC])\b", re.IGNORECASE)),
    (
        "segment",
        re.compile(
            r"\b(?:([LMS])[- ]?(?:protein|segment)|segment[- ]?([LMS]))\b",
            re.IGNORECASE,
        ),
    ),
)


def _unique_in_order(values: Iterable[str]) -> tuple[str, ...]:
    """Remove duplicates without changing the rule or title order."""

    return tuple(dict.fromkeys(values))


def extract_named_protein_tokens(title: str) -> tuple[str, ...]:
    """Extract VP/NS/Gn/Gc tokens and explicit L/M/S protein labels."""

    tokens = []
    for token_kind, pattern in NAMED_PROTEIN_PATTERNS:
        for match in pattern.finditer(title):
            suffix = next(group for group in match.groups() if group is not None)
            if token_kind == "vp":
                value = f"VP{match.group(1).upper()}"
            elif token_kind == "ns":
                suffix = match.group(1)
                value = (
                    f"NS{suffix.lower() if suffix.lower() in {'s', 'm'} else suffix}"
                )
            elif token_kind == "glycoprotein":
                value = f"G{match.group(1).lower()}"
            else:
                value = suffix.upper()
            tokens.append(value)
    return _unique_in_order(tokens)


def classify_viral_protein_title(title: object) -> dict[str, object]:
    """Classify one raw DIAMOND title while preserving exact matched text."""

    if title is None or pd.isna(title) or not str(title).strip():
        return {
            "viral_protein_role": None,
            "viral_protein_matched_roles": (),
            "viral_protein_matched_keywords": (),
            "viral_protein_named_tokens": (),
            "viral_protein_role_ambiguous": False,
        }

    text = str(title)
    matched = [
        (role, match.group(0))
        for role, pattern in PROTEIN_ROLE_PATTERNS
        if (match := pattern.search(text)) is not None
    ]
    if not matched:
        matched = [
            (role, match.group(0))
            for role, pattern in FALLBACK_ROLE_PATTERNS
            if (match := pattern.search(text)) is not None
        ][:1]

    roles = _unique_in_order(role for role, _ in matched)
    return {
        "viral_protein_role": roles[0] if roles else "unknown",
        "viral_protein_matched_roles": roles,
        "viral_protein_matched_keywords": _unique_in_order(
            keyword.lower() for _, keyword in matched
        ),
        "viral_protein_named_tokens": extract_named_protein_tokens(text),
        "viral_protein_role_ambiguous": len(roles) > 1,
    }


def add_viral_protein_title_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Add title-derived columns to rows containing ``best_viral_title``."""

    labels = pd.DataFrame(
        [classify_viral_protein_title(title) for title in frame["best_viral_title"]],
        index=frame.index,
    )
    return pd.concat([frame.copy(), labels], axis=1)
