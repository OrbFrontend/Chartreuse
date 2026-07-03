"""Prose-quality detectors — pure functions that take text (and sometimes
database model shapes) and return structured findings.

Each detector is independently testable. The only shared dependencies are the
text-primitive modules in ``analysis.text`` — ``lexical`` (word-level:
tokenizing, normalizing, n-grams, stopwords) and ``text_segmentation``
(paragraph/sentence/dialogue splitting) — plus ``database.models`` (a layer
below).
The consolidated runner that calls all detectors lives one level up in
``analysis.audit``; public result types are re-exported through the
``analysis`` facade.
"""
