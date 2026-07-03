"""Shared text primitives — the foundation layer of the analysis package.

Pure, dependency-free text operations (stdlib only) that everything else in
``analysis`` builds on: every detector, the ``format_consistency`` rewriter, and
the ``analysis`` facade itself. Two modules:

- ``lexical`` — word-level: tokenizing, normalizing, n-grams, token-sequence
  comparison, and the stopword/content-word floor.
- ``text_segmentation`` — paragraph/sentence/dialogue splitting and block
  extraction.

They have no dependencies within ``analysis``, so they sit below both the
detectors and the rewriter rather than inside any one consumer.
"""
