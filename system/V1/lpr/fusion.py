"""Combining readers into one plate reading, and the confidence level shown to the operator.

Rules (all keep the BRTA layout as the first test: district [মেট্রো] - letter, 6 digits):
  single    : the primary reader alone
  fallback  : the primary reader; only when its reading breaks the layout the helpers run, then MV-BM+format over all
              (papers #24/#34 study: char-YOLO + SVTRv2 fallback 90.17 %, 22 fixed / 0 broken)
  vote      : every reader runs, MV-BM+format (majority of layout-valid readings, tie -> primary reader)
  verify    : primary + the second reader always run; when both give the same layout-valid plate it is final,
              otherwise the other helpers run and MV-BM+format decides (vote accuracy at a fraction of its cost,
              and every plate gets a second opinion for the status level)
Status (what the paper #19 study found works better than a legibility classifier: the readers' own agreement):
  verified  : layout valid and a second, independent reader gives the same plate
  likely    : layout valid, no second opinion (or it disagrees but the primary is confident)
  review    : layout broken, or readers disagree -> send the crop to a human
"""
from . import metrics, vocab

RULES = ('single', 'fallback', 'vote', 'verify')


def same(a, b):
    return bool(a) and vocab.norm_text(a) == vocab.norm_text(b)


def fuse(readings, rule, primary=0):
    """readings: list of (name, text, conf) in reader order (primary first, None for readers that did not run)."""
    have = [r for r in readings if r is not None]
    if rule == 'single' or len(have) == 1:
        return readings[primary][1]
    best = readings[primary][0]
    return metrics.fuse(have, 'MV-BM+format', best)


def needs_helpers(rule, primary_text, second_text=None):
    """Should the remaining readers run? For 'verify' the second reader has already run (second_text)."""
    if rule == 'vote':
        return True
    if rule == 'fallback':
        return not metrics.is_valid_plate(primary_text)
    if rule == 'verify':
        return not (metrics.is_valid_plate(primary_text) and same(primary_text, second_text))
    return False


def status(text, readings, verifier_text=None, char_agreement=None, char_min=0.8):
    """Confidence level for one plate. readings: (name, text, conf) of the readers that ran; verifier_text: the
    reading of the version's always-on status reader, if it has one; char_agreement: share of characters on which
    the character CNNs (papers #8, #10) agree with the char-YOLO reader (V1 only)."""
    if not metrics.is_valid_plate(text):
        return 'review'
    opinions = [t for n, t, c in readings if t is not None]
    if verifier_text is not None:
        opinions.append(verifier_text)
    if len(opinions) >= 2 and sum(same(text, t) for t in opinions) >= 2:
        return 'verified'
    if char_agreement is not None:
        return 'likely' if char_agreement >= char_min else 'review'
    return 'likely' if len(opinions) < 2 else 'review'
