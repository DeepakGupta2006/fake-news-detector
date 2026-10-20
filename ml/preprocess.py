"""
Text cleaning utilities used by both train_model.py and the Flask API.
Keeping this in one shared file guarantees training and inference use
the EXACT same cleaning steps (a very common bug source otherwise).
"""

import re

# A small built-in stopword list so this doesn't depend on downloading
# NLTK corpora (keeps the project runnable fully offline).
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll
he's her here here's hers herself him himself his how how's i i'd i'll
i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our
ours ourselves out over own same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them
themselves then there there's these they they'd they'll they're
they've this those through to too under until up very was wasn't we
we'd we'll we're we've were weren't what what's when when's where
where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Lowercase, strip URLs/HTML/punctuation, and (optionally) drop stopwords.

    This does NOT do stemming/lemmatization to keep the dependency list
    small (no nltk/spacy download required). For a real project, adding
    lemmatization (e.g. spaCy) tends to give a small extra accuracy boost.
    """
    if text is None:
        return ""

    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()

    if remove_stopwords:
        tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
        text = " ".join(tokens)

    return text
