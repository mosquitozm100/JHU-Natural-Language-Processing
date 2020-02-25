These files give embeddings of words and characters into vector spaces
of different dimensions.

Before using these files for any practical purpose, you should
understand what's in them.  Explore them using your findsim program.
And read below to learn something about how they were created.

These embedding vectors were trained on various corpora, using the
publicly available word2vec package.  If you’re curious, you can find
details in Mikolov et al. (2013)'s paper "Distributed Representations
of Words and Phrases and their Compositionality," available at
http://arxiv.org/abs/1301.3781. Specifically, we ran the CBOW
method. word2vec doesn’t produce vector representations for rare words
(c(w) < 5), so we first replaced rare words with the special symbol
OOL ("out of lexicon"), forcing word2vec to learn a vector
representation for OOL.

* chars-* are trained on all of the English and Spanish character
  sequences in english_spanish/train.  

* words-* are trained on the first 1 billion characters of English Wikipedia.

* words-gs-* are trained on the Wikipedia data plus gen_spam/train.
  Thus, these lexicons include embeddings of some extra words that
  appear only in the gen_spam dataset.

* words-gs-only-* are trained on only gen_spam/train.  So these
  lexicons reflect only how the words are used in the gen_spam
  dataset, without also considering how the words are used in
  Wikipedia.  As a result, the training set contains less evidence
  (it's much smaller) but that evidence is more relevant to the
  gen_spam task.  A classic bias/variance tradeoff!

In the case of words-gs-* and words-gs-only-*, we trimmed the files to
remove words that do not appear in the gen_spam data set.  This
shrinks the file size by omitting words that you weren't going to look
up anyway.

In the case of words-gs-*, we converted all words to lowercase before
computing the embeddings.  To look up a word's embedding in one of
these lexicons, you will need to lowercase it first.

Note that although a character like "z" is pronounced differently in
English and Spanish, we have given it a single embedding that is
shared between the two languages.  This wasn't necessary: it was just
to keep things simple.  Since your English language model is separate
from your Spanish language model, it would have been fine for each of
them to use embedding features specific to that language.  Similarly,
it would be fine for the gen and spam language models to use different
embeddings.  The two language models do have to share the same
vocabulary so that their scores are comparable, but they can be
different internally.
