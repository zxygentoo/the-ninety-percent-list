### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ 4179174c-6ee4-520b-9c58-f6262778cb94
begin
    using CodecZstd: LibZstd
    using Downloads, Printf, Random
    using CairoMakie, PlutoUI
end

# ╔═╡ fdfcd5cc-ed29-5406-83a2-023cf5021853
md"""
# An Observation on Generalization

*Sutskever's compression argument, tested with an ordinary compressor and a novel.*

In a 2023 talk at the Simons Institute, [*An Observation on Generalization*](https://simons.berkeley.edu/talks/ilya-sutskever-openai-2023-08-14), Ilya Sutskever asked why unsupervised learning works at all, and answered with compression. This notebook turns that answer into an experiment.

**Prerequisites**

- Basic probability, including conditional probability, and logarithms.
- How a language model is trained, roughly: it predicts the next token, and gradient descent lowers its cross-entropy loss.
- No [information theory](https://en.wikipedia.org/wiki/Information_theory): the notebook introduces what it uses.
"""

# ╔═╡ c331071a-b435-4482-83f4-9f06dd3f0af3
html"""
<iframe src="https://www.youtube-nocookie.com/embed/AKMuA_TVz3A" title="An Observation on Generalization (Ilya Sutskever, Simons Institute, 2023)"
        style="width: 100%; aspect-ratio: 16 / 9; border: 0;"
        allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
"""

# ╔═╡ 2c83d185-422a-5af7-9fce-7c99ce734229
md"""
## The puzzle

Supervised learning comes with a guarantee. If a model does well on its training data, and there are more training examples than the model has degrees of freedom, it will do well on new data from the same distribution.

Unsupervised learning has no such guarantee. We train on a large unlabelled dataset ``X``, for instance by predicting the next word of web text, and hope it helps with a different task ``Y`` that we actually care about, like answering general questions. The objective we optimise is not the one we are judged on.

Why should doing well on ``X`` help with ``Y`` at all?
"""

# ╔═╡ 23bd693b-a7da-573d-8f5d-ac8b61626bd8
md"""
## The answer: exploit the shared structure

Sutskever's answer is to think about a compressor. Compress ``X`` and ``Y`` as one file. A good compressor learns as it reads, so by the time it reaches ``Y`` it can use everything it found in ``X``.

It works like an inside joke. Between friends who were there, one word can stand for a whole story; a stranger needs the story told in full. A compressor that has read ``X`` is one of the friends.

At worst, the compressor can treat ``X`` and ``Y`` separately and join the results, so compressing them together never costs more than a constant extra:

```math
C(X \cdot Y) \;\le\; C(X) + C(Y) + O(1)
```

When they share structure, it costs less. Write ``C(Y \mid X)`` for the compressed size of ``Y`` once the compressor has already seen ``X``:

```math
C(Y \mid X) \;\approx\; C(X \cdot Y) - C(X)
```

The bits that ``X`` saves then measure what the compressor learned from ``X`` that is also true of ``Y``:

```math
\text{saved}(X) = C(Y) - C(Y \mid X)
```

That, says the talk, is what unsupervised learning does: a compressor that has seen ``X`` puts whatever it found there to use on ``Y``.

Two ideas turn this from an analogy into an argument:

- **A model that predicts is a compressor**, so "compressor" can mean a language model.
- **There is a best possible compressor**, which gives "a good compressor" a precise meaning.
"""

# ╔═╡ c65512c4-bab7-44b1-a30f-4f2223cd9eca
md"""
## Prediction is compression

Suppose a model reads a text one symbol at a time and, before each symbol, gives a probability to every possible next one. An [*arithmetic coder*](https://en.wikipedia.org/wiki/Arithmetic_coding) turns those probabilities into a file, charging each symbol for how much it surprised the model:

```math
\text{cost of a symbol} = -\log_2 p \;\text{bits}
```

- An even bet is one yes-or-no answer: ``p = 1/2`` costs exactly 1 bit.
- A confident, correct guess is nearly free: ``p = 0.9`` costs 0.15 bits.
- A surprise is expensive: ``p = 0.001`` costs about 10 bits.

Add up those costs and you have the size of the file:

```math
C(x) \approx \sum_t -\log_2 p(x_t \mid x_{<t})
```

The match is close: an ideal arithmetic coder overshoots the sum by less than two bits for the whole file, not per symbol.

That sum is the [cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy) loss a language model is trained to minimise, counted in bits. Training a model to predict the next token *is* training a compressor, and its loss is the size of the file it would write.

The equivalence runs the other way too. Any compressor implies predictions. A continuation that compresses to ``k`` fewer bits is, in the compressor's eyes, ``2^k`` times more likely: 10 bits shorter, about a thousand times likelier. zstd never states a probability, but each back-reference is a bet that the text is about to repeat itself, and a small file is a run of bets that paid off.

So these two statements are one and the same, which is why the talk can say *compressor* and mean *language model*:

- "``Y`` compresses better after ``X``";
- "after reading ``X``, the model predicts ``Y`` better".

Two reference points for how far prediction can go:

- **People.** Shannon measured English this way in 1951, asking people to guess the next letter of a text, and put it at roughly 0.6 to 1.3 bits per letter.
- **Machines.** The best entries on Matt Mahoney's [Large Text Compression Benchmark](http://mattmahoney.net/dc/text.html) today are neural networks, transformers and LSTMs, predicting the next byte for an arithmetic coder.
"""

# ╔═╡ 0c3334f8-3464-4c24-a358-a10c162d80e1
md"""
## The best possible compressor

How good can a compressor get? The [*Kolmogorov complexity*](https://en.wikipedia.org/wiki/Kolmogorov_complexity) ``K(x)`` of a string is the length of the shortest program that prints it.

- The first million digits of ``\pi`` have tiny ``K``: a few lines of code compute them.
- A million coin flips almost always have ``K`` of about a million bits: there is no shorter description than the flips themselves.

``K`` is the ultimate compressor in a precise sense. For any compressor ``C`` you can write down,

```math
K(x) \;\le\; C(x) + c,
```

where the constant ``c`` is roughly the length of ``C``'s decompressor and does not depend on ``x``. The reason is short: one program that prints ``x`` is ``C``'s decompressor followed by ``C``'s file for ``x``.

No compressor beats ``K`` by more than a constant, on any data.

Its conditional form ``K(Y \mid X)`` is the shortest program that prints ``Y`` when handed ``X``. It uses every regularity that ``X`` and ``Y`` share, and their *algorithmic mutual information* is the ideal version of the saved bits measured below:

```math
I(X : Y) = K(Y) - K(Y \mid X)
```

The catch is that ``K`` cannot be computed: no program finds the shortest program for every string. In the talk it plays two roles:

- **A yardstick.** It makes "a better compressor finds more" precise.
- **A target.** Sutskever suggests that a large neural network approximates it: training by gradient descent searches a vast space of programs, encoded in the network's weights, for one that compresses the training data well.

zstd sits at the other end of the scale, a small fixed program. The experiment asks how much even that gets.
"""

# ╔═╡ dcfb1941-2f44-4294-9394-0c2baccd5fb4
md"""
## What to look for

Three predictions follow, and the rest of this notebook tests each one:

1. **Unrelated data costs nothing.** If ``X`` shares nothing with ``Y``, a good compressor ignores it: ``C(Y \mid X) \approx C(Y)``.
2. **Related data helps, and more of it helps more.** The saving grows with what ``X`` and ``Y`` have in common, and with how much ``X`` there is.
3. **A better compressor finds more.** Given the same ``X``, a compressor closer to ``K`` extracts more of what ``X`` shares with ``Y``.
"""

# ╔═╡ 1d1f89d8-4de4-526c-a60f-862c213a77b2
md"""
## Setup

Pluto installs the packages on first run. The compressor is the zstd C library, called through the plain Julia wrappers in the `CodecZstd` package; nothing outside Julia runs. The texts are downloaded once into `data/` next to this notebook.
"""

# ╔═╡ 833a76f1-a7ef-45b3-9d11-6b92f1ea3120
TableOfContents()

# ╔═╡ 1d77adbb-cc37-53f6-b0f3-635453e64314
begin
	KB = 1_000
	SEED = 1
end

# ╔═╡ ee5e727f-9b96-5369-bb82-a2685b158eec
md"""
## A compressor you can hand a book

[zstd](https://facebook.github.io/zstd/) is an everyday general-purpose compressor. Like most of its kind, it finds repeated strings and replaces each one with a back-reference, "copy 31 bytes from 4,000 bytes ago", then packs the result tightly.

It can also be given a *dictionary*: text it treats as already seen, so back-references may point into it. Hand it ``X`` as the dictionary, and the size of what it writes for ``Y`` is ``C(Y \mid X)``. That takes one call into libzstd.
"""


# ╔═╡ 188edca4-84ae-5275-8020-e2169ac6db41
"""
    compress(y; dictionary, level = 19)

Compress the bytes `y` with zstd at `level`, from 1 (fast) to 19 (thorough).
Back-references may point into `dictionary`, text the compressor treats as already seen.
"""
function compress(y; dictionary = UInt8[], level = 19)
    context = LibZstd.ZSTD_createCCtx()
    file = Vector{UInt8}(undef, LibZstd.ZSTD_compressBound(length(y)))
    n = LibZstd.ZSTD_compress_usingDict(context,
                                        file, length(file),
                                        y, length(y),
                                        dictionary, length(dictionary),
                                        level)
    LibZstd.ZSTD_freeCCtx(context)
    LibZstd.ZSTD_isError(n) == 0 || error("zstd could not compress")
    first(file, n)
end

# ╔═╡ 41817472-c4d9-5438-9b1d-6de4361262fb
"""
    C(y; given = x, level = 19)

Compressed size of `y` in bits, once zstd has seen `x` (by default, nothing).
"""
C(y; given = UInt8[], level = 19) = 8 * length(compress(y; dictionary = given, level))

# ╔═╡ 5b8da8fe-341f-58fb-b542-39295a8dafd8
"Bits that having seen `x` saves on `y`."
saved_bits(y, x; level = 19) = C(y; level) - C(y; given = x, level)

# ╔═╡ 8f96d614-b186-5f22-90f3-eee491e96641
"The bits `x` saves on `y`, as a percentage of what `y` costs alone."
saved_pct(y, x; level = 19) = 100 * saved_bits(y, x; level) / C(y; level)

# ╔═╡ 2efc182d-bac3-58f9-bb99-74a9b806b47e
BOOKS = (
    novel = 1342,     # Pride and Prejudice
    austen = [
        105,          # Persuasion
        158,          # Emma
        161,          # Sense and Sensibility
        141,          # Mansfield Park
        121,          # Northanger Abbey
    ],
    other_authors = [
        1400,         # Great Expectations (Dickens)
        145,          # Middlemarch (Eliot)
        4276,         # North and South (Gaskell)
        1260,         # Jane Eyre (Charlotte Brontë)
        768,          # Wuthering Heights (Emily Brontë)
    ],
)

# ╔═╡ 51a4d8e4-41f0-598a-8551-3be4f8766f07
DATA_DIR = joinpath(@__DIR__, "data")

# ╔═╡ e9fa7345-6560-593c-ab56-3d3dfc4b40ff
"The raw text of Gutenberg book `id`, downloaded once into `data/gutenberg`."
function fetch_book(id)
    path = joinpath(DATA_DIR, "gutenberg", "pg$id.txt")
    if !isfile(path)
        mkpath(dirname(path))
        Downloads.download("https://www.gutenberg.org/cache/epub/$id/pg$id.txt", path)
    end
    read(path, String)
end

# ╔═╡ a240f10f-2e89-5385-90de-b40f823426be
"The book itself: the text between Gutenberg's start and end markers."
function strip_boilerplate(text)
    text = replace(text, "\r\n" => "\n")
    start = findfirst(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", text)
    stop = findfirst(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text)
    strip(SubString(text, nextind(text, last(start)), prevind(text, first(stop))))
end

# ╔═╡ a9d82f5b-d5ec-5333-9450-7c82e9c20de9
"One form for the things transcribers do differently: quotes, dashes, italics and line breaks."
function normalise(text)
    text = replace(text, r"[“”„]" => "\"", r"[‘’]" => "'", r"[—–]" => "--", "_" => "")
    text = replace(text, r"[ \t]+\n" => "\n", r"\n[ \t]+" => "\n")   # spaces at line ends and starts
    text = replace(text, r"(?<!\n)\n(?!\n)" => " ")                   # join lines wrapped inside a paragraph
    replace(text, r"\n{3,}" => "\n\n", r" {2,}" => " ")
end

# ╔═╡ 3b5e9341-0a8e-5a31-804a-b5745e2fed23
"Gutenberg book `id` as clean bytes."
book(id) = id |> fetch_book |> strip_boilerplate |> normalise |> Vector{UInt8}

# ╔═╡ 0f5b588a-3f8e-54c8-b8ec-237f8cd2d5ab
novel = book(BOOKS.novel);

# ╔═╡ cdd84fd7-846e-5ddf-9fc5-e70d6c52f4cc
Y_BYTES = 300KB      # over 256 KB, so that zstd levels compare fairly (see below)

# ╔═╡ 501b4ad0-8a82-59fa-8019-3ab54aaa35f7
X_BYTES = 400KB      # each X in the ladder; at most the length of the earlier chapters

# ╔═╡ 66503f0e-d0f1-5908-bfea-e744f83c1824
Y = last(novel, Y_BYTES);

# ╔═╡ 73a742ce-89b4-5dc4-95f3-6c5c7b57559f
md"""
## Two extremes

Before the experiment, a baseline and its two ends:

- **``Y`` alone** costs some number of bits.
- **``Y`` given itself** as the dictionary costs almost nothing: all the compressor has to write is "copy all of it".
- **``Y`` given random noise** should cost what ``Y`` costs alone, since noise contains nothing to copy.
"""

# ╔═╡ 0658cb43-1ee8-54f3-9576-0f2c4307708e
"`n` random bytes, the same ones on every run."
random_bytes(n) = rand(Xoshiro(SEED), UInt8, n)

# ╔═╡ 60919141-e13c-5016-a555-a6c0363da036
extremes = (
    alone = C(Y),
    given_itself = C(Y; given = copy(Y)),   # a copy: zstd treats a dictionary that overlaps its input in memory as overwritten
    given_noise = C(Y; given = random_bytes(X_BYTES)),
)

# ╔═╡ fa29009c-3120-5d82-a1c5-6a76e9bf80de
md"""
## Joint or conditional

The talk writes the experiment as compressing the concatenation ``X \cdot Y``. The dictionary asks for ``C(Y \mid X)`` directly, without re-encoding ``X``. The two should agree closely. Here they are for the most closely related ``X`` we have, the novel's own earlier chapters:
"""

# ╔═╡ 7567ec87-fa71-5049-a2a6-23ab6b195df8
earlier_chapters = novel[1:end-Y_BYTES];

# ╔═╡ 40abbd27-e0e5-5293-984d-0585b00c6a27
joint_and_conditional = (
    joint = C([earlier_chapters; Y]) - C(earlier_chapters),
    conditional = C(Y; given = earlier_chapters),
)

# ╔═╡ 2674ef0f-4e18-50ab-8ed8-ed4aaf966df8
md"""
## The ladder

Four kinds of ``X``, from nothing in common with ``Y`` to the same book, all the same size:

| ``X`` | what it shares with ``Y`` |
|:--|:--|
| random noise | nothing |
| 19th-century English novels by other authors | the language and the genre |
| Austen's other novels | the author: her vocabulary, rhythm and turns of phrase |
| the earlier chapters of *Pride and Prejudice* | the book: its characters, places and story |

If the argument is right, the saving climbs as we go down the table, and the first row saves nothing.
"""

# ╔═╡ d2fbbfc0-ff05-5358-b99e-ae661108602e
"`n` bytes taken evenly from each of `books`, so that every size has the same mix."
function blend(books, n)
    share = cld(n, length(books))
    first(reduce(vcat, first.(books, share)), n)
end

# ╔═╡ d936643b-44b6-5012-8246-bbfca7b9c11d
"Most bytes `blend` can take from `books` before one of them runs out."
blend_limit(books) = length(books) * minimum(length, books)

# ╔═╡ 6c2a096d-1af9-5d41-ae83-dcdac11042e1
austen = book.(BOOKS.austen);

# ╔═╡ 20dc88ba-af86-5f9a-af0d-2993416832f4
other_authors = book.(BOOKS.other_authors);

# ╔═╡ a5ccc482-782d-5ac8-91b8-5cc7e6b2446a
RUNGS = (
    noise = (
        name = "random noise",
        sample = random_bytes,
        available = typemax(Int),
    ),
    other_authors = (
        name = "same language, other authors",
        sample = n -> blend(other_authors, n),
        available = blend_limit(other_authors),
    ),
    austen = (
        name = "same author, other books",
        sample = n -> blend(austen, n),
        available = blend_limit(austen),
    ),
    own_book = (
        name = "same book, earlier chapters",
        sample = n -> first(earlier_chapters, n),
        available = length(earlier_chapters),
    ),
);

# ╔═╡ a3ac1236-770e-57b8-81c9-71c23f2e9a84
ladder = map(RUNGS) do rung
    X = rung.sample(X_BYTES)
    (; rung.name, bits = saved_bits(Y, X), pct = saved_pct(Y, X))
end

# ╔═╡ 0bfb6f43-c544-57ad-9838-d86502a2d115
X_SIZES = [1KB, 3KB, 10KB, 30KB, 100KB, 400KB, 1_000KB, 2_000KB]

# ╔═╡ f7a08251-3b7b-52d7-b0dc-7ba92760df3b
growth = map(RUNGS) do rung
    sizes = filter(≤(rung.available), X_SIZES)
    (; rung.name, sizes, pcts = [saved_pct(Y, rung.sample(n)) for n in sizes])
end

# ╔═╡ ff1487ad-75fb-5cf0-be5b-94bbaff556fd
md"""
## Compress *harder* helps

zstd's level sets how hard it searches:

- **Level 1** remembers one earlier position for each short string, and takes that match if there is one.
- **The top levels** keep many candidates and compare whole sequences of possible matches to find the cheapest encoding.

The same ``X`` is used at every level, and each level is compared with itself: the saving is measured against ``C(Y)`` at that same level.

One detail keeps the comparison fair. zstd also tunes its search to the size of its input, dictionary included, and uses different settings below 256 KB. ``Y`` is over 256 KB so that, with or without ``X``, a given level means the same search.
"""

# ╔═╡ 287e5d9b-29da-5fb4-b6ed-201cb93544f1
LEVELS = 1:19

# ╔═╡ 4e5674f1-b99f-54a8-a0e6-0b017e330f3a
by_level = map(RUNGS) do rung
    X = rung.sample(X_BYTES)
    (; rung.name, pcts = [saved_pct(Y, X; level) for level in LEVELS])
end

# ╔═╡ adffb301-c5e9-48f1-84e3-a2645c4e1ad1
md"""
## From zstd to neural networks

Now put a language model in zstd's place. Because prediction is compression:

- a model that predicts the next token, driving an arithmetic coder, is a compressor;
- its cross-entropy loss, counted in bits, is the size of that file, so the number training drives down *is* a compressed size;
- pretrain it on ``X``, and its loss on ``Y`` is ``C(Y \mid X)``: this notebook's experiment, with a far better ``C``.

Such models compress well even far from their training data. In [*Language Modeling Is Compression*](https://arxiv.org/abs/2309.10668) (Delétang et al., 2023), Chinchilla 70B, trained mostly on text, beats compressors built for images and audio:

| data | Chinchilla 70B | specialised compressor |
|:--|--:|--:|
| ImageNet image patches | 43.4 % | PNG: 58.5 % |
| LibriSpeech audio | 16.4 % | FLAC: 30.3 % |

Each figure is the compressed size as a share of the raw data.
"""

# ╔═╡ 5b002498-e832-4b38-910c-7ab02794cbe4
md"""
### Sutskever's argument

1. The ideal compressor ``K`` would use every regularity that ``X`` and ``Y`` share.
2. A large network trained by gradient descent is a practical, computable approximation of it: a search, over the programs its weights can express, for one that compresses ``X`` well.
3. The better it compresses ``X``, the more of ``X``'s structure it has captured.
4. Whatever part of that structure ``Y`` shares comes to ``Y`` almost for free. That is why unsupervised pretraining helps with tasks it never trained on.

The two knobs in this notebook have direct counterparts:

| here | for neural networks |
|:--|:--|
| the size of ``X`` | the amount of pretraining data |
| zstd's level | the quality of the compressor, which depends on the network's architecture, its size, and how much it is trained |
"""

# ╔═╡ e418da90-64d3-480e-a17a-fa0cd2a93c25
md"""
### Evidence, and a bet

The talk also points to evidence beyond text. In [image GPT](https://proceedings.mlr.press/v119/chen20s.html) (Chen et al., 2020), a transformer trained only to predict the next pixel of images learned features that a linear classifier could use to recognise objects. The better the model predicted pixels, the better those features were.

This notebook shows the mechanism, shared structure turning into saved bits, with a compressor anyone can run. How far the argument goes is another matter. That neural networks find far deeper shared structure than zstd's repeated strings is the talk's bet, and the evidence for it is how well pretraining works in practice.
"""

# ╔═╡ d6d4b873-4f0f-5d45-95bc-5c06ca7b5897
md"""
## Appendix: helpers

Number formatting and the chart style. The colours run from light to dark blue as ``X`` gets closer to ``Y``.
"""

# ╔═╡ d4fecb2e-acb9-52f4-8689-42de95754bad
commas(n::Integer) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => ",")

# ╔═╡ 33e9cb5e-0cc5-5112-9372-c1665dbfc779
let ending = String(copy(Y)),                    # a copy: String empties the vector it is given
    paragraphs = split(ending, "\n\n")[2:end],   # the first is cut off mid-paragraph
    opening = first(filter(p -> length(p) > 300, paragraphs))
    Markdown.parse("""
    ``Y`` is the last $(commas(Y_BYTES)) bytes of the novel. It opens:

    > $(first(opening, 500)) …
    """)
end

# ╔═╡ 2b32e1ee-45f9-5ea8-88eb-df6515f0ad16
percent(pct) = @sprintf("%.1f %%", pct)

# ╔═╡ ac98ce19-4fb9-5184-ae4b-75c643516d77
let (; joint, conditional) = joint_and_conditional
    Markdown.parse("The two differ by $(commas(abs(joint - conditional))) bits, $(percent(100 * abs(joint - conditional) / conditional)) of ``C(Y \\mid X)``. The same quantity, measured two ways; from here on, every saving is measured with the dictionary.")
end

# ╔═╡ 7c974838-9907-5f5f-ba99-e97c45447ccf
let change = -ladder.noise.bits
    Markdown.parse("""
    - **Random noise** changes the cost of ``Y`` by $(change > 0 ? "+" : "")$(commas(change)) bits, $(percent(-ladder.noise.pct)). The compressor finds nothing to use, and loses almost nothing for having looked (**prediction 1**).
    - **Everything else helps**, in the order of what it shares with ``Y`` (**prediction 2**): novels by other authors save $(percent(ladder.other_authors.pct)), Austen's other novels $(percent(ladder.austen.pct)), the book's own earlier chapters $(percent(ladder.own_book.pct)).

    What zstd picks up is concrete: common words and spellings, Austen's phrasing, the names Elizabeth, Darcy and Bingley. The principle is the one in the talk, though: structure in ``X`` that is also in ``Y`` turns into bits saved on ``Y``.
    """)
end

# ╔═╡ be37c1b0-71c3-5f5a-bc27-7a7f5474f03e
Markdown.parse("""
The texts are the same at every level; only the compressor's effort changes (**prediction 3**):

- other authors' novels save $(percent(first(by_level.other_authors.pcts))) of ``Y`` at level 1, and $(percent(last(by_level.other_authors.pcts))) at level 19;
- the book's earlier chapters go from $(percent(first(by_level.own_book.pcts))) to $(percent(last(by_level.own_book.pcts)));
- noise stays within $(percent(maximum(abs, by_level.noise.pcts))) of zero at every level.

In the talk's terms, a better compressor has lower *regret*: less of what ``X`` could have taught it about ``Y`` goes unused. zstd's top level is still far from ``K``, which by definition leaves nothing computable unused. The talk's claim is that large neural networks get much closer.
""")

# ╔═╡ 63364e6a-ff60-5930-8690-e0de3b5f6abe
per_byte(bits) = @sprintf("%.2f", bits / Y_BYTES)

# ╔═╡ 79f9af21-fa21-46cf-a73d-fc7230bcf74d
md"""
## What it does not show

zstd only finds repeated strings. It cannot see what a reader sees:

- that two texts share a grammar,
- that a character is acting in character,
- that a story is being retold in other words.

That is why the savings here are modest, $(percent(ladder.own_book.pct)) even from the book's own earlier chapters. And it is why zstd spends $(per_byte(extremes.alone)) bits on each byte of ``Y`` where Shannon's readers needed about one per letter: most of the structure of English is invisible to it.
"""

# ╔═╡ bacdf724-9c03-58f6-886c-5e84b98a4716
human(bytes) = bytes >= 1_000KB ? @sprintf("%g MB", round(bytes / 1_000KB; sigdigits = 3)) :
                                  @sprintf("%g KB", round(bytes / KB; sigdigits = 3))

# ╔═╡ cb48734c-71a1-5db1-8d26-683ce53bb608
md"""
## The texts

``Y`` is the ending of Jane Austen's *Pride and Prejudice* (1813): its last $(human(Y_BYTES)). Everything comes from [Project Gutenberg](https://www.gutenberg.org), whose files wrap each book in a licence header and footer; those are cut off.

One more step matters for a byte-level compressor. Different volunteers transcribed these books with different habits:

- *Pride and Prejudice* writes dashes as `--`, *Emma* as `—`;
- some books mark italics with `_underscores_`;
- every file breaks its lines after 70-odd characters.

The compressor would notice those habits, and we want it to notice Austen. So quotes, dashes and line breaks are put into one form, and underscores dropped.
"""

# ╔═╡ 8d6b20f4-95dd-5089-ab20-930e7195ee02
Markdown.parse("""
| | bits | bits per byte of ``Y`` |
|:--|--:|--:|
| ``Y`` alone | $(commas(extremes.alone)) | $(per_byte(extremes.alone)) |
| ``Y`` given itself | $(commas(extremes.given_itself)) | $(per_byte(extremes.given_itself)) |
| ``Y`` given $(human(X_BYTES)) of noise | $(commas(extremes.given_noise)) | $(per_byte(extremes.given_noise)) |

On its own, ``Y`` takes zstd about $(round(extremes.alone / Y_BYTES; digits = 1)) bits per byte, against the 8 bits each byte occupies on disk.

Every ``X`` that follows lands between the last two rows: near noise if it shares nothing with ``Y``, near ``Y`` itself if it shares everything.
""")

# ╔═╡ d4d906b9-08e4-5ccb-a282-3bf2a4ad476b
md"""
## More X helps

The ladder gave every ``X`` the same size. Now let the size vary, from 1 KB to 2 MB, wherever there is enough text. The book's earlier chapters stop at about $(human(length(earlier_chapters))): that is all of them.
"""

# ╔═╡ 0b57676b-9b1e-5093-b9b0-678120392281
let more_or_less = last(growth.austen.pcts) > last(growth.own_book.pcts) ? "more than" : "nearly as much as"
    Markdown.parse("""
    Every related source saves more the more of it there is, and none has stopped improving at the largest size.

    Quantity can also make up for relatedness. $(human(last(growth.austen.sizes))) of Austen's other books save $(percent(last(growth.austen.pcts))), $more_or_less the $(percent(last(growth.own_book.pcts))) saved by $(human(last(growth.own_book.sizes))) of the novel's own earlier chapters.

    That is, in miniature, the bet behind pretraining on broad data: a lot of loosely related text can make up for having little of the closely related kind.
    """)
end

# ╔═╡ 665cbe8f-bb81-5774-ba45-8a6c43c3e692
Markdown.parse("""
## What this shows

With nothing cleverer than zstd, all three predictions hold.

1. **Unrelated data costs nothing.** $(human(X_BYTES)) of random noise changes the cost of ``Y`` by $(percent(abs(ladder.noise.pct))).
2. **Related data helps, and more of it helps more.** Novels by other authors save $(percent(ladder.other_authors.pct)) of ``Y``, Austen's other books $(percent(ladder.austen.pct)), the book's own earlier chapters $(percent(ladder.own_book.pct)). And every source saves more as it grows: $(human(last(growth.austen.sizes))) of Austen's other books save $(percent(last(growth.austen.pcts))).
3. **A better compressor finds more.** The same $(human(X_BYTES)) of other authors' novels saves $(percent(first(by_level.other_authors.pcts))) at zstd's fastest level and $(percent(last(by_level.other_authors.pcts))) at its most thorough.

That is the talk's argument in miniature. A compressor that has read ``X`` knows something about ``Y``. How much grows with:

- what the two share,
- how much ``X`` there is,
- how good the compressor is.
""")

# ╔═╡ 53fc3dfe-847e-50a9-b82f-e1117719d5dc
VIZ = (surface = "#fcfcfb", text = "#0b0b0b", text2 = "#52514e", grid = "#e4e3de")

# ╔═╡ 50ea2d49-f0e3-55a8-b74c-98c2a1daee32
RUNG_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

# ╔═╡ 3a9df0c9-4b50-50fc-8d1b-f1421ba55ab7
RUNG_MARKERS = [:circle, :rect, :diamond, :utriangle]

# ╔═╡ d8179fc3-847c-5194-9b84-ea2d3c4646bd
THEME = Theme(
    fontsize = 13, backgroundcolor = VIZ.surface, textcolor = VIZ.text,
    Axis = (backgroundcolor = VIZ.surface, xgridcolor = VIZ.grid, ygridcolor = VIZ.grid,
            topspinevisible = false, rightspinevisible = false, leftspinecolor = VIZ.text2, bottomspinecolor = VIZ.text2,
            xtickcolor = VIZ.text2, ytickcolor = VIZ.text2, xticklabelcolor = VIZ.text2, yticklabelcolor = VIZ.text2,
            titlealign = :left, titlefont = :regular, titlesize = 14),
    Legend = (framevisible = false, backgroundcolor = VIZ.surface),
)

# ╔═╡ bc8f168f-38d2-500b-a7d6-f317545e94ba
with_theme(THEME) do
    fig = Figure(size = (760, 300))
    rows = 1:length(ladder)
    pcts = [r.pct for r in ladder]
    ax = Axis(fig[1, 1];
              title = "What $(human(X_BYTES)) of X saves on Y",
              xlabel = "bits saved on Y, % of C(Y)",
              yticks = (rows, [r.name for r in ladder]),
              yreversed = true,
              ygridvisible = false)
    barplot!(ax, rows, pcts; direction = :x, width = 0.6, color = RUNG_COLORS)
    text!(ax, max.(pcts, 0), rows;
          text = percent.(pcts), align = (:left, :center), offset = (6, 0), color = VIZ.text, fontsize = 12)
    vlines!(ax, 0; color = VIZ.text2, linewidth = 1)
    xlims!(ax, -1, 1.18 * maximum(pcts))
    fig
end

# ╔═╡ 30322a8e-584a-5ad0-9d57-ce045cffb6a4
with_theme(THEME) do
    fig = Figure(size = (760, 460))
    ax = Axis(fig[1, 1];
              title = "The saving grows with the amount of related text",
              xlabel = "size of X",
              ylabel = "bits saved on Y, % of C(Y)",
              xscale = log10,
              xticks = ([1e3, 1e4, 1e5, 1e6], ["1 KB", "10 KB", "100 KB", "1 MB"]))
    hlines!(ax, 0; color = VIZ.text2, linewidth = 1)
    for (g, color, marker) in zip(growth, RUNG_COLORS, RUNG_MARKERS)
        scatterlines!(ax, g.sizes, g.pcts; color, marker, markersize = 9, linewidth = 2, label = g.name)
    end
    Legend(fig[2, 1], ax; orientation = :horizontal, nbanks = 2)
    fig
end

# ╔═╡ 95429bb2-8f02-5b81-885d-c3d660d5b71d
with_theme(THEME) do
    fig = Figure(size = (760, 460))
    ax = Axis(fig[1, 1];
              title = "A better compressor gets more out of the same $(human(X_BYTES))",
              xlabel = "zstd level",
              ylabel = "bits saved on Y, % of C(Y)",
              xticks = 1:2:19)
    hlines!(ax, 0; color = VIZ.text2, linewidth = 1)
    for (l, color, marker) in zip(by_level, RUNG_COLORS, RUNG_MARKERS)
        scatterlines!(ax, LEVELS, l.pcts; color, marker, markersize = 9, linewidth = 2, label = l.name)
    end
    Legend(fig[2, 1], ax; orientation = :horizontal, nbanks = 2)
    fig
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
CodecZstd = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
Downloads = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
CairoMakie = "~0.15.16"
CodecZstd = "~0.8.7"
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.2"
manifest_format = "2.1"
project_hash = "2a5452a2863affc5c849667d888fba72cacb85ff"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
registries = "General"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "e71ee7b4aa06b045259a7d6101e1cb45ad140bce"
registries = "General"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.1"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
registries = "General"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "7063ad1083578215c7c4bf410368150abe8d5524"
registries = "General"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.45"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "dee8388a9bec89c8713ad176f96960ae2f98598a"
registries = "General"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.7.4"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptiveArrayPools]]
deps = ["Preferences", "Printf", "Random"]
git-tree-sha1 = "5feaa5055572e809b69b62f06826b1c1c8d3ebf3"
registries = "General"
uuid = "4f381ef7-9af0-4cbe-99d4-cf36d7b0f233"
version = "0.3.11"

    [deps.AdaptiveArrayPools.extensions]
    AdaptiveArrayPoolsCUDAExt = "CUDA"
    AdaptiveArrayPoolsMetalExt = "Metal"

    [deps.AdaptiveArrayPools.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
registries = "General"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
registries = "General"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
registries = "General"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "94eab0b3ccdcac361188cc661daf69d4433c1818"
registries = "General"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.2.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
registries = "General"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
registries = "General"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "8c290a1b223deaeea9aea44b235d24546da8eb98"
registries = "General"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.4.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
registries = "General"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
registries = "General"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "66188d9d103b92b6cd705214242e27f5737a1e5e"
registries = "General"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.2"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
registries = "General"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
registries = "General"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Cairo_jll", "Colors", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "90fc72982362b84546b64b44fa13e39a4ccb461e"
registries = "General"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.15.16"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "0cfb779e24fdced77a05df1dbe99200821a9f563"
registries = "General"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.8+1"

[[deps.ChainRulesCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2575de43dcab7c966651d86d7a42a75f477c53fa"
registries = "General"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.3"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZstd]]
deps = ["TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "da54a6cd93c54950c15adf1d336cfd7d71f51a56"
registries = "General"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.8.7"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "07da79661b919001e6863b81fc572497daa58349"
registries = "General"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
registries = "General"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "a7cf827a79b378a381e31e5c7feca853ec19cb94"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.4"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
registries = "General"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "a633156e74096f90b3b5e833529292b651b5671a"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.3"

[[deps.CommonSolve]]
deps = ["PrecompileTools"]
git-tree-sha1 = "6c389fa857f6ca5a95474b52a52023fd77f24cb7"
registries = "General"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.14"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
registries = "General"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.5.5+2"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
registries = "General"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ComputePipeline]]
deps = ["Observables", "Preferences"]
git-tree-sha1 = "7bc84b769c1d384315e7b5c4ac03a6c303e6cf35"
registries = "General"
uuid = "95dc2771-c249-4cd0-9c9f-1f3b4330693c"
version = "0.1.8"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
registries = "General"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
registries = "General"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoreMath]]
deps = ["CoreMath_jll"]
git-tree-sha1 = "8c0480f92b1b1796239156a1b9b1bfb1b39499b4"
registries = "General"
uuid = "b7a15901-be09-4a0e-87d2-2e66b0e09b5a"
version = "0.1.0"

[[deps.CoreMath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a692a4c1dc59a4b8bc0b6403876eb3250fde2bc3"
registries = "General"
uuid = "a38c48d9-6df1-5ac9-9223-b6ada3b5572b"
version = "0.1.0+0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
registries = "General"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
registries = "General"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
registries = "General"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "e1e28f37cbdca4b594ad86ecfe2ee8f25796ebff"
registries = "General"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.7.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "Roots", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "2ddbc341251ed713d39e503b504593a5d6789551"
registries = "General"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.132"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
registries = "General"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
registries = "General"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
registries = "General"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "83231673ea4d3d6008ac74dc5079e77ab2209d8f"
registries = "General"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2bfb1e047e2ad0a5ca94365340bde8005d637568"
registries = "General"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.4+0"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d9d3cd382f2c999f684e6bfa4039fcc16c153b6f"
registries = "General"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "9.0.2+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "65e55303b72f4a567a51b174dd2c47496efeb95a"
registries = "General"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.1"

[[deps.FastInterpolations]]
deps = ["AdaptiveArrayPools", "HelpPlots", "LinearAlgebra", "Preferences", "Printf"]
git-tree-sha1 = "f7bcd496f0a2e4e1174e7e1f667f27fb671fd7df"
registries = "General"
uuid = "9ea80cae-fc13-4c00-8066-6eaedb12f34b"
version = "0.4.20"

    [deps.FastInterpolations.extensions]
    FastInterpolationsChainRulesCoreExt = "ChainRulesCore"
    FastInterpolationsColorVectorSpaceExt = ["ColorTypes", "ColorVectorSpace"]
    FastInterpolationsEnzymeExt = "Enzyme"
    FastInterpolationsForwardDiffExt = "ForwardDiff"
    FastInterpolationsRecipesBaseExt = "RecipesBase"
    FastInterpolationsSymbolicsExt = ["Symbolics", "SymbolicUtils"]

    [deps.FastInterpolations.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ColorTypes = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
    ColorVectorSpace = "c3611d14-8923-5661-9e6a-0046d554d3a4"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SymbolicUtils = "d1185830-fcd6-423d-90d6-eec64667417b"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6621fef488e496356c9c9625d0562c12a6070819"
registries = "General"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.20.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport"]
git-tree-sha1 = "a1b2fbfe98503f15b665ed45b3d149e5d8895e4c"
registries = "General"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.9.0"

    [deps.FilePaths.extensions]
    FilePathsGlobExt = "Glob"
    FilePathsURIParserExt = "URIParser"
    FilePathsURIsExt = "URIs"

    [deps.FilePaths.weakdeps]
    Glob = "c27321d9-0574-5035-807b-f59d2c89b15c"
    URIParser = "30578b45-9adc-5946-b283-645ec420af67"
    URIs = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
registries = "General"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "417289ac840d201005d9ed16789a46defccbe36d"
registries = "General"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.18.1"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
registries = "General"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
registries = "General"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
registries = "General"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
registries = "General"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
registries = "General"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FreeTypeAbstraction]]
deps = ["BaseDirs", "ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "Mmap"]
git-tree-sha1 = "4ebb930ef4a43817991ba35db6317a05e59abd11"
registries = "General"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.8"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
registries = "General"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Gamma]]
deps = ["LogExpFunctions"]
git-tree-sha1 = "becc397f7cfb06e343496ae6ffb04818a851da51"
registries = "General"
uuid = "a0844989-3bd2-4988-8bea-c9407ab0941b"
version = "1.2.0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "ec46c5825710fa1a15d468acb2d93cc939a7a5fe"
registries = "General"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.13"

    [deps.GeometryBasics.extensions]
    ExtentsExt = "Extents"
    GeometryBasicsGeoInterfaceExt = "GeoInterface"
    IntervalSetsExt = "IntervalSets"

    [deps.GeometryBasics.weakdeps]
    Extents = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
    GeoInterface = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
registries = "General"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3efbbc027441271444dcd0c0a46f2d119dc4329"
registries = "General"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "6.1.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "ea9f1fb37ff10597353b56f3ba0ebd525acef6e0"
registries = "General"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.90.1+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
registries = "General"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
registries = "General"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "ef70da5e123a06a29e2d6ddff0f09985bc226491"
registries = "General"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.3"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "9d9531a9cb63a9edc33836414e82a07e81710de2"
registries = "General"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.14004.0+0"

[[deps.HelpPlots]]
git-tree-sha1 = "07d389d81ca31e42d68bd88cfa1e900a63b7d378"
registries = "General"
uuid = "5880b2e8-ad0a-48c9-bacd-30dc9c1357e0"
version = "1.1.0"

    [deps.HelpPlots.extensions]
    HelpPlotsExt = "Plots"

    [deps.HelpPlots.weakdeps]
    Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[[deps.HypergeometricFunctions]]
deps = ["Gamma", "LinearAlgebra"]
git-tree-sha1 = "31bb6c92405c084617facc1d7ed9eb6c402d061e"
registries = "General"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.30"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
registries = "General"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
registries = "General"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
registries = "General"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
registries = "General"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
registries = "General"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
registries = "General"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "f0f005f997dfb8c5fe23920d99458a9619873893"
registries = "General"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.10"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
registries = "General"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc8d0cd653e55213df9b75ebc6fe4a8d3254c65"
registries = "General"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.2.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
registries = "General"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
registries = "General"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "c72458f1962faeb003bf23cbdb75164fe6280906"
registries = "General"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "48922d06068130f87e43edef52382e6a94305ae6"
registries = "General"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.3"

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "CoreMath", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Printf", "Random", "RoundingEmulator"]
git-tree-sha1 = "1c531bf0f8a5c60a340926e058fd3f209b5eef5d"
registries = "General"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.12"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticIrrationalConstantsExt = "IrrationalConstants"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticMakieExt = "Makie"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

    [deps.IntervalArithmetic.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    IrrationalConstants = "92d709cd-6900-40b7-9082-c6be49f344b6"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.IntervalSets]]
git-tree-sha1 = "0db6aea5b64caa1c47e9fdd27394f0296f05e8bf"
registries = "General"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.15"

    [deps.IntervalSets.extensions]
    IntervalSetsMakieExt = "Makie"
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

    [deps.IntervalSets.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.InverseFunctions]]
git-tree-sha1 = "5cc6ad3e2bfc75a11eed92632ee341d37f4a4536"
registries = "General"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.18"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
registries = "General"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
registries = "General"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
registries = "General"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
registries = "General"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
registries = "General"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "9939eb6f799a963b3392ac4a1d49d3fc0646eb80"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.11.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
registries = "General"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "037babc10853eeb8e585418922246cb97b8e5b74"
registries = "General"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+1"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "9eda8292dd3268b3b7ec9df21bbfac24e177ec52"
registries = "General"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.12"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
registries = "General"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "39bca05343661c347aae0bca57a5994a0bf4f08d"
registries = "General"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.2.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e5b100780d4d30d63b4618d7930d48af409c1772"
registries = "General"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "23.1.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
registries = "General"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
registries = "General"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "1.0.0"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "Zstd_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.22.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.7+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "OpenSSL_jll", "Zlib_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.104+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
registries = "General"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
registries = "General"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
registries = "General"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b8a91037379e1fd661e9b5ff3d7a770424b1f9f2"
registries = "General"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.4+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
registries = "General"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9e887184f306b28dbfcff88cefbb0ec80f1aa9e0"
registries = "General"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.4+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.13.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "28d620bcdb37a3fb02de9f521928ee453690f414"
registries = "General"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.3"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
registries = "General"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "ComputePipeline", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FastInterpolations", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "Pkg", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "5fbafd8aa82e808712f7dee8b52ef87118243fd6"
registries = "General"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.24.16"

    [deps.Makie.extensions]
    MakieDynamicQuantitiesExt = "DynamicQuantities"

    [deps.Makie.weakdeps]
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
registries = "General"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "aa1078778be5a8e5259ff04fbc3d258b3e78d464"
registries = "General"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.9"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
registries = "General"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
registries = "General"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2026.8.13"

[[deps.MuladdMacro]]
deps = ["PrecompileTools"]
git-tree-sha1 = "283bf85d4a767481dd924dff0eee1735e95f449e"
registries = "General"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.7"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
registries = "General"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
registries = "General"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
registries = "General"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
registries = "General"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
registries = "General"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "38a93f17e431141c6470bb67a88952a7c4f0e928"
registries = "General"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.34+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.30+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
registries = "General"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "OpenJPH_jll", "Zlib_jll", "Zstd_jll", "libdeflate_jll"]
git-tree-sha1 = "eb47260eaf61dca5c0e17bd126f4426edac35975"
registries = "General"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.5.2+0"

[[deps.OpenJPH_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll"]
git-tree-sha1 = "1b1f3cabb1130670812b848f4623d69d578ea805"
registries = "General"
uuid = "95196869-6d61-5868-8eee-c38c33804e0c"
version = "0.32.0+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.8+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
registries = "General"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
registries = "General"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "b079a6affebde167dad8ffe0d670b53917fbcb68"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.46.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "123266c25174ef6c8d4718920abc206452cf8de6"
registries = "General"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.41"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "32b657a0d57c310a1a172bfc8c8cf68c5e674323"
registries = "General"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.5"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
registries = "General"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
registries = "General"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1912a9f1b9ca55005b03ba075f8e19993583e237"
registries = "General"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.2+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "663e8b48b789916221e0765393b289ca6c88f24e"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
registries = "General"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "Zstd_jll", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.13.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
registries = "General"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Reexport", "Statistics"]
git-tree-sha1 = "ddcc36fd83fb4cd459a53c891981697bde0d3de6"
registries = "General"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.5.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
registries = "General"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
registries = "General"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
registries = "General"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "aef1987d90dcc2e2cbd919ad10b2d25bcce4f418"
registries = "General"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.1"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
registries = "General"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Printf"]
git-tree-sha1 = "d72534986f4a4641b4d2492a154ec0b0e6985284"
registries = "General"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.12.0"
weakdeps = ["Distributed"]

    [deps.ProgressMeter.extensions]
    ProgressMeterDistributedExt = "Distributed"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
registries = "General"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "472daaa816895cb7aee81658d4e7aec901fa1106"
registries = "General"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
registries = "General"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["Base64", "Dates", "FileWatching", "InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
registries = "General"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
registries = "General"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
registries = "General"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
registries = "General"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
registries = "General"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
registries = "General"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d40b2fe70437b01397d2a4d5b020008da4e7019"
registries = "General"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.2+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "e6ffa7bd4f8d0b8744d18f219304738d952f9a37"
registries = "General"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "3.0.11"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
registries = "General"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "1.0.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "37ac518b3c924f110d82d50173eaf4ebeaf27ad7"
registries = "General"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.8.1"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
registries = "General"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "57aa595158717ef165e6f5ab639fe2e3178c0a2b"
registries = "General"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.SignedDistanceFields]]
deps = ["Statistics"]
git-tree-sha1 = "3949ad92e1c9d2ff0cd4a1317d5ecbba682f4b92"
registries = "General"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
registries = "General"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2c79185e5261b159474903598c571ddd9f12ff7b"
registries = "General"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.6"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
registries = "General"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.13.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "429071b23f4c9a13fb6582f807cc2ef454082408"
registries = "General"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
registries = "General"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "de349d5585559b692f518b6d3fc7c179d64b2364"
registries = "General"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.23"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
registries = "General"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
registries = "General"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
registries = "General"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "adb9da019510162e67a4493fc235c23203d8b09e"
registries = "General"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.13"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91a5737baed20ee31f3faea0e51f57461f6a689e"
registries = "General"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "2.2.1"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "4daaddbe37aa80b4551bf7c56adf53e7cfcdf35a"
registries = "General"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.4"

    [deps.StructArrays.extensions]
    AdaptExt = "Adapt"
    GPUArraysCoreExt = "GPUArraysCore"
    KernelAbstractionsExt = "KernelAbstractions"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"
    StaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ad435e30854ed0f50f4b7f67edd71664afecc5d8"
registries = "General"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.10.0"

    [deps.StructUtils.extensions]
    StructUtilsDataFramesExt = ["DataFrames"]
    StructUtilsLazilyInitializedFieldsExt = ["LazilyInitializedFields"]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazilyInitializedFields = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.10.1+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
registries = "General"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a94d9bdda1b7bed0046cea645639ab3f62196fac"
registries = "General"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.14.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
registries = "General"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TiffImages]]
deps = ["CodecZstd", "ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "9ca5f1f2d42f80df4b8c9f6ab5a64f438bbd9976"
registries = "General"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.9"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
registries = "General"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
registries = "General"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
registries = "General"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
deps = ["Serialization"]
git-tree-sha1 = "50eda8011f098869a76da69a524ff59e0298711a"
registries = "General"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
registries = "General"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "1f0f9f401753701a7e4113b5056ca38d33875b55"
registries = "General"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.29.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    LatexifyExt = ["Latexify", "LaTeXStrings"]
    NaNMathExt = "NaNMath"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
registries = "General"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
registries = "General"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e52eca002a11c30a858185efdfb15311e1c7a6bf"
registries = "General"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "ce91b54a5f90c4f27df6950706234b34e49cf79e"
registries = "General"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.14+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
registries = "General"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
registries = "General"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
registries = "General"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
registries = "General"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a54cf54677c9f6eac0d97e9d757c233b56bc21b9"
registries = "General"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.13+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
registries = "General"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
registries = "General"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
registries = "General"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.2+0"

[[deps.Zstd_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl"]
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
registries = "General"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1210ba774d3427387d307bf1f416d699b7c39417"
registries = "General"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.15.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cb007192783c56d8249db4cf0e3495001edfe414"
registries = "General"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdeflate_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7ee221abc4f125ce18e73d48367cde0e16bc93b9"
registries = "General"
uuid = "46979653-d7f6-5232-b59e-dd310c4598de"
version = "1.26.0+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
registries = "General"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
registries = "General"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "1fcd6cf05d23adab0b5ffc496d020537215751dc"
registries = "General"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.59+1"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "e067c8bae65bb40866552296a2d5b4dc65b8928f"
registries = "General"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.80.702+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
registries = "General"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
registries = "General"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "52d3b9475133c3bc8c0a7f90f18bfc8cdb5a443a"
registries = "General"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.70.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.8.3+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
registries = "General"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
registries = "General"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╟─fdfcd5cc-ed29-5406-83a2-023cf5021853
# ╟─c331071a-b435-4482-83f4-9f06dd3f0af3
# ╟─2c83d185-422a-5af7-9fce-7c99ce734229
# ╟─23bd693b-a7da-573d-8f5d-ac8b61626bd8
# ╟─c65512c4-bab7-44b1-a30f-4f2223cd9eca
# ╟─0c3334f8-3464-4c24-a358-a10c162d80e1
# ╟─dcfb1941-2f44-4294-9394-0c2baccd5fb4
# ╟─1d1f89d8-4de4-526c-a60f-862c213a77b2
# ╠═4179174c-6ee4-520b-9c58-f6262778cb94
# ╟─833a76f1-a7ef-45b3-9d11-6b92f1ea3120
# ╠═1d77adbb-cc37-53f6-b0f3-635453e64314
# ╟─ee5e727f-9b96-5369-bb82-a2685b158eec
# ╠═188edca4-84ae-5275-8020-e2169ac6db41
# ╠═41817472-c4d9-5438-9b1d-6de4361262fb
# ╠═5b8da8fe-341f-58fb-b542-39295a8dafd8
# ╠═8f96d614-b186-5f22-90f3-eee491e96641
# ╟─cb48734c-71a1-5db1-8d26-683ce53bb608
# ╠═2efc182d-bac3-58f9-bb99-74a9b806b47e
# ╠═51a4d8e4-41f0-598a-8551-3be4f8766f07
# ╠═e9fa7345-6560-593c-ab56-3d3dfc4b40ff
# ╠═a240f10f-2e89-5385-90de-b40f823426be
# ╠═a9d82f5b-d5ec-5333-9450-7c82e9c20de9
# ╠═3b5e9341-0a8e-5a31-804a-b5745e2fed23
# ╠═0f5b588a-3f8e-54c8-b8ec-237f8cd2d5ab
# ╠═cdd84fd7-846e-5ddf-9fc5-e70d6c52f4cc
# ╠═501b4ad0-8a82-59fa-8019-3ab54aaa35f7
# ╠═66503f0e-d0f1-5908-bfea-e744f83c1824
# ╟─33e9cb5e-0cc5-5112-9372-c1665dbfc779
# ╟─73a742ce-89b4-5dc4-95f3-6c5c7b57559f
# ╠═0658cb43-1ee8-54f3-9576-0f2c4307708e
# ╠═60919141-e13c-5016-a555-a6c0363da036
# ╟─8d6b20f4-95dd-5089-ab20-930e7195ee02
# ╟─fa29009c-3120-5d82-a1c5-6a76e9bf80de
# ╠═7567ec87-fa71-5049-a2a6-23ab6b195df8
# ╠═40abbd27-e0e5-5293-984d-0585b00c6a27
# ╟─ac98ce19-4fb9-5184-ae4b-75c643516d77
# ╟─2674ef0f-4e18-50ab-8ed8-ed4aaf966df8
# ╠═d2fbbfc0-ff05-5358-b99e-ae661108602e
# ╠═d936643b-44b6-5012-8246-bbfca7b9c11d
# ╠═6c2a096d-1af9-5d41-ae83-dcdac11042e1
# ╠═20dc88ba-af86-5f9a-af0d-2993416832f4
# ╠═a5ccc482-782d-5ac8-91b8-5cc7e6b2446a
# ╠═a3ac1236-770e-57b8-81c9-71c23f2e9a84
# ╟─bc8f168f-38d2-500b-a7d6-f317545e94ba
# ╟─7c974838-9907-5f5f-ba99-e97c45447ccf
# ╟─d4d906b9-08e4-5ccb-a282-3bf2a4ad476b
# ╠═0bfb6f43-c544-57ad-9838-d86502a2d115
# ╠═f7a08251-3b7b-52d7-b0dc-7ba92760df3b
# ╟─30322a8e-584a-5ad0-9d57-ce045cffb6a4
# ╟─0b57676b-9b1e-5093-b9b0-678120392281
# ╟─ff1487ad-75fb-5cf0-be5b-94bbaff556fd
# ╠═287e5d9b-29da-5fb4-b6ed-201cb93544f1
# ╠═4e5674f1-b99f-54a8-a0e6-0b017e330f3a
# ╟─95429bb2-8f02-5b81-885d-c3d660d5b71d
# ╟─be37c1b0-71c3-5f5a-bc27-7a7f5474f03e
# ╟─665cbe8f-bb81-5774-ba45-8a6c43c3e692
# ╟─79f9af21-fa21-46cf-a73d-fc7230bcf74d
# ╟─adffb301-c5e9-48f1-84e3-a2645c4e1ad1
# ╟─5b002498-e832-4b38-910c-7ab02794cbe4
# ╟─e418da90-64d3-480e-a17a-fa0cd2a93c25
# ╟─d6d4b873-4f0f-5d45-95bc-5c06ca7b5897
# ╟─d4fecb2e-acb9-52f4-8689-42de95754bad
# ╟─2b32e1ee-45f9-5ea8-88eb-df6515f0ad16
# ╟─63364e6a-ff60-5930-8690-e0de3b5f6abe
# ╟─bacdf724-9c03-58f6-886c-5e84b98a4716
# ╟─53fc3dfe-847e-50a9-b82f-e1117719d5dc
# ╟─50ea2d49-f0e3-55a8-b74c-98c2a1daee32
# ╟─3a9df0c9-4b50-50fc-8d1b-f1421ba55ab7
# ╟─d8179fc3-847c-5194-9b84-ea2d3c4646bd
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
