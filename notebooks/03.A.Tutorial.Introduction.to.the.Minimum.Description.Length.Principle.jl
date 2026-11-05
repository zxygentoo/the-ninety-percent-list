### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ 75387367-b23d-469a-be55-5ff5a256494b
begin
    using CairoMakie, PlutoUI
    using Printf, Random
end

# ╔═╡ 9e9ed333-b8c3-4513-97e7-f610b20fa6d0
md"""
# A Tutorial Introduction to the Minimum Description Length Principle

*We can't find the best description. What if we settle for a smaller machine?*

**Prerequisites**

- [Notebook 02](https://zxygentoo.github.io/the-ninety-percent-list/02.Kolmogorov.Complexity.and.Algorithmic.Randomness.html), or the idea that ``K(x)`` is the length of the shortest program that prints ``x``.
- Basic probability theory.
- Information measured in bits, ``-\log_2 P(x)``.

---

[Notebook 02](https://zxygentoo.github.io/the-ninety-percent-list/02.Kolmogorov.Complexity.and.Algorithmic.Randomness.html) ended on a deal. The best compressor, ``K``, can't be computed, so practical compressors give up the guarantee of the shortest description to get *a* description, in time.

There is more than one way to strike that deal. ``Kt`` keeps every program and bounds the time. zstd fixes one scheme and makes no promise about it. A neural network searches a rich family by gradient descent. The *Minimum Description Length* principle, MDL, strikes it this way: shrink the language until the best description in it can be computed exactly.

Peter Grünwald's [tutorial](https://arxiv.org/abs/math/0406077) is our guide, and the section numbers below refer to it.
"""

# ╔═╡ b37ccf49-8d6b-4838-ba75-872c194c3df2
TableOfContents()

# ╔═╡ 6852bb93-1cbc-43dc-94bd-c44cc60c92be
md"""
## A small machine with knobs

``K`` describes strings in a language that can say anything computable, and that is what makes it uncomputable. To know a description is the shortest, we have to rule out every shorter one, including programs that never stop.

MDL swaps the universal machine for a small one, with a few knobs. The knobs are its *parameters*, usually written ``\theta``, and each setting makes the machine expect different strings. A description is then two things: a setting of the knobs, and a codeword that the machine turns back into the string.

A small machine like this is a *model*: a family of probability distributions, one for each setting.

Notebook 02 had no such split. A program was the whole description: it held the pattern and the noise together, and the machine read nothing else. MDL separates them. The setting holds the pattern, and the codeword holds whatever the setting can't predict.

The swap has a price. Even in the luckiest case, MDL finds the best description its model can express, never a better one.

What the swap buys is certainty. Within the model, the best description is computed directly: no search, and nothing left running. And the choice of language, which ``K`` hides in a machine-dependent constant, is now out in the open: it is the model we picked.

How does a distribution describe a string?
"""

# ╔═╡ bc23a13c-e665-4d59-90f0-a68b248f88b5
md"""
## A distribution is a code

To send a string, a sender and a receiver agree on a *code* in advance: a codeword for every string. Here are two codes for the four strings of 2 bits:

| string | fair code | lopsided code |
|:-:|:-:|:-:|
| `00` | `00` | `0` |
| `01` | `01` | `10` |
| `10` | `10` | `110` |
| `11` | `11` | `111` |

In both, no codeword is the start of another, so the receiver always knows where a codeword ends. The lopsided code gives `00` a single bit, and pays for it elsewhere: `10` and `11` now cost three.

That is no accident. A codeword of ``L`` bits claims every message that starts with it, a share ``2^{-L}`` of all possible messages. The single bit `0` claims half of them, `10` a quarter, and `110` and `111` an eighth each:

| | `00` | `01` | `10` | `11` | total |
|:--|:-:|:-:|:-:|:-:|:-:|
| fair code | ¼ | ¼ | ¼ | ¼ | 1 |
| lopsided code | ½ | ¼ | ⅛ | ⅛ | 1 |

The shares can't add up to more than everything. Shorten one codeword, and others must grow.

Those shares behave exactly like probabilities, and that is the bridge from codes to statistics:

!!! info "Codes are distributions (Kraft's inequality, §2.2.2)"
    In a code where no codeword starts another, the codelengths ``L(x)`` satisfy

    ```math
    \sum_x 2^{-L(x)} \le 1
    ```

    and any whole-number lengths that satisfy it belong to some such code. So the lengths of a code give probabilities, ``P(x) = 2^{-L(x)}``, that add up to at most 1, and a distribution gives a code with

    ```math
    L(x) = -\log_2 P(x)
    ```

    - ``L(x)`` is the length of the codeword for ``x``, in bits;
    - ``2^{-L(x)}`` is the share of all messages that codeword claims;
    - ``\le 1``: together, the shares can't claim more than everything.

So probability is a budget of 1, shared among all the strings. A likely string gets a big share and a short codeword; an unlikely one gets a small share and a long codeword.

### Arithmetic coding

[Arithmetic coding](https://en.wikipedia.org/wiki/Arithmetic_coding) turns a string into a codeword, one bit at a time, and needs only one thing from the model: given the bits so far, how likely the next bit is to be a 1. We call that the model's *predictor*.

Start with the interval ``[0, 1)``. For each bit, split the interval in the predictor's proportions, 0 on the left and 1 on the right, and keep the side of the actual bit. A likely bit keeps most of the interval; a surprise keeps a sliver. At the end, the interval's width is the string's probability, and the codeword names a number inside it, ``c``, in about ``-\log_2 P(x)`` bits.

The receiver knows the predictor and the length, and replays the same splits. It doesn't know the bits, but it knows ``c``, so at each split it keeps whichever side ``c`` falls on.

Here is the coder. Its intervals are exact fractions, so nothing is lost to rounding.
"""

# ╔═╡ ac3ef555-c4c5-44b8-bcfa-c207cd665802
# The interval [0, 1), as (low, high), in exact fractions.
WHOLE = (big(0) // 1, big(1) // 1)

# ╔═╡ 9c47fc2f-170d-41a9-8d77-27d2afb75b94
"The predictor's split of the interval: its 0 side, then its 1 side."
function sides((low, high), p_one)
    cut = low + (high - low) * (1 - p_one)
    (low, cut), (cut, high)
end

# ╔═╡ 8e7ed6d1-c282-4d53-b01e-5e3cc2807807
"The interval after each bit of `x`: the side of that bit, as `predict` splits it."
intervals(x, predict) = accumulate(eachindex(x); init = WHOLE) do interval, t
    zero_side, one_side = sides(interval, predict(x[1:t-1]))
    x[t] == '0' ? zero_side : one_side
end

# ╔═╡ 516998c3-cbc3-481b-98ea-4618980ee161
"The shortest codeword whose every continuation lands in the interval: `m` bits that name `j / 2^m`."
function codeword((low, high))
    for m in Iterators.countfrom(0)
        j = ceil(BigInt, low * big(2)^m)
        (j + 1) // big(2)^m <= high && return m == 0 ? "" : string(j; base = 2, pad = m)
    end
end

# ╔═╡ 6686b3c5-d261-496d-b4f6-b1e992ab103d
encode(x, predict) = codeword(last(intervals(x, predict)))

# ╔═╡ b0601d04-26a8-4342-b621-a7a9d2cc790e
"The number a codeword names: its bits, after a binary point."
number(code) = isempty(code) ? big(0) // 1 : parse(BigInt, code; base = 2) // big(2)^length(code)

# ╔═╡ 14ef12f7-4060-4068-81ba-0ca8e890759c
"Read `n` bits back from `code`: at each split, keep whichever side the codeword's number falls on."
function decode(code, predict, n)
    c = number(code)
    x, _ = foldl(1:n; init = ("", WHOLE)) do (x, interval), _
        zero_side, one_side = sides(interval, predict(x))
        c < first(one_side) ? (x * '0', zero_side) : (x * '1', one_side)
    end
    x
end

# ╔═╡ b6f86f8c-0017-4d96-b9bf-88dbcc9bf8ba
"The ideal codelength, −log₂ P(x), in bits: the final interval's width is P(x)."
ideal(x, predict) = let (low, high) = last(intervals(x, predict))
    log2(1 / Float64(high - low))
end

# ╔═╡ 78208971-5f13-4f4b-9458-46178e82f425
md"""
From here on we count the ideal length, ``-\log_2 P(x)``, in fractional bits. The coder always comes within 2 bits of it, for a whole string.

A model, then, is a family of codes, one for each setting of its knobs. Each setting is a *member* of the family. For a given string, the best member is the one that gives it the most probability, and so the shortest codeword. We call that codeword's length the string's *best fit*: ``-\log_2 P(x \mid \hat\theta)``, where ``\hat\theta`` is the best setting.

Let's see some example models and how they fit.
"""

# ╔═╡ f1302616-023a-4596-b1b5-bab3769302b0
md"""
## Four machines

All four describe strings of ``n`` bits:

| model | parameters | says |
|:--|:-:|:--|
| coin | 0 | every bit is 1 with probability ½ |
| Bernoulli | 1 | every bit is 1 with probability ``\theta`` |
| Markov | 2 | a bit is 1 with probability ``\theta_0`` after a 0, and ``\theta_1`` after a 1; the first bit is a fair coin |
| Digits | 1 | the string is the first ``n`` binary digits of ``\theta``, a number between 0 and 1 |

The first three are ordinary statistics. Digits is the odd one out. Write ``\theta`` in binary, ``0.x_1 x_2 x_3 \ldots``, and Digits predicts the string ``x_1 x_2 \ldots x_n``, with certainty: each member puts all its probability on a single string.

### The machines

To the coder, each model is a predictor with knobs. The coin has no knobs, Bernoulli one, and Markov two. Digits has one, a number between 0 and 1 that can hold as many digits as we like.
"""

# ╔═╡ 891d29ca-9e98-4ec5-bc78-e8f02d3250a4
predict_coin() = prefix -> 1 // 2

# ╔═╡ 8994982b-795d-4167-9ffe-76741582215d
predict_bernoulli(θ) = prefix -> θ

# ╔═╡ 1b4a5364-5019-4b46-b891-eedc56924475
"The first bit is a fair coin; after that, θ₀ after a 0 and θ₁ after a 1."
predict_markov(θ₀, θ₁) = prefix -> isempty(prefix) ? 1 // 2 : last(prefix) == '0' ? θ₀ : θ₁

# ╔═╡ 99316a9a-9864-49d2-9bf0-ee4ef03776a6
"θ, written in binary digits: certain that the next bit is θ's next digit."
predict_digits(θ) = prefix -> θ[length(prefix) + 1] == '1' ? 1 // 1 : 0 // 1

# ╔═╡ e2a85bf1-ce9b-4043-ba74-6ffb1c2184c6
md"""
### Fitting is turning the knob

Bernoulli's knob has a name: ``\theta`` is the coin's *bias*, the chance it comes up 1. Turn it, and the same string costs a different number of bits:
"""

# ╔═╡ ad357c47-1f0f-495f-8ead-fdbbffc0eebf
"Binary entropy, in bits: what each flip of a coin with bias `p` costs, on average."
entropy(p) = p == 0 || p == 1 ? 0.0 : -p * log2(p) - (1 - p) * log2(1 - p)

# ╔═╡ 79656853-1eaf-48e6-be5d-86bb6cdadfbd
"The best fit, in bits, of `total` coin flips that came up 1 `ones` times: the bias set to `ones / total`."
fit_flips(ones, total) = total == 0 ? 0.0 : total * entropy(ones / total)

# ╔═╡ 272d95ec-6528-4d49-9aee-b298d019e7b1
"The coin has nothing to choose: one bit per bit."
fit_coin(x) = float(length(x))

# ╔═╡ 0fda43b9-a7fc-433b-81c0-bcbffae49f58
"Bernoulli is one coin for the whole string."
fit_bernoulli(x) = fit_flips(count(==('1'), x), length(x))

# ╔═╡ 969263bd-7f33-4cfe-a69e-5e84f5b4f2ba
"Markov is two coins: one flipped after each 0, and one after each 1. The first bit is a fair coin, 1 bit."
function fit_markov(x)
    pairs = collect(zip(x, Iterators.drop(x, 1)))    # each bit, with the bit after it
    after(a) = fit_flips(count(==((a, '1')), pairs), count(p -> first(p) == a, pairs))
    1 + after('0') + after('1')
end

# ╔═╡ 5d9c2ce8-4d5b-4ca4-a179-38c4a5f688b0
"Whatever `x` is, some θ has it as its first digits, and that θ gives `x` probability 1: 0 bits."
fit_digits(x) = 0.0

# ╔═╡ 75377882-5301-4050-a3c0-748a65972d22
MODELS = (
    coin      = fit_coin,
    bernoulli = fit_bernoulli,
    markov    = fit_markov,
    digits    = fit_digits
)

# ╔═╡ 60ce4a08-4274-4147-aeb3-dd35086fe852
md"""
## The best fit memorizes

Here are three strings of 20 bits, each from a different source: fair coin flips; a biased coin, which comes up 1 with probability 0.2; and a sticky source, where each bit repeats the one before it with probability 0.9.
"""

# ╔═╡ 1f67903a-80e4-4636-a4ac-5dbff0dedde4
N = 20

# ╔═╡ 9b1ebcba-e8d2-43a1-b459-f00ee7774b5f
md"""
## The price of hindsight

Go back to the four strings of 2 bits, and to Bernoulli. Each member of the model is an honest distribution: whatever ``\theta`` is, its probabilities add up to 1. Here are three members:

| string | ``\theta = 0`` | ``\theta = \tfrac{1}{2}`` | ``\theta = 1`` |
|:-:|:-:|:-:|:-:|
| `00` | 1 | ¼ | 0 |
| `01` | 0 | ¼ | 0 |
| `10` | 0 | ¼ | 0 |
| `11` | 0 | ¼ | 1 |
| total | 1 | 1 | 1 |

These three are the only ones that matter: for each string, one of them is the best member there is.

The best fit chooses ``\theta`` *after* seeing the string: ``\theta = 0`` for `00`, ``\theta = \tfrac{1}{2}`` for `01`, and so on. It takes the best entry in each row:
"""

# ╔═╡ 473c8ac6-5f8e-48e3-b568-622d8a7a76e8
"Every bit string of length `n`."
strings(n) = (string(i; base = 2, pad = n) for i in 0:2^n-1)

# ╔═╡ f15f73ef-576a-4e99-89e9-8bfd85beac16
md"""
!!! info "Parametric complexity, and the NML code (§1.4, §2.5.3)"
    Add up a model's best fits over every string of ``n`` bits. For a single distribution the total is 1; when ``\theta`` is chosen in hindsight, it is more:

    ```math
    S(n) = \sum_{y \text{ of } n \text{ bits}} P(y \mid \hat\theta(y))
    ```

    The overspend, in bits, is the model's *parametric complexity*:

    ```math
    \mathrm{COMP}(n) = \log_2 S(n)
    ```

    Divide every best fit by ``S(n)``, and they add up to exactly 1: a real distribution, and so a real code, the *normalized maximum likelihood* (NML) code. Its codelength is the best fit plus the complexity:

    ```math
    \begin{aligned}
    P_{\mathrm{NML}}(x) &= \frac{P(x \mid \hat\theta(x))}{S(n)} \\[6pt]
    L(x) = -\log_2 P_{\mathrm{NML}}(x) &= -\log_2 P(x \mid \hat\theta(x)) + \log_2 S(n) \\[2pt]
    &= \text{best fit} + \mathrm{COMP}(n)
    \end{aligned}
    ```

    - ``\hat\theta(x)`` is the member that fits ``x`` best, chosen after seeing it, and ``P(x \mid \hat\theta(x))`` is the probability it gives ``x``;
    - ``S(n)`` is the total of those best fits: 2.5 for Bernoulli at ``n = 2``;
    - dividing by ``S(n)`` is the scaling above: `00` gets ``1 / 2.5 = 0.4``;
    - ``L(x)`` adds the same ``\mathrm{COMP}(n)`` to every string: ``\log_2 2.5 = 1.32`` bits for Bernoulli at ``n = 2``.

    No other code has a smaller extra cost on its worst string: NML is the minimax optimal code for the model (§2.5.3).

One word has changed meaning since notebook 02. There, complexity, ``K(x)``, belonged to a string. Here it belongs to a model: ``\mathrm{COMP}(n)`` is fixed before any string arrives.

In code, the overspend is one line, the same for every model:
"""

# ╔═╡ d81b6ed2-f795-4943-95ec-a92a33e9b763
"Parametric complexity: log2 of the total probability a model's best fits promise to the strings of length `n`."
comp(fit, n) = log2(sum(x -> exp2(-fit(x)), strings(n)))

# ╔═╡ 1a3a3e17-5840-458f-a9f7-62cbe49b0867
md"""
## Complexity is what a model can fit

How big is each model's overspend? Here it is for every string length up to 20 bits, each a sum over all ``2^n`` strings of that length:
"""

# ╔═╡ d520f757-fb6e-44dd-bff1-421b351e195e
COMPS = [map(f -> comp(f, n), MODELS) for n in 1:N];

# ╔═╡ c1655611-a149-4d3c-8932-3ef7718adbea
md"""
### What each model sees

Why do Bernoulli and Markov stay so small? A model's best fit depends on a string only through a few counts, its *view* of the string:

- the coin sees only the length;
- Bernoulli sees ``k``, the number of 1s;
- Markov sees the first bit, and how often each pair of neighbours occurs: `00`, `01`, `10`, `11`;
- Digits sees every bit.

Two strings with the same view get the same best fit. Here is how many different views each model has of the strings of 20 bits:
"""

# ╔═╡ 888cbb21-10b2-4c62-858e-dc74e4a14cbe
VIEWS = (
    coin = x -> length(x),
    bernoulli = x -> count(==('1'), x),
    markov = x -> (first(x), sort(collect(zip(x, Iterators.drop(x, 1))))),
    digits = identity,
);

# ╔═╡ f95de403-1c2d-4a6e-b073-fe26e87fb1f4
md"""
## Fit + COMP

Now charge each model for what it can fit. The NML code gives every model a score for a string: its best fit plus its complexity. That is the length of a real code, not a penalty someone made up. It is the price of choosing in hindsight, paid back. The shortest code wins.
"""

# ╔═╡ f92f13ea-f4d7-487c-8fb6-48a5e597f921
"Each model's NML codelength for `x`: its best fit plus its complexity."
score(x) = map((f, c) -> f(x) + c, MODELS, COMPS[length(x)])

# ╔═╡ 8c8da1a1-654b-40c2-bf30-815aa1003583
md"""
## Too simple, too complex

Climb a ladder of models. In an *order-``k`` Markov* model, each bit is a coin chosen by the ``k`` bits before it, so there are ``2^k`` coins. Order 0 is Bernoulli, order 1 is our Markov model, and order 5 has 32 coins.

To keep the rungs comparable, every order pays a fair coin for the same first few bits, and models the rest. Then the ladder is nested: each rung can do whatever the one below it can, so its best fit can only be better.

The source alternates, `0101…`, and each bit breaks the pattern with probability 0.1. That makes it an order-1 chain, so the right rung is ``k = 1``.
"""

# ╔═╡ 2d18e56b-52a4-4506-b62b-6fa1724509ce
# The first bits, paid for as fair coins by every order.
PREFIX = 5

# ╔═╡ f5e0ff8b-8726-422d-9107-3fc6275ad079
ORDERS = 0:PREFIX

# ╔═╡ fd4dc448-78ac-4190-83f9-61010a9e8929
"The bits that follow each context of `k` bits in `x`, after the prefix."
function followers(x, k)
    groups = Dict{String,Vector{Char}}()
    for t in PREFIX+1:length(x)
        push!(get!(groups, x[t-k:t-1], Char[]), x[t])
    end
    groups
end

# ╔═╡ fb4fc2c8-f003-4bba-b973-2ef99e16674b
"An order-`k` Markov model is one coin for each context of `k` bits."
fit_order(k) = x -> PREFIX + sum(v -> fit_flips(count(==('1'), v), length(v)), values(followers(x, k)))

# ╔═╡ 22d2bd2f-4db8-40a2-9bf0-2df0fc96ea71
# Each order's complexity at 20 bits, summed over all 2^20 strings: a few seconds.
ORDER_COMPS = [comp(fit_order(k), N) for k in ORDERS];

# ╔═╡ 76438f4d-3dff-4602-a5e6-b1e35ac34459
"Each order's NML codelength for `x`."
order_scores(x) = [fit_order(k)(x) for k in ORDERS] .+ ORDER_COMPS

# ╔═╡ b7310c9a-bf82-4ddd-8667-3d9e3cc25a4c
md"""
### Held-out data agrees

For a neural network, the complexity is out of reach: it would sum the network's best fit over every possible dataset. In practice, people hold some data out, and measure the loss there.

Here is that for the ladder. Each rung is fit to the 20 bits above, then asked to predict 10,000 fresh bits from the same source. Its prediction for a bit is the share of 1s that followed the same context in the 20 bits, nudged towards ½, so that a context it never saw predicts ½ rather than 0 or 1:
"""

# ╔═╡ 526009b4-17f4-4970-999f-cad7c5532ba0
"The chance of a 1, from the bits `seen` in the same context, with half a 0 and half a 1 added."
p_one(seen) = (count(==('1'), seen) + 1/2) / (length(seen) + 1)

# ╔═╡ 686baf9d-fd2a-4470-9a62-3e697980e2a9
"The bits it costs to code bit `b` when a 1 was predicted with probability `p`."
cost(b, p) = -log2(b == '1' ? p : 1 - p)

# ╔═╡ 31c5d2dd-ea02-4c8c-b560-f2391d25e8f5
"Held-out loss in bits per bit: the order-`k` model fit to `x`, predicting the fresh string `y`."
heldout(x, y, k) = let seen = followers(x, k)
    sum(t -> cost(y[t], p_one(get(seen, y[t-k:t-1], Char[]))), PREFIX+1:length(y)) / (length(y) - PREFIX)
end

# ╔═╡ 7f300a9e-5ac3-4ba6-90f7-e1a74ab18f36
md"""
## Appendix: helpers

The sources of strings, the models' names, number formatting, and the chart style. The series use the first slots of a colour-blind-safe categorical palette, in fixed order.
"""

# ╔═╡ fc41683f-2197-4243-bdbd-0629c4b4d7ac
"`n` fair coin flips."
coin_flips(rng, n) = join(rand(rng, 0:1, n))

# ╔═╡ 14fb5bda-5a00-4ab0-ae88-5d6479e427b3
"`n` flips of a coin that comes up 1 with probability `p`."
biased(rng, n; p = 0.2) = join(Int.(rand(rng, n) .< p))

# ╔═╡ 20a9c6a8-2966-4b3a-90e1-fa90f743173e
"Each bit repeats the one before it with probability `stay`: the running count of switches, mod 2."
sticky(rng, n; stay = 0.9) = join(mod.(rand(rng, 0:1) .+ cumsum([0; rand(rng, n - 1) .> stay]), 2))

# ╔═╡ 1f44c11b-e71d-4aab-bb72-b0048593de1b
EXAMPLES = let rng = Xoshiro(3)
    [(source = "coin flips", x = coin_flips(rng, N)),
     (source = "biased", x = biased(rng, N)),
     (source = "sticky", x = sticky(rng, N))]
end;

# ╔═╡ 7f9c977f-a056-4d23-ba20-e4294302031b
"Alternating bits: each bit repeats the one before it, breaking the pattern, with probability `p`."
alternating(rng, n; p = 0.1) = sticky(rng, n; stay = p)

# ╔═╡ 7ff5325e-0c0b-42bf-b988-229f6de6743d
# A string from the source, and 10,000 fresh bits from it for later.
ladder = let rng = Xoshiro(1)
    x = alternating(rng, N)
    fresh = alternating(rng, 10_000)
    fits = [fit_order(k)(x) for k in ORDERS]
    (; x, fresh, fits, scores = fits .+ ORDER_COMPS)
end;

# ╔═╡ 69948aa5-3134-44da-82ef-7caf624226c7
# Which order MDL picks, for each of 1,000 strings from the source.
ladder_picks = let rng = Xoshiro(2)
    [ORDERS[argmin(order_scores(alternating(rng, N)))] for _ in 1:1000]
end;

# ╔═╡ c6c8f9e9-5c2a-437b-8ef7-18f55f08aaf6
NAMES = (coin = "coin", bernoulli = "Bernoulli", markov = "Markov", digits = "Digits")

# ╔═╡ dca70f21-6f81-4a4e-8957-0bd15d0b0f7a
"The name and value of the smallest entry of a named tuple. Ties go to the first."
best(scores) = let (value, name) = findmin(scores)
    (; name, value)
end

# ╔═╡ cc8e4a7c-28a2-4469-8532-0e7bbe0c677a
"`x` with `d` decimals."
fixed(x, d = 1) = Printf.format(Printf.Format("%.$(d)f"), x)

# ╔═╡ c09c5b7a-0923-4e5e-8350-973245432a84
let x = "01000100"
    machines = [
        ("coin", "none", predict_coin()),
        ("Bernoulli", "``\\theta = 1/4``", predict_bernoulli(1 // 4)),
        ("Markov", "``\\theta_0 = 2/5``, ``\\theta_1 = 0``", predict_markov(2 // 5, 0 // 1)),
        ("Digits", "``\\theta = 0.$(x)\\ldots_2``", predict_digits(x)),
    ]
    rows = map(machines) do (name, knobs, predict)
        code = encode(x, predict)
        "| $name | $knobs | $(fixed(ideal(x, predict), 2)) | $(isempty(code) ? "(nothing)" : "`$code`") | $(length(code)) | `$(decode(code, predict, length(x)))` |"
    end
    Markdown.parse("""
    Here is `$x` through all four machines, each with its knobs set to fit the string best:

    | machine | knobs | ideal (bits) | codeword | bits | decodes to |
    |:--|:--|--:|:--|--:|:--|
    $(join(rows, "\n"))

    - **The coin** halves the interval at every step, so its codeword is the string itself: one bit per bit.
    - **Bernoulli** expects 0s, and pays 2 bits for each 1.
    - **Markov** flips one coin after each 0 and another after each 1. In this string a 1 is never followed by another 1, so ``\\theta_1 = 0``, and every bit after a 1 is free.
    - **Digits** sends nothing at all. Its knob already holds the string, and the receiver needs the knob to decode. The string didn't vanish: it moved into the setting.
    """)
end

# ╔═╡ c8a21d50-cb30-4685-859e-5a0d2a6f8f1c
let promises = [(x, exp2(-fit_bernoulli(x))) for x in strings(2)]
    total = sum(last, promises)
    rows = ["| `$x` | $(fixed(count(==('1'), x) / 2)) | $(fixed(p, 2)) | $(fixed(fit_bernoulli(x))) |" for (x, p) in promises]
    Markdown.parse("""
    | string | best ``\\theta`` | its probability | best fit (bits) |
    |:-:|:-:|:-:|:-:|
    $(join(rows, "\n"))
    | total | | $(fixed(total, 2)) | |

    Each row was answered as if its string were the only one that could happen. Together they promise $(fixed(total)), out of a budget of 1.

    Try to send with these promises. `00` is promised 0 bits, and so is `11`. The receiver gets an empty message: which one was it? There is no telling. The best fit chose ``\\theta`` from the string, and the receiver hasn't seen the string. A choice made in hindsight is a promise no code can keep.

    The way out is to scale every promise down until they add up to 1: divide by $(fixed(total)). The four probabilities become $(join((fixed(p / total) for (_, p) in promises), ", ")), a real distribution, and so a real code. Every string pays the same for it: ``\\log_2 $(fixed(total)) = $(fixed(log2(total), 2))`` bits on top of its best fit. That is the price of not knowing ``\\theta`` in advance, and nothing in it was chosen by us. And ``\\theta`` itself is never sent: the scaled code works without it, and those $(fixed(log2(total), 2)) bits are its price.
    """)
end

# ╔═╡ 6e2c0b6e-e484-4711-9ef6-3dd83089f727
let c = COMPS[N]
    Markdown.parse("""
    - **The coin** stays at 0. A single distribution has nothing to choose, so it never overspends.
    - **Digits** climbs the diagonal: its complexity is exactly ``n``. Each member promises one string certainty, so the promises add up to ``2^n``. Its best fit, 0 bits, plus its complexity, ``n`` bits, is the length of the string itself: memorizing buys nothing.
    - **Bernoulli and Markov** grow slowly, to $(fixed(c.bernoulli, 2)) and $(fixed(c.markov, 2)) bits at $N bits.

    Digits has one parameter, like Bernoulli, and is by far the most complex model here. Markov has two, and costs only $(fixed(c.markov - c.bernoulli)) bits more than Bernoulli. A single parameter can hold any number of bits; what counts is how much the model can do with it.
    """)
end

# ╔═╡ 70ae7e3a-aed5-4bdf-8210-9d06a1b88f60
let rows = map(EXAMPLES) do (; source, x)
        s = score(x)
        "| $source | `$x` | $(join(fixed.(values(s)), " | ")) | $(NAMES[best(s).name]) |"
    end
    margin(s) = let v = sort(collect(values(s))); v[2] - v[1] end
    sticky, biased = score(EXAMPLES[3].x), score(EXAMPLES[2].x)
    Markdown.parse("""
    | source | string | coin | Bernoulli | Markov | Digits | MDL picks |
    |:--|:--|--:|--:|--:|--:|:--|
    $(join(rows, "\n"))

    Scores in bits.

    - **The coin flips** stay with the coin. Markov fits them a little better, by chance, but not by enough to pay for its complexity.
    - **Digits** scores exactly $N on every string, the same as the coin. Its perfect fit is cancelled by its complexity, bit for bit, so memorizing buys nothing. (Ties go to the coin, listed first.)
    - **The sticky string** goes to Markov, by $(fixed(margin(sticky))) bits.
    - **The biased string** goes to Markov too, not to Bernoulli. It bunches its 1s, `11` twice, and Markov, which sees what follows what, describes it $(fixed(biased.bernoulli - biased.markov)) bits shorter. MDL judges the string in front of it, not the source we drew it from, and in $N bits a chance pattern can pay for itself.

    Those were four unrelated models. What happens along a ladder of ever richer ones?
    """)
end

# ╔═╡ 898157c5-a8a2-4376-8201-f03dd626b542
let losses = [heldout(ladder.x, ladder.fresh, k) for k in ORDERS]
    rows = ["| $k | $(fixed(l, 2)) |" for (k, l) in zip(ORDERS, losses)]
    Markdown.parse("""
    | order ``k`` | held-out loss (bits per bit) |
    |:-:|--:|
    $(join(rows, "\n"))

    Validation agrees: ``k = $(ORDERS[argmin(losses)])``, at $(fixed(minimum(losses), 2)) bits per bit. The source itself costs $(fixed(entropy(0.1), 2)), the price of a bit that breaks the pattern one time in ten.

    The agreement isn't luck: validation and MDL are close kin. Grünwald's *prequential* interpretation (§2.6.4) codes a string by predicting each bit from the ones before it, paying ``-\\log_2`` of the prediction, then learning from the bit. The total is a codelength, and it punishes memorizing, because a model that memorizes predicts the next bit badly. A validation set does something similar once, in a single split.

    Large language models make the link literal. In a single pass over the data, every token is predicted before the model trains on it, so the summed training loss is a prequential codelength for the data. Blier and Ollivier [used such codes](https://arxiv.org/abs/1802.07044) to show that deep networks compress their training data well, even counting the cost of the model.
    """)
end

# ╔═╡ 06de6922-cc2a-4b3f-94cd-72bf03952cc5
let periodic = repeat("0001", N ÷ 4)
    four = best(score(periodic))
    rungs = order_scores(periodic)
    Markdown.parse("""
    ---

    This notebook began by settling for a smaller machine. In it, the best fit memorized, and what caught it was pricing the whole model by how far its hindsight overspends the budget of 1: a price that counts the strings a model can fit, not the parameters it has.

    The language also decides what can be seen. None of our four models sees the period in `0001` repeated: the best, $(NAMES[four.name]), describes $N bits of it in $(fixed(four.value)). Order $(ORDERS[argmin(rungs)]) of the ladder sees it, since any three bits fix the next, and MDL picks it at $(fixed(minimum(rungs))) bits. MDL finds what its model can express, and nothing more.

    A single weight, like Digits' ``\\theta``, can hold any number of bits, which is why Hinton and van Camp, next on the list, measure the *information* in a network's weights rather than counting them. One twist to carry along: Blier and Ollivier found that the variational codes grown from that idea give "surprisingly poor" bounds on deep networks, while prequential codes do well.
    """)
end

# ╔═╡ dcd3e964-6927-4c7c-b53a-3681930f0283
"A fraction, as `numerator/denominator`."
fraction(p) = "$(numerator(p))/$(denominator(p))"

# ╔═╡ 2bfba18c-093f-48e3-9ac9-f0403b0ea14c
let x = "01000100", predict = prefix -> 1 // 4
    code = encode(x, predict)
    c = number(code)
    before = [WHOLE; intervals(x, predict)[1:end-1]]
    rows = map(enumerate(before)) do (t, (low, high))
        p_one = predict(x[1:t-1])
        p_bit = x[t] == '1' ? p_one : 1 - p_one
        _, one_side = sides((low, high), p_one)
        "| $t | `$(x[t])` | $(fraction(p_bit)) | [$(fixed(Float64(low), 4)), $(fixed(Float64(high), 4))) | $(fixed(Float64((c - low) / (high - low)), 3)) | `$(c < first(one_side) ? 0 : 1)` | $(fixed(-log2(p_bit), 3)) |"
    end
    wrong = prefix -> 9 // 10
    Markdown.parse("""
    Encode `$x` with a predictor that expects a 1 one time in four, `prefix -> 1 // 4`. The codeword is `$code`, $(length(code)) bits, which names ``c = 0.$(code)_2 = $(Float64(c))``. Now decode it:

    | bit | value | its probability | interval before | where ``c`` falls | decoded | cost (bits) |
    |:-:|:-:|:-:|:--|:-:|:-:|--:|
    $(join(rows, "\n"))

    "Where ``c`` falls" is its place in the interval, from 0 at the left end to 1 at the right. The 0 side covers the first ¾, so the decoder's rule is one comparison: below ¾, the bit is 0; otherwise, 1. The decoded column comes from ``c`` alone, and it matches the string.

    The two 1s are the surprises. Each keeps a quarter of the interval and costs 2 bits; each 0 keeps three quarters and costs 0.415. Together they cost $(fixed(ideal(x, predict), 2)) bits, the ideal. The codeword is a little longer, because it must be whole bits and must leave room for whatever comes after it. `101` alone also names 0.625, but a number that merely starts 0.101… can run up to 0.75, outside the interval. The trailing zeros pin it down.

    A model needn't be right about every bit. The bits it finds unlikely are still sent; they just cost more. So the code stays lossless, and an imperfect model shows up as a longer codeword, not as lost data. A predictor that is badly wrong about this string, expecting a 1 nine times in ten, still sends it exactly, in $(length(encode(x, wrong))) bits.
    """)
end

# ╔═╡ ab41a3d5-2220-49b6-b24f-f92aa1940957
let x = "01000100", settings = [1 // 10, 1 // 5, 1 // 4, 1 // 3, 1 // 2, 3 // 4, 9 // 10]
    row(f) = join((f(θ) for θ in settings), " | ")
    Markdown.parse("""
    | ``\\theta`` | $(row(fraction)) |
    |:--|$(repeat("--:|", length(settings)))
    | ideal (bits) | $(row(θ -> fixed(ideal(x, predict_bernoulli(θ)), 2))) |
    | codeword (bits) | $(row(θ -> length(encode(x, predict_bernoulli(θ))))) |

    Fitting a model means turning its knobs until the codeword is shortest. Here the best setting is ``\\theta = 1/4``, the share of 1s in the string, and settings far from it cost far more.

    Turning a knob one setting at a time is slow, and for coins there is a shortcut. Flip a coin ``n`` times and get ``k`` ones, and the bias that fits best is always ``k/n``. With that bias, each flip costs ``H(k/n)`` bits on average, where ``H`` is the [binary entropy](https://en.wikipedia.org/wiki/Binary_entropy_function), so the flips cost ``n \\, H(k/n)`` bits in all.

    All four models come down to coin flips, so each best fit is one short function, which takes a string and returns its best fit in bits.
    """)
end

# ╔═╡ 4f92e3af-e842-49c2-aa36-e08e7b6a93f2
"A whole number, with commas between the thousands."
commas(n::Integer) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => ",")

# ╔═╡ c0cc560b-e646-43a8-bf3b-ed58f95e2d15
let rows = map(EXAMPLES) do (; source, x)
        fits = map(f -> f(x), MODELS)
        "| $source | `$x` | $(join(fixed.(values(fits)), " | ")) | $(NAMES[best(fits).name]) |"
    end
    trained = EXAMPLES[2].x    # the biased string
    fresh = let rng = Xoshiro(4)
        [biased(rng, N) for _ in 1:1000]
    end
    bernoulli_bits = sum(s -> ideal(s, predict_bernoulli(count(==('1'), trained) // N)), fresh) / length(fresh)
    matches = count(==(trained), fresh)
    Markdown.parse("""
    | source | string | coin | Bernoulli | Markov | Digits | max likelihood picks |
    |:--|:--|--:|--:|--:|--:|:--|
    $(join(rows, "\n"))

    Pick the model with the best fit, in bits, as maximum likelihood does, and Digits wins every time, at 0 bits. It hasn't found anything: it has **memorized** the string, and it would memorize any other just as well.

    Why is that bad? A model is for what comes next, and memorizing says nothing about it. Fit to the biased string, Bernoulli has learned that 1s are rare: it describes fresh strings from the same source in $(fixed(bernoulli_bits)) bits on average, against the coin's $N. Digits, fit to the same string, is certain the next one will be identical. It gives every other string probability 0, so a code built on it can send $(matches == 0 ? "none" : "only $matches") of $(commas(length(fresh))) fresh strings.

    Its perfect fit was borrowed, too. Digits fits at 0 bits only because its knob holds the string, and the receiver needs the knob to decode, as the four machines showed.

    ### Charging for θ

    Counting parameters doesn't catch it either. Digits has one, the same as Bernoulli, and one fewer than Markov.

    The obvious repair is to charge for the parameter: describe ``\\theta`` first, then the string with its help, and add the two up. That two-part code is what Grünwald calls *crude* MDL (§1.3). But how many bits is ``\\theta``?

    - Bernoulli's best ``\\theta`` is ``k/n``, one of $(N + 1) fractions, so about $(fixed(log2(N + 1))) bits pick it out.
    - Digits needs all $N digits of its ``\\theta``, and pays $N bits for them: no better than sending the string as it is.

    So crude MDL can catch Digits, but only after we decide how to write ``\\theta`` down, and that decision is ours. Round Bernoulli's ``\\theta`` to 2 decimal places or to 10, and its total shifts with the choice. Grünwald's verdict on such codes: the procedure is "in danger of becoming arbitrary" (§1.4).

    Is there a price for a model that doesn't depend on how we write ``\\theta`` down?
    """)
end

# ╔═╡ 979fdbab-e938-4791-a387-b038ab595fcc
let views = map(view -> length(unique(view, strings(N))), VIEWS), c = COMPS[N]
    rows = ["| $(NAMES[name]) | $(commas(views[name])) | $(fixed(log2(views[name]), 2)) | $(fixed(c[name], 2)) |" for name in keys(VIEWS)]
    Markdown.parse("""
    | model | views | ``\\log_2`` views | complexity |
    |:--|--:|--:|--:|
    $(join(rows, "\n"))

    For Digits, the complexity is exactly the log of its views: it fits every one of them perfectly. Bernoulli's is well below. Of its $(views.bernoulli) views, only about $(fixed(exp2(c.bernoulli))) count: neighbouring biases, 9 ones in 20 and 10 ones in 20, say, are too close to tell apart from $N bits.

    That is Grünwald's *counting interpretation* (§2.6.2): a model's complexity is the log of how many of its members can be told apart.
    """)
end

# ╔═╡ 8e9e240c-cf4f-47bb-a001-93dca98f4b72
let (; x, fits) = ladder
    rows = ["| $k | $(count(==(k), ladder_picks)) |" for k in ORDERS]
    Markdown.parse("""
    On one string from the source, `$x`:

    - **Too simple.** Order 0 sees only how many 1s there are, about half, and fits the string like a fair coin: $(fixed(fits[1])) bits.
    - **About right.** Order 1 sees the alternation: $(fixed(fits[2])) bits of fit, for $(fixed(ORDER_COMPS[2])) bits of complexity.
    - **Too complex.** Orders 2 to 5 fit at most $(fixed(fits[2] - minimum(fits[3:end]))) bit better, and pay up to $(fixed(ORDER_COMPS[end] - ORDER_COMPS[2])) bits more for their extra coins.

    Order 5 has 32 coins and is still less complex than Digits with its one parameter: $(fixed(ORDER_COMPS[end])) bits against $N.

    It isn't one string's luck. Over $(commas(length(ladder_picks))) strings from the source, MDL picks:

    | order ``k`` | strings |
    |:-:|--:|
    $(join(rows, "\n"))
    """)
end

# ╔═╡ fa7af5a4-f282-4796-98a6-8733bdcd70c7
VIZ = (surface = "#fcfcfb", text = "#0b0b0b", text2 = "#52514e", grid = "#e4e3de")

# ╔═╡ 71eced53-6238-4d35-bbc9-04950b90aaee
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# ╔═╡ 82c6e1ff-4967-41cc-b722-f5289fa024ad
THEME = Theme(
    fontsize = 13, backgroundcolor = VIZ.surface, textcolor = VIZ.text,
    Axis = (backgroundcolor = VIZ.surface, xgridcolor = VIZ.grid, ygridcolor = VIZ.grid,
            topspinevisible = false, rightspinevisible = false, leftspinecolor = VIZ.text2, bottomspinecolor = VIZ.text2,
            xtickcolor = VIZ.text2, ytickcolor = VIZ.text2, xticklabelcolor = VIZ.text2, yticklabelcolor = VIZ.text2,
            titlealign = :left, titlefont = :regular, titlesize = 14),
    Legend = (framevisible = false, backgroundcolor = VIZ.surface),
);

# ╔═╡ 7e5c017c-5806-4719-8075-d9789eca93e8
with_theme(THEME) do
    fig = Figure(size = (760, 440))
    ax = Axis(fig[1, 1]; xticks = 0:4:N, xlabel = "string length n (bits)", ylabel = "complexity (bits)",
              title = "Digits, with one parameter, overspends the most")
    for (name, color) in zip(keys(MODELS), SERIES_COLORS)
        values = [c[name] for c in COMPS]
        lines!(ax, 1:N, values; color, linewidth = 2, label = NAMES[name])
        text!(ax, N, last(values); text = NAMES[name], align = (:left, :center), offset = (8, 0), color = VIZ.text)
    end
    xlims!(ax, 0, N + 3)
    Legend(fig[2, 1], ax; orientation = :horizontal)
    fig
end

# ╔═╡ bf5a144b-8f72-4a05-9e3e-db0e8b559941
let (; fits, scores) = ladder
    kbest = ORDERS[argmin(scores)]
    series = [("best fit", fits), ("complexity", ORDER_COMPS), ("fit + complexity", scores)]
    with_theme(THEME) do
        fig = Figure(size = (760, 440))
        ax = Axis(fig[1, 1]; xticks = ORDERS, xlabel = "order k", ylabel = "bits",
                  title = "Fit falls, complexity rises, and their sum bottoms out at k = $kbest")
        for ((label, values), color) in zip(series, SERIES_COLORS)
            scatterlines!(ax, ORDERS, values; color, label, linewidth = 2, markersize = 9)
            text!(ax, last(ORDERS), last(values); text = label, align = (:left, :center), offset = (10, 0), color = VIZ.text)
        end
        scatter!(ax, [kbest], [minimum(scores)]; color = VIZ.text, markersize = 12, strokecolor = VIZ.surface, strokewidth = 2)
        xlims!(ax, -0.3, last(ORDERS) + 1.8)
        ylims!(ax, 0, nothing)
        Legend(fig[2, 1], ax; orientation = :horizontal)
        fig
    end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
CairoMakie = "~0.15.16"
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.3"
manifest_format = "2.1"
project_hash = "fc2096343272ca2231623ee0d1e2e80b1bc1ee91"

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
git-tree-sha1 = "d0f666deaf8099a7d7ac229a74402ada88898be1"
registries = "General"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.9.0+0"

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
git-tree-sha1 = "ef46833ad806dc03d95dbe86db3d113965c461cd"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.12.2"

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
version = "2026.10.20"

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
version = "10.47.0+0"

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
git-tree-sha1 = "044ed182312a8ad0f28437f048f3e1eae9093a80"
registries = "General"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.11.1"

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
# ╟─9e9ed333-b8c3-4513-97e7-f610b20fa6d0
# ╠═75387367-b23d-469a-be55-5ff5a256494b
# ╟─b37ccf49-8d6b-4838-ba75-872c194c3df2
# ╟─6852bb93-1cbc-43dc-94bd-c44cc60c92be
# ╟─bc23a13c-e665-4d59-90f0-a68b248f88b5
# ╠═ac3ef555-c4c5-44b8-bcfa-c207cd665802
# ╠═9c47fc2f-170d-41a9-8d77-27d2afb75b94
# ╠═8e7ed6d1-c282-4d53-b01e-5e3cc2807807
# ╠═516998c3-cbc3-481b-98ea-4618980ee161
# ╠═6686b3c5-d261-496d-b4f6-b1e992ab103d
# ╠═b0601d04-26a8-4342-b621-a7a9d2cc790e
# ╠═14ef12f7-4060-4068-81ba-0ca8e890759c
# ╠═b6f86f8c-0017-4d96-b9bf-88dbcc9bf8ba
# ╟─2bfba18c-093f-48e3-9ac9-f0403b0ea14c
# ╟─78208971-5f13-4f4b-9458-46178e82f425
# ╟─f1302616-023a-4596-b1b5-bab3769302b0
# ╠═891d29ca-9e98-4ec5-bc78-e8f02d3250a4
# ╠═8994982b-795d-4167-9ffe-76741582215d
# ╠═1b4a5364-5019-4b46-b891-eedc56924475
# ╠═99316a9a-9864-49d2-9bf0-ee4ef03776a6
# ╟─c09c5b7a-0923-4e5e-8350-973245432a84
# ╟─e2a85bf1-ce9b-4043-ba74-6ffb1c2184c6
# ╟─ab41a3d5-2220-49b6-b24f-f92aa1940957
# ╠═ad357c47-1f0f-495f-8ead-fdbbffc0eebf
# ╠═79656853-1eaf-48e6-be5d-86bb6cdadfbd
# ╠═272d95ec-6528-4d49-9aee-b298d019e7b1
# ╠═0fda43b9-a7fc-433b-81c0-bcbffae49f58
# ╠═969263bd-7f33-4cfe-a69e-5e84f5b4f2ba
# ╠═5d9c2ce8-4d5b-4ca4-a179-38c4a5f688b0
# ╠═75377882-5301-4050-a3c0-748a65972d22
# ╟─60ce4a08-4274-4147-aeb3-dd35086fe852
# ╠═1f67903a-80e4-4636-a4ac-5dbff0dedde4
# ╠═1f44c11b-e71d-4aab-bb72-b0048593de1b
# ╟─c0cc560b-e646-43a8-bf3b-ed58f95e2d15
# ╟─9b1ebcba-e8d2-43a1-b459-f00ee7774b5f
# ╠═473c8ac6-5f8e-48e3-b568-622d8a7a76e8
# ╟─c8a21d50-cb30-4685-859e-5a0d2a6f8f1c
# ╟─f15f73ef-576a-4e99-89e9-8bfd85beac16
# ╠═d81b6ed2-f795-4943-95ec-a92a33e9b763
# ╟─1a3a3e17-5840-458f-a9f7-62cbe49b0867
# ╠═d520f757-fb6e-44dd-bff1-421b351e195e
# ╟─7e5c017c-5806-4719-8075-d9789eca93e8
# ╟─6e2c0b6e-e484-4711-9ef6-3dd83089f727
# ╟─c1655611-a149-4d3c-8932-3ef7718adbea
# ╠═888cbb21-10b2-4c62-858e-dc74e4a14cbe
# ╟─979fdbab-e938-4791-a387-b038ab595fcc
# ╟─f95de403-1c2d-4a6e-b073-fe26e87fb1f4
# ╠═f92f13ea-f4d7-487c-8fb6-48a5e597f921
# ╟─70ae7e3a-aed5-4bdf-8210-9d06a1b88f60
# ╟─8c8da1a1-654b-40c2-bf30-815aa1003583
# ╠═2d18e56b-52a4-4506-b62b-6fa1724509ce
# ╠═f5e0ff8b-8726-422d-9107-3fc6275ad079
# ╠═fd4dc448-78ac-4190-83f9-61010a9e8929
# ╠═fb4fc2c8-f003-4bba-b973-2ef99e16674b
# ╠═22d2bd2f-4db8-40a2-9bf0-2df0fc96ea71
# ╠═76438f4d-3dff-4602-a5e6-b1e35ac34459
# ╠═7ff5325e-0c0b-42bf-b988-229f6de6743d
# ╟─bf5a144b-8f72-4a05-9e3e-db0e8b559941
# ╠═69948aa5-3134-44da-82ef-7caf624226c7
# ╟─8e9e240c-cf4f-47bb-a001-93dca98f4b72
# ╟─b7310c9a-bf82-4ddd-8667-3d9e3cc25a4c
# ╠═526009b4-17f4-4970-999f-cad7c5532ba0
# ╠═686baf9d-fd2a-4470-9a62-3e697980e2a9
# ╠═31c5d2dd-ea02-4c8c-b560-f2391d25e8f5
# ╟─898157c5-a8a2-4376-8201-f03dd626b542
# ╟─06de6922-cc2a-4b3f-94cd-72bf03952cc5
# ╟─7f300a9e-5ac3-4ba6-90f7-e1a74ab18f36
# ╟─fc41683f-2197-4243-bdbd-0629c4b4d7ac
# ╟─14fb5bda-5a00-4ab0-ae88-5d6479e427b3
# ╟─20a9c6a8-2966-4b3a-90e1-fa90f743173e
# ╟─7f9c977f-a056-4d23-ba20-e4294302031b
# ╟─c6c8f9e9-5c2a-437b-8ef7-18f55f08aaf6
# ╟─dca70f21-6f81-4a4e-8957-0bd15d0b0f7a
# ╟─cc8e4a7c-28a2-4469-8532-0e7bbe0c677a
# ╟─dcd3e964-6927-4c7c-b53a-3681930f0283
# ╟─4f92e3af-e842-49c2-aa36-e08e7b6a93f2
# ╟─fa7af5a4-f282-4796-98a6-8733bdcd70c7
# ╟─71eced53-6238-4d35-bbc9-04950b90aaee
# ╟─82c6e1ff-4967-41cc-b722-f5289fa024ad
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
