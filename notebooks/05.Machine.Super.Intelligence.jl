### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ 05725c63-8e71-54fb-a753-1d7886e63d3a
begin
    using CairoMakie, PlutoUI
    using Printf, Random
end

# ╔═╡ 942b639c-bf53-5823-8d4f-de5881d3c04f
md"""
# Machine Super Intelligence

*Bet on every predictor at once. What does it cost not to know which one is right?*

**Prerequisites**

- [Notebook 02](https://zxygentoo.github.io/the-ninety-percent-list/02.Kolmogorov.Complexity.and.Algorithmic.Randomness.html), or the idea that ``K(x)`` is the length of the shortest program that prints ``x``.
- [Notebook 03](https://zxygentoo.github.io/the-ninety-percent-list/03.A.Tutorial.Introduction.to.the.Minimum.Description.Length.Principle.html), or the idea that a distribution is a code: ``x`` costs ``-\log_2 P(x)`` bits.
- [Notebook 04](https://zxygentoo.github.io/the-ninety-percent-list/04.Keeping.Neural.Networks.Simple.by.Minimizing.the.Description.Length.of.the.Weights.html), or the KL divergence: the bits lost, on average, by coding with the wrong distribution.
- Bayes' rule.

---

A process ``\mu`` produces a bit sequence. Here are the first ten bits:

`1010101010`

What's the chance the next bit is a 1?

Alternation says it's certain. But a process that alternates for ten bits, then prints only 0s, fits just as well, and so do endlessly many others. Even if we know that some program computes ``\mu``, no run of bits, however long, singles that program out.

Solomonoff's answer is not to choose. Bet on every predictor a program can describe, all at once, and bet more on those with shorter programs: alternation takes a few bits to write, while the switch to 0s must also spell out when it happens, so its program is longer.

The bet is a mixture of all of them, written ``\xi``.

Its promise (Theorem 2.7.1) is that whatever the true process ``\mu``, the mixture's total error stays below a constant, proportional to ``K(\mu)``, ``\mu``'s *Kolmogorov complexity*: the length of the shortest program that computes it. It's notebook 02's ``K`` for a process.

``\xi`` can't be computed, since its list holds every program. So we build a small one, short enough to write out, and check its promise empirically, in two ways:

- **In bits:** on every sequence, ``\xi`` needs at most ``K(\mu)`` bits more than ``\mu`` itself.
- **In predictions:** on average, ``\xi``'s total squared error is at most ``\frac{\ln 2}{2} K(\mu)``.

This is the prediction half of chapter 2 (§2.4–2.7) of Shane Legg's [thesis](https://www.vetta.org/documents/Machine_Super_Intelligence.pdf). Section, definition and page numbers refer to it.
"""

# ╔═╡ b10a5215-4d41-5ccd-a6ad-eaec38622508
TableOfContents()

# ╔═╡ 6b0eadbf-0ef0-432a-af32-5e3d4519ce69
md"""
## The predictor

A *predictor* reads the last ``k`` bits and gives the chance that the next one is a 1.

We call ``k`` its *order*, and the last ``k`` bits its *context*. There are ``2^k`` possible contexts, and a predictor of order ``k`` gives a chance for each of them.

To write a predictor down as a program, a string of bits, we have to encode its order and its ``2^k`` chances.

The order is simple. We write it in unary, which marks its own end with a 0:
"""

# ╔═╡ 26956974-c66e-5449-8757-79b8acacbc04
"""
`n` in unary: `n` ones, then a zero, so a reader can tell where it ends.

    unary(0) == "0"
    unary(1) == "10"
    unary(2) == "110"
    unary(8) == "111111110"
"""
unary(n) = "1"^n * "0"

# ╔═╡ 2b8418e1-76c6-51d9-a691-942ab98641e3
md"""
The chances need a *precision* ``b``: each is written in ``b`` bits, which split ``[0, 1]`` into ``2^b`` equal cells. Each bit halves the cell it's in, and the chance is the middle of the last cell. With no bits, that's the middle of ``[0, 1]``, ½.

```
b = 0   [                1/2                ]
b = 1   [       1/4       |       3/4       ]
b = 2   [  1/8   |  3/8   |  5/8   |  7/8   ]
```

So no chance is ever 1 or 0: no prediction is ever certain, or infinitely costly.
"""

# ╔═╡ 94769f6a-50c6-5099-80af-8566c99deb74
"""
The chances `b` bits can name: the middles of `2^b` equal cells of [0, 1].

    chances(0) == [1/2]
    chances(1) == [1/4, 3/4]
    chances(2) == [1/8, 3/8, 5/8, 7/8]
"""
chances(b) = [(j + 1 / 2) / 2^b for j in 0:2^b-1]

# ╔═╡ 82136905-954c-4d96-90dc-dd6d300edb9b
md"""
Now we can name a predictor in three fields:

1. the order ``k``, in unary;
2. the precision ``b``, in unary as well;
3. the ``2^k`` chances, ``b`` bits each.
"""

# ╔═╡ 6f460fc0-1c46-5991-800a-2211606e0558
"A predictor of order `k` whose chances `θ`, one per context, have `b` bits each."
predictor(k, b, θ) = (order = k, precision = b, θ = collect(θ))

# ╔═╡ 1f353388-be7b-479f-9964-07be2100eb3a
md"""`program` writes a predictor down as bits:"""

# ╔═╡ c9a17ceb-a92d-51cc-b89d-506fa05fc193
"""
Predictor `ν`'s program, as a string of bits.

    program(predictor(0, 0, [1/2])) == "00"
    program(predictor(1, 2, [7/8, 1/8])) == "101101100"
"""
function program(ν)
    field(θ) = string(Int(θ * 2^ν.precision - 1 / 2); base = 2, pad = ν.precision)
    unary(ν.order) * unary(ν.precision) * join(field.(ν.θ))
end

# ╔═╡ bce49332-9e80-45bf-ab6a-fdc047cccfd8
md"""Here is an example program:"""

# ╔═╡ 0eac926e-f5ac-4854-b88d-1650031004fa
program(predictor(1, 2, [7/8, 1/8]))

# ╔═╡ bfa94811-1dd0-4bdc-81e7-3d71a0433faf
md"""
And it reads:

- `10`: order 1, one chance for each value of the last bit;
- `110`: precision 2;
- `11` and `00`: a 1 has the chance 7/8 after a 0, and 1/8 after a 1.

It predicts alternation, broken one time in eight.

The shortest program is `00`, the fair coin: order 0, precision 0, and one chance, ½, that takes no bits.

> This is a variant of Grünwald's very crude code for Markov chains ([tutorial](https://arxiv.org/abs/math/0406077), Example 2.8).
"""

# ╔═╡ d013c228-c565-4b44-8e2b-b62bf520e6d8
md"""
There are infinitely many programs the language can write, and our compute is finite, so we stop at some maximum order and precision:
"""

# ╔═╡ ed295beb-ac4d-558a-a933-fa512c4b14f1
MAX_ORDER = 2

# ╔═╡ ae285ef6-24d8-53e8-9212-7093a98b5996
MAX_PRECISION = 3

# ╔═╡ a586f028-7139-4df9-a28c-1a229a782c45
md"""The list of every program up to those limits:"""

# ╔═╡ fadc1764-1b80-5912-87ad-80040c7bb536
PREDICTORS = [
    predictor(k, b, θ)
        for k in 0:MAX_ORDER
        for b in 0:MAX_PRECISION
        for θ in Iterators.product(ntuple(_ -> chances(b), 2^k)...)
]

# ╔═╡ 20d72b26-89e2-53ff-8c33-15fed8880aa1
"`ν`'s chance of a 1 after each context of `MAX_ORDER` bits."
predictions(ν) = [ν.θ[c%2^ν.order+1] for c in 0:2^MAX_ORDER-1]

# ╔═╡ 4195c305-897c-59e5-a916-570c19b2a8ea
"The programs that describe the same predictor as `μ` in our list."
programs_for(μ) = [predictions(ν) == predictions(μ) for ν in PREDICTORS]

# ╔═╡ bfeacea6-0693-4a05-a48c-a371b2d9f198
md"""
## Making predictions

A predictor predicts from its context: the last ``k`` bits. Before the first bits there is no context yet, so missing bits count as zeros:
"""

# ╔═╡ ca52db12-954f-55d9-b71e-81a24e630d59
"""
The `k` bits before position `t`, read in order as a binary number. Bits before
the start count as zeros.

    x = [1, 0, 1, 1]
    context(x, 4, 2) == 0b01    # x[2], x[3] are 0, 1
    context(x, 3, 2) == 0b10    # x[1], x[2] are 1, 0
    context(x, 4, 1) == 1       # x[3] is 1
    context(x, 1, 2) == 0b00    # nothing seen yet
"""
context(x, t, k) = sum(((t - j ≥ 1 ? x[t-j] : 0) << (j - 1) for j in 1:k); init = 0)

# ╔═╡ 43e64bb7-9da7-4066-a124-3b999164c722
md"Its prediction is the chance of a 1 that its program names for that context:"

# ╔═╡ 300b0f43-de6d-5f1e-87c1-d45a71510915
"`ν`'s chance that bit `t` is a 1, having seen `x[1:t-1]`."
predict(ν, x, t) = ν.θ[context(x, t, ν.order)+1]

# ╔═╡ 1b2a7cb7-6569-4093-9806-bca1e665cbb3
md"Once the bit comes, a prediction is scored by its *likelihood*: the chance it gave that bit, ``p`` for a 1 and ``1 - p`` for a 0:"

# ╔═╡ a66a76b2-9f64-4ff0-8f5c-5ccf3eba8d9c
likelihood(p, bit) = bit == 1 ? p : 1 - p

# ╔═╡ d1af595e-6a01-45fd-8885-1068e5a2dcd4
md"A distribution is a code (notebook 03), so in bits a likelihood ``p`` becomes a cost: ``-\log_2 p`` bits to send the bit. Over a sequence, the costs add up:"

# ╔═╡ 14956fcc-0dd5-5271-8230-c434cbe22bfa
"The bits it takes to send `x[1:t]` with the predictions `ps`, for every `t`."
cost(ps, x) = cumsum(-log2.(likelihood.(ps, x)))

# ╔═╡ ff744d04-317b-42ce-9217-22a05535dc7f
md"""
Picture a sender and a receiver: the sender codes the bits with the predictor, using an arithmetic coder, and the receiver decodes them with the same predictor. What it costs depends on how well the predictor fits the bits.

Here are the intro's ten bits, sent with the fair coin and with the alternation predictor:
"""

# ╔═╡ c10841f5-ec8d-415c-84a6-216611f5074c
TEN_BITS = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# ╔═╡ afdc66a8-40d9-54c1-9a04-6862b90e2f4d
map((
    fair_coin = predictor(0, 0, [1/2]),
    alternation = predictor(1, 2, [7/8, 1/8]))) do ν
    last(cost([predict(ν, TEN_BITS, t) for t in eachindex(TEN_BITS)], TEN_BITS))
end

# ╔═╡ d0c35b35-3ffa-59cd-bf7d-8c551ab00845
md"""
The fair coin gives every bit ½, so each costs 1 bit, 10 bits in all: it saves nothing. The alternation predictor gives each of these bits 7/8, so each costs about 0.19, 1.93 bits in all (had a bit broken the pattern, it would have cost 3).

A predictor that fits the bits makes them cheap; one that doesn't makes them expensive.
"""

# ╔═╡ 70dcb2b6-b983-5952-82d1-00adc71fbb7d
md"""
### Sender and receiver

Let's see it in action (related code is in the appendix).

The sender starts with the interval ``[0, 1)`` and an empty code. Code digits name a cell the way a chance's bits do: each halves it, starting from ``[0, 1)``. For each bit, it:

1. predicts the bit from the bits before it;
2. cuts the interval at that prediction: the 0 side on the left, the 1 side on the right, each as wide as its likelihood;
3. keeps the side of the bit;
4. sends code digits while the interval fits inside one half of the code's cell, 0 for the left half and 1 for the right, each halving the cell. While the interval straddles the cell's middle, it sends nothing.

After the last bit, it adds the fewest digits that make the code's cell fit inside the interval.
"""

# ╔═╡ ab726d26-082c-4c83-b20c-063d45f438b3
md"""
## Bet on all the predictors

``\xi`` doesn't choose: it bets on every predictor on the list, and more on those with shorter programs. Flip a fair coin for each bit, and stop as soon as the bits spell a program. A program of ``\ell`` bits comes up with chance ``2^{-\ell}``, and that's its weight:
"""

# ╔═╡ 69c7a1fb-4ccf-5622-a0fb-fcedd41c9ab5
WEIGHTS = [exp2(-length(program(ν))) for ν in PREDICTORS]

# ╔═╡ 40f37aae-7096-519f-81e6-ac14fd65dec5
md"""
The bet, Legg's ``\xi`` (Def. 2.5.4), adds up every program's chance of ``x``, times its weight, so a predictor with several programs counts once for each:

```math
\xi(x) = \sum_\nu 2^{-\ell(\nu)}\, \nu(x).
```

That gives every sequence a chance, so ``\xi`` can predict too, though not from the last ``k`` bits: its chance of a 1 after ``x`` depends on all of ``x``:

```math
\xi(1 \mid x) = \frac{\xi(x1)}{\xi(x)} = \sum_\nu \frac{2^{-\ell(\nu)}\, \nu(x)}{\xi(x)}\; \nu(1 \mid x).
```

Each fraction is ``\nu``'s weight after ``x`` by Bayes' rule: its prior times the chance it gave ``x``, rescaled to sum to 1.

``\nu(x)`` is the product of the chances ``\nu`` gave each bit, so the rule can take one bit at a time, in one line:
"""

# ╔═╡ db7c766c-5609-519c-818d-d51b0b8407ce
"""
Bayes' rule: weigh each predictor by the likelihood its prediction `p` gives `bit`, then rescale.

    # the fair coin and alternation 3/4, after a 0
    # a 1 favours alternation
    update([1//2, 1//2], [1//2, 3//4], 1) == [2//5, 3//5]
    # a 0 favours the fair coin
    update([1//2, 1//2], [1//2, 3//4], 0) == [2//3, 1//3]

    # a bit both predict alike changes nothing
    update([1//4, 3//4], [1//2, 1//2], 1) == [1//4, 3//4]
"""
update(w, p, bit) = w .* likelihood.(p, bit) ./ sum(w .* likelihood.(p, bit))

# ╔═╡ e4b8adb2-fc28-4390-a563-25bd3ac64fd6
md"All predictions in one lookup table, one row per predictor and one column per context:"

# ╔═╡ 9bf87089-2b58-5388-b04a-5cdf527ebee0
PREDICTIONS = permutedims(reduce(hcat, predictions.(PREDICTORS)))

# ╔═╡ 0767132c-ef65-5b65-8d30-d7413b06e1ef
"Every predictor's chance that bit `t` of `x` is a 1: the table's column for that bit's context."
predictions_at(x, t) = view(PREDICTIONS, :, context(x, t, MAX_ORDER) + 1)

# ╔═╡ f29b720f-66bc-5730-84ec-83c897fef1bc
"ξ's weights along `x`: entry `t` is what it holds before bit `t`, and the last entry what it holds after them all."
function weights_along(x)
    start = WEIGHTS ./ sum(WEIGHTS)
    [[start]; accumulate((w, t) -> update(w, predictions_at(x, t), x[t]), eachindex(x); init = start)]
end

# ╔═╡ 9b1d9232-85e9-5be2-8cdd-1d3b3203d271
md"""
Back to the intro's ten bits, and four predictors to watch:
"""

# ╔═╡ e919d87b-6530-5def-a0cd-79ec01fd46b9
LEADERS = [
    "fair coin" => predictor(0, 0, [1/2]),
    "alternation, 3/4" => predictor(1, 1, [3/4, 1/4]),
    "alternation, 7/8" => predictor(1, 2, [7/8, 1/8]),
    "alternation, 15/16" => predictor(1, 3, [15/16, 1/16]),
]

# ╔═╡ 949dd8cc-dbe5-5200-a20e-f31f7cd8af51
md"""
Each bar below is ``\xi``'s whole budget, before the first bit and after each one: these four predictors' shares, each with its programs added up, and everything else on top:
"""

# ╔═╡ da958c7c-b6ee-5384-b106-de96d63dbf87
"ξ's chance that bit `t` of `x` is a 1, with the weights `w`: every predictor's chance, weighed."
mix(w, x, t) = sum(w .* predictions_at(x, t))

# ╔═╡ e2306bfc-c7d9-4ac8-b8c8-5cd6da4e8af6
md"""So ``\xi``'s prediction of a 1 after those ten bits:"""

# ╔═╡ eefa07df-af38-5faf-9e45-1876af0b0389
ANSWER = mix(last(weights_along(TEN_BITS)), TEN_BITS, 11)

# ╔═╡ e6bebb61-e47d-5363-8152-1ba43837e636
md"""
Ten bits of alternation moved the weight from the short programs, which predicted them poorly, to the alternators, which predicted them well. But the weight is spread over alternators of different sharpness, and the fair coin still keeps some: ten bits can't rule it out. No predictor on our list is certain, and neither is the bet: an average of their chances can never be surer than 15/16.
"""

# ╔═╡ 6c0f2e1a-722d-5a88-ba33-39c262e1e4f3
md"""
## Paying the price

Suppose the bits come from one predictor on the list, the truth ``\mu``, and ``\xi`` isn't told which. What does not knowing cost?

Start with what ``\xi`` believes: the weight it puts on the truth. Its prior weight is the chance the coin flips spell one of ``\mu``'s programs:

```math
W_\mu = \sum_{\nu \text{ describes } \mu} 2^{-\ell(\nu)}.
```

After the first ``t`` bits, ``x``, Bayes' rule has multiplied it by ``\mu``'s chance of ``x`` over ``\xi``'s. And a weight can't pass 1:

```math
W_\mu(x) = W_\mu\, \frac{\mu(x)}{\xi(x)} \le 1.
```

The ratio says two things:
1. ``\xi`` spreads its bets over every predictor, so it usually gives ``x`` less chance than the truth does: the ratio is how far it falls short.
2. The truth's weight grows by the same ratio: that's what ``\xi`` learns. A lesson costs what it teaches.
"""

# ╔═╡ 07693073-f6a5-58c2-9a99-5003ed13b613
price(μ) = -log2(sum(WEIGHTS[programs_for(μ)]))

# ╔═╡ 4e2fd55e-500b-4b26-a542-b1580f962635
md"""
What ``\xi`` has *paid* is how many more bits it spends on ``x`` than the truth does:

```math
\text{paid}_t = \log_2 \frac{\mu(x)}{\xi(x)} = -\log_2 \xi(x) - \bigl(-\log_2 \mu(x)\bigr).
```

What it still *owes* is the bits it would still take to name ``\mu``, with what it has learned. The two always add up to the price:

```math
\text{paid}_t + \text{owed}_t = \text{price}.
```

``\xi`` is never told ``\mu``'s name, but it pays the price anyway, in installments, through the data.

> Elsewhere, what ``\xi`` has paid is called its *regret* against ``\mu``; Legg never names it.
"""

# ╔═╡ e711492e-b1ef-56f4-86dd-6168f3b92a23
md"""
The installments come one per bit. Each bit, Bayes' rule multiplies the truth's weight by ``\mu``'s chance of the bit over ``\xi``'s, so each bit moves the same amount from owed to paid:

```math
\log_2 \frac{\mu(x_t \mid x_{<t})}{\xi(x_t \mid x_{<t})} .
```

When the truth saw the bit coming better than ``\xi`` did, ``\xi`` pays, and learns. When ``\xi`` saw it better, the bit was misleading: ``\xi`` gains bits, and loses certainty.
"""

# ╔═╡ 7c7ee428-0693-5196-9f70-c16bb7178624
PAID_UPFRONT = -log2(sum(WEIGHTS))

# ╔═╡ bccd5d9c-0d2b-5cfe-bb9e-354f6dc49c40
md"""
### Claim 1: in bits

What ``\xi`` owes can't fall below 0, so on every sequence,

```math
\text{paid}_t \le \text{price}.
```

It caps what ``\xi`` pays, not what it gets back: misleading bits can take what it has paid below 0, putting it ahead of the truth, for now.

Nothing here needs ``\mu`` to be the truth. Against any predictor ``\nu`` on the list, on any sequence, ``\xi`` needs at most the price of ``\nu`` more bits than ``\nu`` does, so it never needs more than notebook 03's two-part code: name ``\nu``, then send the bits with its help. ``\xi`` does nearly as well as the best predictor on the list, whichever that turns out to be, without ever choosing one.

Betting on whichever predictor leads so far has no such guarantee. Give it two coins, at 1/4 and 3/4, and bits that always go against the leader (they come out alternating): it pays 2 bits a bit, while either coin alone pays about 1.2.

> Legg's *dominance* (p. 36), ``2^{K(\mu)}\, \xi(x) \ge \mu(x)``, is the same bound in probability, with ``K(\mu)`` in place of the price.
"""

# ╔═╡ f0b0efeb-cce2-5461-8f7c-f20cce5ba8c1
md"""
### Claim 2: in predictions

Solomonoff's convergence theorem (Legg's Theorem 2.7.1) states the promise in predictions:

```math
\sum_{t=1}^{\infty} S_t \le \frac{\ln 2}{2} K(\mu),
```

where ``S_t`` is the squared gap between ``\xi``'s chance of a 1 and ``\mu``'s at step ``t``, averaged over every history ``\mu`` could produce.

Infinitely many squared gaps, and a finite total: the gaps must shrink to 0. Whatever ``\mu`` is, ``\xi``'s predictions converge to ``\mu``'s, without ``\xi`` ever having to single out ``\mu``'s program.

Unlike Claim 1, this one holds only on average. Averaged over every history, each bit costs ``\xi`` something, and the further its prediction is from the truth's, the more: at least ``\frac{2}{\ln 2}`` bits times the squared gap. Claim 1 caps what ``\xi`` pays on every sequence, so on average too: all of it comes out of the price, and

```math
\sum_t S_t \le \frac{\ln 2}{2}\, \text{price}:
```

Solomonoff's theorem, with the price in place of ``K(\mu)``. On a single sequence, it can fail: a misleading bit refunds what ``\xi`` has paid, but nothing refunds a squared gap.

> The two steps have names: a bit's average cost is a KL divergence, never negative (notebook 04), and Pinsker's inequality bounds it below by the squared gap. Legg's sketch of the proof (p. 39) works with the same sum, in nats.
"""

# ╔═╡ de44bd39-1e6e-52b3-b187-2d82e8bea8fb
md"""
## Testing the claims

We test both claims on four sources, each a predictor from the list, with programs from 2 to 19 bits:
"""

# ╔═╡ ddfbda75-1b44-5501-a9dc-8b421b87771a
SOURCES = [
    (name = "fair coin", μ = predictor(0, 0, [1 / 2])),
    (name = "coin at 1/8", μ = predictor(0, 2, [1 / 8])),
    (name = "alternating, flips at 1/8", μ = predictor(1, 2, [7 / 8, 1 / 8])),
    (name = "order 2", μ = predictor(2, 3, [1 / 16, 11 / 16, 13 / 16, 7 / 16])),
]

# ╔═╡ b1c2acb1-878c-5e54-b0b4-bae24b84c0a5
md"""
A test draws ``N`` bits from a source, bets on them, and records what ``\xi`` has paid, what it owes, and the squared gaps, bit by bit:
"""

# ╔═╡ 6608cd0a-e892-5ca5-851e-e88dbb3df674
N = 3000

# ╔═╡ a1767280-2cc6-55d8-ba38-498a4f923a76
"`n` bits drawn from `μ`, each a 1 with the chance `μ` gives it."
sample(μ, n, rng) =
    foldl((x, t) -> push!(x, rand(rng) < predict(μ, x, t) ? 1 : 0), 1:n; init = Int[])

# ╔═╡ 364bb9bb-dbb8-5bcf-a639-a3aa958fca4c
"""
`N` bits drawn from `μ` with `seed`, and ξ's bet on them, bit by bit: what it has
paid, what it owes, and its running Σ(ξ − μ)².
"""
function trial(μ, seed)
    x = sample(μ, N, Xoshiro(seed))
    w = weights_along(x)
    ξs = [mix(w[t], x, t) for t in 1:N]
    μs = [predict(μ, x, t) for t in 1:N]
    truth = programs_for(μ)
    (; paid = PAID_UPFRONT .+ cost(ξs, x) .- cost(μs, x),
       owed = [-log2(sum(w[t+1][truth])) for t in 1:N],
       S = cumsum((ξs .- μs) .^ 2))
end

# ╔═╡ 120b372d-e133-5d3f-9742-fb115ea82ad7
md"""
### In bits, on every sequence

One sequence from each source first:
"""

# ╔═╡ ddc467e0-9956-567f-a678-b614a402e47e
EXAMPLES = [trial(s.μ, 1) for s in SOURCES]

# ╔═╡ ebe14752-9b79-5ce1-b2e6-64cff35c45bb
md"""
``\xi`` pays the price early, while it finds out which predictor is talking, and from then on it codes at the truth's own rate. Stack what it owes on what it has paid, and the two parts of the price show:
"""

# ╔═╡ 94fc4917-ac38-51c9-892d-1ac751e4c3a1
RUNS = 1000

# ╔═╡ 1ee88810-ebd2-5f59-986a-8da7523aad6e
md"""
Drawing from a source is the easy test of a claim about every sequence. The hard test builds a sequence against ``\xi``: each bit is the one ``\xi`` gives less chance.
"""

# ╔═╡ cc976cec-b99f-53ac-a43e-b36d2e329785
"`n` bits built against ξ: each is the bit ξ gives less chance, or a 0 on a tie."
against(n) =
    first(
        foldl(1:n; init = (Int[], WEIGHTS ./ sum(WEIGHTS))) do (x, w), t
        push!(x, mix(w, x, t) < 1 / 2 ? 1 : 0)
        (x, update(w, predictions_at(x, t), x[t]))
    end)

# ╔═╡ 78797b07-984c-512d-8a72-ced0da21df6a
md"""
## The full picture

Our list stops at order 2 and precision 3, and both claims need the truth to be on it. What if it isn't: a coin at 0.3, say? No program of ours names it, at any precision. Here is what ``\xi`` pays for its bits, betting on coins alone, with precision capped at 3, as on our list, and at 16, more than a million bits can use:
"""

# ╔═╡ 20c2ffe7-f0d1-56a9-9ec9-13b0bc345802
"""
What ξ has paid after `n` bits from a coin at 0.3, with ones at exactly 3 in 10, betting on the coins of
every precision up to `B`. Their weighted chances are summed in log₂, the largest factored out so the rest don't underflow.
"""
function paid_on_coin(n, B)
    ones = round(Int, 3n / 10)
    loglik(θ) = ones * log2(θ) + (n - ones) * log2(1 - θ)
    terms = [loglik(θ) - (2b + 2) for b in 0:B for θ in chances(b)]    # a coin's program: 0, unary(b), b bits
    top = maximum(terms)
    loglik(0.3) - (top + log2(sum(exp2.(terms .- top))))
end

# ╔═╡ cd539602-f958-50ee-9819-7f9fea6c7202
md"""
The two agree until a few thousand bits, when the data starts to tell 0.3 from 5/16, the nearest chance at precision 3. From then on, the capped ``\xi`` pays a little on every bit, and runs off the chart.

The other keeps moving to finer chances. Each halving of the cell costs two bits of program, one for the chance and one for the unary precision, and needs four times the data to pay off: about a bit per doubling of the data, ``\log_2 n``. The chance's half is notebook 03's COMP, the log of how many coins the data can tell apart. The precision's half is the price of not knowing in advance how much precision the data will justify.

That limit is our language's, not Solomonoff's. A coin at 0.3 is computable, so Legg's list has it, with a short program, and his ``\xi`` pays a constant price for it. Only a coin whose chance has no short program makes his ``\xi`` keep paying, about ``\frac{1}{2} \log_2 n``: COMP again.
"""

# ╔═╡ 9237f845-61c5-5b63-9363-a1dac0a6e75a
md"""
While we were writing this notebook, 3Blue1Brown published [*This algorithm changed how I think about compression*](https://www.youtube.com/watch?v=qbsb9DpdZ6s), on arithmetic codes: the coder our sender and receiver share. It turns a whole message into one number in ``[0, 1)``, pinned down to finer and finer precision as the symbols come in, each narrowing the interval to the share its chance gives it.
"""

# ╔═╡ baf84fac-12da-50b2-923b-61883ba2d5e5
html"""
<iframe src="https://www.youtube-nocookie.com/embed/qbsb9DpdZ6s" title="This algorithm changed how I think about compression (3Blue1Brown, 2026)"
        style="width: 100%; aspect-ratio: 16 / 9; border: 0;"
        allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
"""

# ╔═╡ bee35bd4-698d-563c-879f-2cf80429f8e8
md"""
It's almost too perfect a picture of this notebook. Every cost here, ``-\log_2 p``, is the bits it takes to narrow the interval by a factor of ``p``, and every bit of our precision field halves a cell.

> The video even links Sutskever's talk from [notebook 01](https://zxygentoo.github.io/the-ninety-percent-list/01.An.Observation.on.Generalization.html), and prequential coding from [notebook 03](https://zxygentoo.github.io/the-ninety-percent-list/03.A.Tutorial.Introduction.to.the.Minimum.Description.Length.Principle.html).

This notebook stops at §2.7, Solomonoff induction: the prediction half of Legg's chapter 2. It's a deliberate choice. The rest of the chapter makes ``\xi`` act. AIXI (§2.10) takes the actions with the most reward expected under ``\xi``, and Legg's universal intelligence (chapter 4) scores an agent by its reward in every computable environment, weighed by ``2^{-K}``. Both are built on ``\xi``, which is why we stopped here: prediction is the more fundamental idea. We may come back to the agent half in a future update of this notebook.
"""

# ╔═╡ 71391962-660d-5d7f-86ab-c9319fa462ad
md"""
## Appendix: helpers

The arithmetic coder from Making predictions, number formatting, a threaded map, and the chart style.
"""

# ╔═╡ c1101a91-3b22-5788-a3b1-1c817c8f659e
# The interval [0, 1), as (low, high), in exact fractions, so nothing is lost to rounding.
WHOLE = (big(0) // 1, big(1) // 1)

# ╔═╡ 491b798f-57f6-5219-a476-6c233c759cae
"Where the prediction `p` cuts the interval: the 0 side is to its left, the 1 side to its right."
cut((low, high), p) = low + (high - low) * (1 - Rational{BigInt}(p))

# ╔═╡ 3ccd84b9-df22-566a-bbfc-a47705401a44
"The interval after each bit of `x`, sent with `ν`: the side of that bit, where its prediction cuts."
intervals(ν, x) = accumulate(eachindex(x); init = WHOLE) do (low, high), t
    at = cut((low, high), predict(ν, x, t))
    x[t] == 0 ? (low, at) : (at, high)
end

# ╔═╡ 202c4c72-88aa-584b-bc57-762289b84b83
"The cell a run of code bits names: every number whose binary digits start with them."
function cell(code)
    low = isempty(code) ? big(0) // 1 : parse(BigInt, code; base = 2) // big(2)^length(code)
    (low, low + big(1) // big(2)^length(code))
end

# ╔═╡ 4cd13998-ab74-5b2d-b10f-8d98e48becc4
"The digits every number in the interval shares: the longest run whose cell holds all of it."
function settled((low, high))
    digits(m) = m == 0 ? "" : string(floor(BigInt, low * big(2)^m); base = 2, pad = m)
    m = 0
    while high <= last(cell(digits(m + 1)))
        m += 1
    end
    digits(m)
end

# ╔═╡ 68c08ab6-3f9f-5302-86e0-9fb996cb4a5a
"The shortest codeword that pins the interval down: `m` bits naming `j / 2^m`, with every continuation inside."
function codeword((low, high))
    for m in Iterators.countfrom(1)
        j = ceil(BigInt, low * big(2)^m)
        (j + 1) // big(2)^m <= high && return string(j; base = 2, pad = m)
    end
end

# ╔═╡ 8abab510-3437-5815-aaa1-75362cd707fa
"The code sent by the end of each bit of `x`: the digits its interval has settled, and after the last bit, the codeword."
sent(ν, x) = let after = intervals(ν, x)
    [settled.(after[1:end-1]); codeword(last(after))]
end

# ╔═╡ 3b3a36da-42e1-5233-b2a7-1d75cb8a575c
CODES = map((
    fair_coin = predictor(0, 0, [1/2]),
    alternation = predictor(1, 2, [7/8, 1/8]))) do ν
    codeword(last(intervals(ν, TEN_BITS)))
end

# ╔═╡ 71857ff0-130b-5a3d-807f-a0ed8c5694d8
"Which side of the cut `at` a cell lies on: 0 or 1, or `nothing` while the cut still splits it."
side((low, high), at) = low >= at ? 1 : high <= at ? 0 : nothing

# ╔═╡ 4204233e-773b-51a2-a9d3-9394b4d98f97
"""
Read `n` bits back from `code` with `ν`. Before each bit, predict and cut, then read code bits only until the cell
they name lies on one side of the cut: that side is the bit. Returns the bits, and the code read by each one.
"""
function decode(ν, code, n)
    steps = accumulate(1:n; init = (Int[], WHOLE, "")) do (x, (low, high), read), t
        at = cut((low, high), predict(ν, x, t))
        while isnothing(side(cell(read), at))
            read = code[1:length(read)+1]
        end
        bit = side(cell(read), at)
        ([x; bit], bit == 0 ? (low, at) : (at, high), read)
    end
    (bits = first(last(steps)), read = last.(steps))
end

# ╔═╡ 704d8f96-cd78-5ddf-9a90-30cb68d4dc6a
"`x` with `d` decimals."
fixed(x, d = 1) = Printf.format(Printf.Format("%.$(d)f"), x)

# ╔═╡ 28d1adda-8b0c-43c7-a18b-a89d7cc5028d
md"""
No program is the start of another, so the flips spell at most one program, and the weights add up to at most 1. Over the whole language they add up to exactly 1: each unary field ends at its first 0, which comes with chance 1/2 + 1/4 + 1/8 + ⋯ = 1. On our list, the order's 0 must come within 3 flips, with chance 7/8, and the precision's within 4, with chance 15/16, so the weights sum to 7/8 × 15/16 = $(fixed(sum(WEIGHTS), 2)): the rest is the chance of spelling a program the language can write but the list leaves out, of order 3 and up or precision 4 and up.

> Legg's prior (Def. 2.5.3) is the same, with one weight per predictor: ``2^{-K}``.
"""

# ╔═╡ ca4ece31-c355-5b7b-ba80-05518918610d
md"""
Counted in bits, the product becomes a sum. Take ``-\log_2`` of both sides:

```math
\underbrace{-\log_2 W_\mu(x)}_{\text{owed}_t} \;=\; \underbrace{-\log_2 W_\mu}_{\text{price}} \;-\; \underbrace{\log_2 \frac{\mu(x)}{\xi(x)}}_{\text{paid}_t}.
```

The *price* is the bits it takes to name ``\mu``, with the weights as the code. It's at most the length of ``\mu``'s shortest program, our ``K(\mu)``, and less when ``\mu`` has several (with Legg's one weight per predictor, it would be exactly ``K(\mu)``). The fair coin has three, `00`, `100` and `1100`, whose weights add up to 7/16, so its price is $(fixed(price(predictor(0, 0, [1/2])), 2)) bits, though its shortest program has 2.
"""

# ╔═╡ 104348f3-a806-593a-9ba0-9f4d8299dc7c
md"""
For our list, before the first bit, ``\xi`` has already paid $(fixed(PAID_UPFRONT, 2)) bits: ``-\log_2`` of the weights' total, $(fixed(sum(WEIGHTS), 2)). That's its stake on the programs left off the list, which it never wins back, so it starts out owing the rest of the price.
"""

# ╔═╡ a47f5125-3bd6-59a7-94c6-ed0aeb7fd301
let rows = map(SOURCES) do (; name, μ)
        "| $name | `$(program(μ))` | $(length(program(μ))) | $(count(programs_for(μ))) | $(fixed(price(μ), 2)) |"
    end
    Markdown.parse("""
    | source | program | ``\\ell`` | programs for it | price |
    |:--|:--|--:|--:|--:|
    $(join(rows, "\n"))

    Most sources have one program that dominates, so their price is about its length.
    """)
end

# ╔═╡ 90501124-d3c4-5649-b5e5-0dbafbfd08e9
"An interval `(low, high)`, as `[low, high)` to 3 decimals."
span((low, high)) = "[$(fixed(Float64(low), 3)), $(fixed(Float64(high), 3)))"

# ╔═╡ 981c8c99-4fd2-5ba3-b745-c986a106bb46
let
    function table(ν)
        x = TEN_BITS
        after = intervals(ν, x)
        before = [WHOLE; after[1:end-1]]
        code = sent(ν, x)
        rows = map(eachindex(x)) do t
            "| $t | $(fixed(predict(ν, x, t), 3)) | $(span(before[t])) | `$(x[t])` | $(span(after[t])) | $(isempty(code[t]) ? "—" : "`$(code[t])`") |"
        end
        """
        | bit | chance of a 1 | interval before | coded | interval after | code so far |
        |:-:|:-:|:--|:-:|:--|:--|
        $(join(rows, "\n"))
        """
    end
    Markdown.parse("""
    The fair coin:

    $(table(predictor(0, 0, [1/2])))

    Alternation:

    $(table(predictor(1, 2, [7/8, 1/8])))
    """)
end

# ╔═╡ 37b6fd25-1dba-55a2-8d41-aa7a689e52d8
let alternation = predictor(1, 2, [7/8, 1/8])
    (low, high) = last(intervals(alternation, TEN_BITS))
    md"""
    The fair coin halves the interval at every bit, so each bit sends one digit, and the code is the ten bits themselves. Alternation keeps 7/8 of the interval at every bit, so after ten bits it is still $(fixed(Float64(high - low), 2)) wide, and it has straddled ½, the middle of ``[0, 1)``, the whole way, so it sent nothing until the end. At the end, $(Markdown.Code(CODES.alternation)) is the shortest run whose cell, $(span(cell(CODES.alternation))), fits inside. That's $(length(CODES.alternation)) bits, about a bit over the cost: the code comes in whole bits, and a whole cell must fit, but that never adds more than 2 (notebook 03).

    The receiver knows the predictor and that ten bits are coming, but not the bits. It starts with ``[0, 1)`` and no code read. For each bit, it:

    1. predicts the bit from the bits decoded so far;
    2. cuts the interval at that prediction, as the sender did;
    3. reads code digits while the cut falls inside the cell of the code read so far, since the bit could still go either way; each digit halves the cell;
    4. decodes the bit of the side the cell lies on, and keeps that side.
    """
end

# ╔═╡ 1f86c813-7a0d-577a-8531-4f98ea6d426d
let
    function table(ν, code)
        (; bits, read) = decode(ν, code, length(TEN_BITS))
        after = intervals(ν, bits)
        before = [WHOLE; after[1:end-1]]
        rows = map(eachindex(bits)) do t
            "| $t | `$(read[t])` | $(span(cell(read[t]))) | $(fixed(predict(ν, bits, t), 3)) | $(span(before[t])) | `$(bits[t])` | $(span(after[t])) |"
        end
        """
        | bit | code read | its cell | chance of a 1 | interval before | decoded | interval after |
        |:-:|:--|:--|:-:|:--|:-:|:--|
        $(join(rows, "\n"))
        """
    end
    fair, alternation = predictor(0, 0, [1/2]), predictor(1, 2, [7/8, 1/8])
    back = all(decode(ν, code, length(TEN_BITS)).bits == TEN_BITS
               for (ν, code) in ((fair, CODES.fair_coin), (alternation, CODES.alternation)))
    Markdown.parse("""
    The fair coin, from `$(CODES.fair_coin)`:

    $(table(fair, CODES.fair_coin))

    Alternation, from `$(CODES.alternation)`:

    $(table(alternation, CODES.alternation))

    In every row, the cell of the code read sits inside the interval after: that's why the bit comes out right. The fair coin reads one code bit per bit. Alternation reads its three at bits 1, 2 and 6, and decodes the other seven from the cell it already holds. $(back ? "Both decode to" : "They don't decode to") the ten bits that were sent.

    Each cut needs the bit before it: the receiver can't predict bit 2 until it has decoded bit 1. That's why the two must share the predictor, and why the bits come out in order.
    """)
end

# ╔═╡ 08c82a03-9fa4-5b9d-a374-b5563d56bbf1
"A whole number, with commas between the thousands."
commas(n::Integer) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => ",")

# ╔═╡ e223181b-546d-526a-8f1d-a1f959bc3aeb
md"""
The list holds $(commas(length(PREDICTORS))) programs.

Some programs describe the same predictor: the fair coin is `00`, but also `100` and `1100`. We find those by comparing what programs predict in every context:
"""

# ╔═╡ 97700adc-8d57-5055-b652-e03984aa6501
let gap = maximum(maximum(abs.(e.paid .+ e.owed .- price(s.μ))) for (s, e) in zip(SOURCES, EXAMPLES))
    md"""
    On all four sequences, paid + owed matches the price to within $(@sprintf("%.0e", gap)) bits, at every step.

    Now $(commas(RUNS)) sequences from each source:
    """
end

# ╔═╡ 5d96ecad-cb6a-5d5b-96d6-c8d68c1c9656
let x = against(N), w = weights_along(x)
    ξs = [mix(w[t], x, t) for t in 1:N]
    paid = PAID_UPFRONT + last(cost(ξs, x)) - N
    settled = findlast(p -> abs(p - 1 / 2) ≥ 0.01, ξs) + 1
    md"""
    They start $(Markdown.Code(join(x[1:30]))). Each one costs ``\xi`` at least 1 bit, so it can never pull ahead of the fair coin, which pays exactly 1 a bit. Yet over $(commas(N)) bits, it pays only $(fixed(paid, 2)) bits more: the fair coin's whole price, and no more. From bit $(settled) on, its predictions stay within 0.01 of ½. The bits built to beat ``\xi`` end up looking like coin flips, and all they can take is the price.
    """
end

# ╔═╡ 09c99de7-c2fa-550b-b2c3-7c1e6c82aae5
md"""
### In predictions, on average

``S_t`` is an average over every sequence ``\mu`` could produce, so we estimate it from the same $(commas(RUNS)) sequences per source, ± 2 standard errors. The bound is ``\frac{\ln 2}{2}`` times the price:
"""

# ╔═╡ 72ed5461-7de0-5be6-9e20-b2070b71b343
"`map`, with each call on its own task, so the calls share the threads."
tmap(f, xs) = fetch.([Threads.@spawn f(x) for x in xs])

# ╔═╡ 0314eace-722a-5c30-9edc-4ee05acf1020
# `RUNS` sequences per source, one column per sequence.
TRIALS = map(SOURCES) do (; μ)
    runs = tmap(seed -> let o = trial(μ, seed); (o.paid, o.S) end, 1:RUNS)
    (paid = reduce(hcat, first.(runs)), S = reduce(hcat, last.(runs)))
end

# ╔═╡ 2f6b3cb7-1efb-5826-b335-c532f8d011f1
let over = sum(count(>(price(s.μ) + 1e-9), maximum(e.paid; dims = 1)) for (s, e) in zip(SOURCES, TRIALS)),
    lowest = minimum(minimum(e.paid) for e in TRIALS)
    md"""
    ``\xi`` pays more than the price on $(over == 0 ? "none" : over) of the $(commas(length(SOURCES) * RUNS)) sequences. But on some, what it has paid falls far below 0, as low as $(fixed(lowest)) bits: when the bits happen to look like another predictor's, ``\xi`` follows it and beats the truth.

    The price limits only how much ``\xi`` can lose. What it can win is limited too, by chance: on a sequence from ``\mu``, the chance that what it has paid ever falls ``c`` bits below 0 is at most ``2^{-c}``. Across $(commas(length(SOURCES) * RUNS)) sequences, that makes a low much under $(fixed(-log2(length(SOURCES) * RUNS), 0)) bits unlikely.

    > Averaged over ``\mu``'s sequences of a given length, ``\xi(x)/\mu(x)`` is ``\sum_x \xi(x)``, at most 1. So it reaches ``2^c`` with chance at most ``2^{-c}`` (Markov's inequality), and by Ville's inequality, at any length at all.
    """
end

# ╔═╡ 790985b6-c3c8-564f-8216-c443092cd97b
let
    stats = map(SOURCES, TRIALS) do (; name, μ), (; S)
        final = S[end, :]
        mean = sum(final) / RUNS
        se = sqrt(sum(abs2, final .- mean) / (RUNS - 1) / RUNS)
        bound = log(2) / 2 * price(μ)
        (; name, mean, se, bound, above = count(>(bound), final))
    end
    # what ξ still owes after N bits, on average, is the price less what it has paid
    left = maximum(log(2) / 2 * (price(s.μ) - sum(t.paid[end, :]) / RUNS) for (s, t) in zip(SOURCES, TRIALS))
    rows = ["| $(s.name) | $(fixed(s.mean, 3)) ± $(fixed(2s.se, 3)) | $(fixed(s.bound, 3)) | $(s.above) |" for s in stats]
    above = [s.above for s in stats]
    Markdown.parse("""
    | source | mean Σ (ξ − μ)², ± 2 s.e. | bound | sequences above it |
    |:--|--:|--:|--:|
    $(join(rows, "\n"))

    Every mean is under its bound, while $(minimum(above)) to $(maximum(above)) of each source's $(commas(RUNS)) sequences end above it: the bound holds on average, not sequence by sequence.

    We stopped at bit $(commas(N)), but the sums run forever. What they leave out comes out of what ``\\xi`` still owes, as Claim 2's sum came out of the price, so on average it's no more than ``\\frac{\\ln 2}{2}`` times that: $(fixed(left, 3)) at the most, for any source.
    """)
end

# ╔═╡ 33708e56-08e2-5be5-b599-2d26008dbf7a
# Sequences drawn per panel.
SHOWN = 100

# ╔═╡ e0e939af-fa4b-5f1c-82bf-660dc87bf2d8
VIZ = (surface = "#fcfcfb", text = "#0b0b0b", text2 = "#52514e", grid = "#e4e3de")

# ╔═╡ da9c47c3-26ea-5e4e-9755-8df333ff3e1e
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# ╔═╡ e715cea6-339d-5d77-8109-d8cfdf9d2ccd
THEME = Theme(
    fontsize = 13, backgroundcolor = VIZ.surface, textcolor = VIZ.text,
    Axis = (backgroundcolor = VIZ.surface, xgridcolor = VIZ.grid, ygridcolor = VIZ.grid,
            topspinevisible = false, rightspinevisible = false, leftspinecolor = VIZ.text2, bottomspinecolor = VIZ.text2,
            xtickcolor = VIZ.text2, ytickcolor = VIZ.text2, xticklabelcolor = VIZ.text2, yticklabelcolor = VIZ.text2,
            titlealign = :left, titlefont = :regular, titlesize = 14),
    Legend = (framevisible = false, backgroundcolor = VIZ.surface),
);

# ╔═╡ 634f41b9-e54e-55a8-a517-67ab466e5494
let path = weights_along(TEN_BITS)
    shares = map(path) do w
        leaders = [sum(w[programs_for(ν)]) for (_, ν) in LEADERS]
        [leaders; 1 - sum(leaders)]
    end
    labels = [first.(LEADERS); "everything else"]
    colors = [SERIES_COLORS; "#c8c7c0"]
    groups = repeat(eachindex(labels); outer = length(path))
    with_theme(THEME) do
        fig = Figure(size = (760, 440))
        ax = Axis(fig[1, 1]; xlabel = "bits seen", ylabel = "share of the weight", xticks = 0:length(TEN_BITS),
                  title = "Ten bits of alternation move the budget from the fair coin to the alternators")
        barplot!(ax, repeat(0:length(TEN_BITS); inner = length(labels)), reduce(vcat, shares);
                 stack = groups, color = colors[groups], gap = 0.25)
        ylims!(ax, 0, 1)
        Legend(fig[2, 1], [PolyElement(color = c) for c in colors], labels; orientation = :horizontal)
        fig
    end
end

# ╔═╡ c0e0be39-665b-5fce-92e5-00b755be7543
with_theme(THEME) do
    fig = Figure(size = (760, 460))
    ax = Axis(fig[1, 1]; xscale = log10, xlabel = "bits seen", ylabel = "paid (bits)",
              title = "What ξ pays climbs, then flattens at the price (dashed)")
    for ((; name, μ), (; paid), color) in zip(SOURCES, EXAMPLES, SERIES_COLORS)
        hlines!(ax, price(μ); color, linestyle = :dash, linewidth = 1)
        lines!(ax, 1:N, paid; color, linewidth = 2, label = name)
    end
    Legend(fig[2, 1], ax; orientation = :horizontal)
    fig
end

# ╔═╡ 8b85a889-8683-515d-b641-79368ec90f80
with_theme(THEME) do
    fig = Figure(size = (760, 580))
    for (i, ((; name, μ), (; paid, owed))) in enumerate(zip(SOURCES, EXAMPLES))
        ax = Axis(fig[fldmod1(i, 2)...]; xscale = log10, title = "$name: price $(fixed(price(μ), 2))",
                  xlabel = i > 2 ? "bits seen" : "", ylabel = isodd(i) ? "bits" : "")
        band!(ax, 1:N, zeros(N), paid; color = (SERIES_COLORS[1], 0.6))
        band!(ax, 1:N, paid, paid .+ owed; color = (SERIES_COLORS[4], 0.4))
        hlines!(ax, price(μ); color = VIZ.text, linestyle = :dash, linewidth = 1)
    end
    Legend(fig[3, 1:2],
           [PolyElement(color = (SERIES_COLORS[1], 0.6)), PolyElement(color = (SERIES_COLORS[4], 0.4))],
           ["paid", "owed"]; orientation = :horizontal)
    fig
end

# ╔═╡ 77f6c10b-7c89-5e29-8879-76f2a6f66baf
with_theme(THEME) do
    fig = Figure(size = (760, 520))
    for (i, ((; name, μ), (; paid), color)) in enumerate(zip(SOURCES, TRIALS, SERIES_COLORS))
        ax = Axis(fig[fldmod1(i, 2)...]; xscale = log10, title = name,
                  xlabel = i > 2 ? "bits seen" : "", ylabel = isodd(i) ? "paid (bits)" : "")
        for j in 1:SHOWN
            lines!(ax, 1:N, paid[:, j]; color = (color, 0.15))
        end
        hlines!(ax, price(μ); color = VIZ.text, linestyle = :dash, linewidth = 1)
    end
    Label(fig[0, :], "$SHOWN sequences per source; the price is dashed"; halign = :left, color = VIZ.text2)
    fig
end

# ╔═╡ e9341e50-d24e-50c3-8a63-c6a0d5d11925
with_theme(THEME) do
    fig = Figure(size = (760, 520))
    for (i, ((; name, μ), (; S), color)) in enumerate(zip(SOURCES, TRIALS, SERIES_COLORS))
        bound = log(2) / 2 * price(μ)
        ax = Axis(fig[fldmod1(i, 2)...]; xscale = log10, title = name,
                  xlabel = i > 2 ? "bits seen" : "", ylabel = isodd(i) ? "Σ (ξ − μ)²" : "")
        for j in 1:SHOWN
            lines!(ax, 1:N, S[:, j]; color = (color, 0.15))
        end
        lines!(ax, 1:N, vec(sum(S; dims = 2)) / RUNS; color = VIZ.text, linewidth = 2)
        hlines!(ax, bound; color = VIZ.text, linestyle = :dash, linewidth = 1)
        ylims!(ax, 0, 3bound)
    end
    Label(fig[0, :], "$SHOWN sequences per source; the mean of all $(commas(RUNS)) is black, the bound dashed";
          halign = :left, color = VIZ.text2)
    fig
end

# ╔═╡ 97a0f4d3-d7df-58b2-871a-0a59c0d0c27b
let ns = round.(Int, exp10.(1:0.1:6))
    with_theme(THEME) do
        fig = Figure(size = (760, 460))
        ax = Axis(fig[1, 1]; xscale = log10, xlabel = "bits seen", ylabel = "paid (bits)",
                  title = "A coin at 0.3: capped precision pays on every bit, unlimited about a bit per doubling")
        lines!(ax, ns, paid_on_coin.(ns, MAX_PRECISION); color = SERIES_COLORS[1], linewidth = 2,
               label = "precision up to $MAX_PRECISION")
        lines!(ax, ns, paid_on_coin.(ns, 16); color = SERIES_COLORS[2], linewidth = 2, label = "precision up to 16")
        lines!(ax, ns, log2.(ns); color = VIZ.text, linestyle = :dash, linewidth = 1, label = "log₂ n")
        lines!(ax, ns, log2.(ns) / 2; color = VIZ.text2, linestyle = :dot, linewidth = 1.5, label = "½ log₂ n, COMP's rate")
        ylims!(ax, 0, 25)
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
# ╟─942b639c-bf53-5823-8d4f-de5881d3c04f
# ╠═05725c63-8e71-54fb-a753-1d7886e63d3a
# ╟─b10a5215-4d41-5ccd-a6ad-eaec38622508
# ╟─6b0eadbf-0ef0-432a-af32-5e3d4519ce69
# ╠═26956974-c66e-5449-8757-79b8acacbc04
# ╟─2b8418e1-76c6-51d9-a691-942ab98641e3
# ╠═94769f6a-50c6-5099-80af-8566c99deb74
# ╟─82136905-954c-4d96-90dc-dd6d300edb9b
# ╠═6f460fc0-1c46-5991-800a-2211606e0558
# ╟─1f353388-be7b-479f-9964-07be2100eb3a
# ╠═c9a17ceb-a92d-51cc-b89d-506fa05fc193
# ╟─bce49332-9e80-45bf-ab6a-fdc047cccfd8
# ╠═0eac926e-f5ac-4854-b88d-1650031004fa
# ╟─bfa94811-1dd0-4bdc-81e7-3d71a0433faf
# ╟─d013c228-c565-4b44-8e2b-b62bf520e6d8
# ╠═ed295beb-ac4d-558a-a933-fa512c4b14f1
# ╠═ae285ef6-24d8-53e8-9212-7093a98b5996
# ╟─a586f028-7139-4df9-a28c-1a229a782c45
# ╠═fadc1764-1b80-5912-87ad-80040c7bb536
# ╟─e223181b-546d-526a-8f1d-a1f959bc3aeb
# ╠═20d72b26-89e2-53ff-8c33-15fed8880aa1
# ╠═4195c305-897c-59e5-a916-570c19b2a8ea
# ╟─bfeacea6-0693-4a05-a48c-a371b2d9f198
# ╠═ca52db12-954f-55d9-b71e-81a24e630d59
# ╟─43e64bb7-9da7-4066-a124-3b999164c722
# ╠═300b0f43-de6d-5f1e-87c1-d45a71510915
# ╟─1b2a7cb7-6569-4093-9806-bca1e665cbb3
# ╠═a66a76b2-9f64-4ff0-8f5c-5ccf3eba8d9c
# ╟─d1af595e-6a01-45fd-8885-1068e5a2dcd4
# ╠═14956fcc-0dd5-5271-8230-c434cbe22bfa
# ╟─ff744d04-317b-42ce-9217-22a05535dc7f
# ╠═c10841f5-ec8d-415c-84a6-216611f5074c
# ╠═afdc66a8-40d9-54c1-9a04-6862b90e2f4d
# ╟─d0c35b35-3ffa-59cd-bf7d-8c551ab00845
# ╟─70dcb2b6-b983-5952-82d1-00adc71fbb7d
# ╟─981c8c99-4fd2-5ba3-b745-c986a106bb46
# ╟─37b6fd25-1dba-55a2-8d41-aa7a689e52d8
# ╟─1f86c813-7a0d-577a-8531-4f98ea6d426d
# ╟─ab726d26-082c-4c83-b20c-063d45f438b3
# ╠═69c7a1fb-4ccf-5622-a0fb-fcedd41c9ab5
# ╟─28d1adda-8b0c-43c7-a18b-a89d7cc5028d
# ╟─40f37aae-7096-519f-81e6-ac14fd65dec5
# ╠═db7c766c-5609-519c-818d-d51b0b8407ce
# ╟─e4b8adb2-fc28-4390-a563-25bd3ac64fd6
# ╠═9bf87089-2b58-5388-b04a-5cdf527ebee0
# ╠═0767132c-ef65-5b65-8d30-d7413b06e1ef
# ╠═f29b720f-66bc-5730-84ec-83c897fef1bc
# ╟─9b1d9232-85e9-5be2-8cdd-1d3b3203d271
# ╠═e919d87b-6530-5def-a0cd-79ec01fd46b9
# ╟─949dd8cc-dbe5-5200-a20e-f31f7cd8af51
# ╟─634f41b9-e54e-55a8-a517-67ab466e5494
# ╠═da958c7c-b6ee-5384-b106-de96d63dbf87
# ╟─e2306bfc-c7d9-4ac8-b8c8-5cd6da4e8af6
# ╠═eefa07df-af38-5faf-9e45-1876af0b0389
# ╟─e6bebb61-e47d-5363-8152-1ba43837e636
# ╟─6c0f2e1a-722d-5a88-ba33-39c262e1e4f3
# ╟─ca4ece31-c355-5b7b-ba80-05518918610d
# ╠═07693073-f6a5-58c2-9a99-5003ed13b613
# ╟─4e2fd55e-500b-4b26-a542-b1580f962635
# ╟─e711492e-b1ef-56f4-86dd-6168f3b92a23
# ╠═7c7ee428-0693-5196-9f70-c16bb7178624
# ╟─104348f3-a806-593a-9ba0-9f4d8299dc7c
# ╟─bccd5d9c-0d2b-5cfe-bb9e-354f6dc49c40
# ╟─f0b0efeb-cce2-5461-8f7c-f20cce5ba8c1
# ╟─de44bd39-1e6e-52b3-b187-2d82e8bea8fb
# ╠═ddfbda75-1b44-5501-a9dc-8b421b87771a
# ╟─a47f5125-3bd6-59a7-94c6-ed0aeb7fd301
# ╟─b1c2acb1-878c-5e54-b0b4-bae24b84c0a5
# ╠═6608cd0a-e892-5ca5-851e-e88dbb3df674
# ╠═a1767280-2cc6-55d8-ba38-498a4f923a76
# ╠═364bb9bb-dbb8-5bcf-a639-a3aa958fca4c
# ╟─120b372d-e133-5d3f-9742-fb115ea82ad7
# ╠═ddc467e0-9956-567f-a678-b614a402e47e
# ╟─c0e0be39-665b-5fce-92e5-00b755be7543
# ╟─ebe14752-9b79-5ce1-b2e6-64cff35c45bb
# ╟─8b85a889-8683-515d-b641-79368ec90f80
# ╟─97700adc-8d57-5055-b652-e03984aa6501
# ╠═94fc4917-ac38-51c9-892d-1ac751e4c3a1
# ╠═0314eace-722a-5c30-9edc-4ee05acf1020
# ╟─77f6c10b-7c89-5e29-8879-76f2a6f66baf
# ╟─2f6b3cb7-1efb-5826-b335-c532f8d011f1
# ╟─1ee88810-ebd2-5f59-986a-8da7523aad6e
# ╠═cc976cec-b99f-53ac-a43e-b36d2e329785
# ╟─5d96ecad-cb6a-5d5b-96d6-c8d68c1c9656
# ╟─09c99de7-c2fa-550b-b2c3-7c1e6c82aae5
# ╟─e9341e50-d24e-50c3-8a63-c6a0d5d11925
# ╟─790985b6-c3c8-564f-8216-c443092cd97b
# ╟─78797b07-984c-512d-8a72-ced0da21df6a
# ╠═20c2ffe7-f0d1-56a9-9ec9-13b0bc345802
# ╟─97a0f4d3-d7df-58b2-871a-0a59c0d0c27b
# ╟─cd539602-f958-50ee-9819-7f9fea6c7202
# ╟─9237f845-61c5-5b63-9363-a1dac0a6e75a
# ╟─baf84fac-12da-50b2-923b-61883ba2d5e5
# ╟─bee35bd4-698d-563c-879f-2cf80429f8e8
# ╟─71391962-660d-5d7f-86ab-c9319fa462ad
# ╠═c1101a91-3b22-5788-a3b1-1c817c8f659e
# ╠═491b798f-57f6-5219-a476-6c233c759cae
# ╠═3ccd84b9-df22-566a-bbfc-a47705401a44
# ╠═202c4c72-88aa-584b-bc57-762289b84b83
# ╠═4cd13998-ab74-5b2d-b10f-8d98e48becc4
# ╠═68c08ab6-3f9f-5302-86e0-9fb996cb4a5a
# ╠═8abab510-3437-5815-aaa1-75362cd707fa
# ╠═3b3a36da-42e1-5233-b2a7-1d75cb8a575c
# ╠═71857ff0-130b-5a3d-807f-a0ed8c5694d8
# ╠═4204233e-773b-51a2-a9d3-9394b4d98f97
# ╠═704d8f96-cd78-5ddf-9a90-30cb68d4dc6a
# ╠═90501124-d3c4-5649-b5e5-0dbafbfd08e9
# ╠═08c82a03-9fa4-5b9d-a374-b5563d56bbf1
# ╠═72ed5461-7de0-5be6-9e20-b2070b71b343
# ╠═33708e56-08e2-5be5-b599-2d26008dbf7a
# ╠═e0e939af-fa4b-5f1c-82bf-660dc87bf2d8
# ╠═da9c47c3-26ea-5e4e-9755-8df333ff3e1e
# ╠═e715cea6-339d-5d77-8109-d8cfdf9d2ccd
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
