### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ 65cc8b96-25fe-5676-aac0-d4715c84314e
begin
    using CairoMakie, PlutoUI
    using Printf, Random, Statistics
    using ComponentArrays: ComponentArray, getaxes
    import Lux, Optimisers, Zygote
end

# ╔═╡ b3dce1ff-f7f7-5a3b-97e3-a19f8dcaca7a
md"""
# Keeping Neural Networks Simple by Minimizing the Description Length of the Weights

*A weight can hold any number of bits. How many does a network need?*

**Prerequisites**

- [Notebook 03](https://zxygentoo.github.io/the-ninety-percent-list/03.A.Tutorial.Introduction.to.the.Minimum.Description.Length.Principle.html), or MDL's two-part code and COMP (parametric complexity).
- Basic understanding of the Gaussian distribution.
- Basic understanding of a neural network: how it's trained, and how held-out data tunes it.

---

[Notebook 03](https://zxygentoo.github.io/the-ninety-percent-list/03.A.Tutorial.Introduction.to.the.Minimum.Description.Length.Principle.html)'s Digits is a machine whose single knob holds every bit of the string it describes. Counting parameters didn't catch it (Bernoulli has one knob too); counting bits did.

Geoffrey Hinton and Drew van Camp's [paper](https://www.cs.toronto.edu/~hinton/absps/colt93.pdf) counts the bits in a neural network's weights, and trains the network to keep them few. This was one of the first serious attempts to put MDL to work on neural networks, and the loss it wrote down is still in use today.

In this notebook, we test the paper's central claim:

> A network generalizes well when its weights hold much less information than its training targets, so it should be trained on the expected squared error plus the information in its weights.

We keep the paper's basic approach but update its engineering:
- [Lux](https://lux.csail.mit.edu) describes the network.
- [Adam](https://arxiv.org/abs/1412.6980) replaces [conjugate gradient](https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method).
- Random samples replace the paper's tables for pushing Gaussians through the hidden units.

Section numbers refer to the paper.
"""

# ╔═╡ ca4d9ebf-335b-573f-8a78-32072021bcc1
TableOfContents()

# ╔═╡ 0e7627a5-f61c-5b54-a1ad-df392620ffb1
md"""
## COMP can't really work for neural nets

Notebook 03 priced a model by its fit plus its COMP. COMP looks past the data at hand to every dataset the model could have been handed, gives each one its best fit, and adds up the probabilities those fits assign. The total is above 1, since every dataset picked its own fit, and COMP is that overspend, in bits.

A network of a given shape is a model too, and each setting of its weights is a member. What would COMP say about one?

### Too many datasets

For strings of ``n`` bits, there are ``2^n`` datasets, and each needs its own best fit. Here is Bernoulli's COMP, summed the long way:
"""

# ╔═╡ 73b6b9b6-1ba8-5d35-a029-3329024fbf64
"The best fit, in bits, of `total` coin flips that landed 1 `ones` times: the bias set to `ones / total`."
bernoulli_fit(ones, total) = sum(
	c -> c == 0 ? 0.0 : -c * log2(c / total), (ones, total - ones)
)

# ╔═╡ 32501cfd-cdf6-5491-98be-191229948e55
"Bernoulli's COMP by brute force: every string of `n` bits, as the integer it spells, each with its own best fit."
bernoulli_comp(n) = log2(
	sum(x -> exp2(-bernoulli_fit(count_ones(x), n)), 0:2^n-1)
)

# ╔═╡ b14d1afc-72b0-5118-bdab-1944df3afb78
# Each row sums 2ⁿ strings; the times are this machine's.
COMP_TIMINGS = begin
    bernoulli_comp(4)    # compile first, so the times are of the sum alone
    map((16, 20, 24)) do n
        seconds = @elapsed comp = bernoulli_comp(n)
        (; n, comp, seconds)
    end
end;

# ╔═╡ 2d3bba55-ba86-57a8-a905-659e57b6bb77
md"""
### A network with more weights than cases

The paper's one experiment (§9) predicts how well peptide molecules work: 128 inputs, only 105 training cases, and a network with 521 weights that "overfits the 105 training cases very badly" without a penalty.

Our task keeps that shape, more weights than cases, but has a truth we can draw. The target depends on two inputs, ``x_1`` and ``x_2``, through two ridges crossing at right angles:

```math
f(x_1, x_2) = \tanh(2x_1 + x_2) - \tanh(x_1 - 2x_2),
```

plus noise with a standard deviation of 0.1. Twenty more inputs are distractors: random numbers the target ignores.
"""

# ╔═╡ 0e2eeebd-951a-5eea-b995-04fd66b688dd
begin
    INPUTS = 22      # x₁, x₂, and 20 distractors
    HIDDEN = 8
    NOISE = 0.1
end;

# ╔═╡ 883df49d-a6a1-5194-b5e5-90c916d52d50
"Two ridges crossing at right angles: four plateaus."
truth(x₁, x₂) = tanh(2x₁ + x₂) - tanh(x₁ - 2x₂)

# ╔═╡ e6b75c28-b9ba-5ee8-a231-a6fca1845bef
"`n` cases: standard Gaussian inputs, and the truth of the first two plus noise as the target."
function dataset(rng, n)
    x = randn(rng, INPUTS, n)
    (; x, y = reshape(truth.(x[1, :], x[2, :]), 1, n) .+ NOISE .* randn(rng, 1, n))
end

# ╔═╡ 0a7ef374-adca-5b9b-8159-1f85b9197338
TRAIN, TEST = dataset(Xoshiro(4), 100), dataset(Xoshiro(5), 2000);

# ╔═╡ ce3d4e6b-baa9-4627-8cb7-7d262ff86f52
md"""The network has one hidden layer of $HIDDEN tanh units and a linear output."""

# ╔═╡ f0cc85b0-16f3-5503-b720-3983d7094b28
MODEL = Lux.Chain(
	Lux.Dense(INPUTS => HIDDEN, tanh),
	Lux.Dense(HIDDEN => 1)
)

# ╔═╡ fcb787e3-5f33-57a4-a550-fd6e5f3ddf17
"A fresh set of weights, as one flat vector with named parts: `layer_1.weight`, `layer_1.bias`, and so on."
weights(rng) = ComponentArray(Lux.f64(first(Lux.setup(rng, MODEL))))

# ╔═╡ 71c1f97f-0824-5767-8a69-03e6955c0682
"Minimize `loss(θ, t)` over `θ` with Adam, `t` being the step; the learning rate drops tenfold from step `steps - cooldown` on."
function train(loss, θ; steps, lr = 1e-2, cooldown = 0)
    opt = Optimisers.setup(Optimisers.Adam(lr), θ)
    for t in 1:steps
        t == steps - cooldown && Optimisers.adjust!(opt, lr / 10)
        g = only(Zygote.gradient(θ -> loss(θ, t), θ))
        opt, θ = Optimisers.update(opt, θ, g)
    end
    θ
end

# ╔═╡ 5e8b963b-a5d0-536a-b53f-968e33d67bae
md"""
### Noise in, noise out

Train the network the plain way first: squared error alone, no penalty, minimized with Adam.
"""

# ╔═╡ 837af198-c6af-58f0-be18-50d0030689c8
md"""
We measure error as the paper does (eq. 27): squared error on the test set, relative to always guessing the mean. A score of 1 is no better than the mean. A score of 0 is out of reach, since no predictor can foresee the noise in the test targets. Even the truth misses each by its noise, about 0.1² in squared error against the targets' variance of about 1.3, so it scores about 0.007. That is the floor.
"""

# ╔═╡ 5a4add4f-3828-5aca-b76c-3960b85c880d
relative_error(ŷ, data) = sum(abs2, ŷ .- data.y) / sum(abs2, data.y .- mean(data.y))

# ╔═╡ 92788641-8d09-5fd1-b9c2-4c7e1c087ec3
# The truth's own relative test error: the noise alone.
FLOOR = relative_error(reshape(truth.(TEST.x[1, :], TEST.x[2, :]), 1, :), TEST)

# ╔═╡ 74a4508b-259f-541d-ba00-7aff49ee6f0f
md"""
Here is what the plain network learned, drawn on the plane. Each pixel gets fresh random values for the 20 distractors, as a test case would:
"""

# ╔═╡ eccb9cb7-c861-5a93-a115-86f10262b9ec
md"""
## The weights and the misfits

The paper sets the problem up as a game between two people (§2).

The *sender* sees a set of training cases, each an input and its target. The *receiver* sees only the inputs. They have agreed on the shape of a network in advance, and the sender wants the receiver to end up with every target.

So the sender trains the network and sends its weights. Then, for each case, the sender sends the *misfit*: the target minus the network's output. The receiver runs the network on each input, adds the misfit, and has the target.

The bill has two parts, the bits for the weights and the bits for the misfits, and they pull against each other. A network that fits closely leaves small misfits, which are cheap to send, but its weights may cost a lot. A crude network is cheap to send, and leaves large misfits. MDL picks the network that makes the sum smallest.

### Squared error is a codelength

A real number takes infinitely many bits, so it can only be sent to some tolerance, ``t``: the receiver learns which bin of width ``t`` it falls in. Under a distribution ``P``, that bin's probability is about the height of ``P`` there, times the bin's width:
"""

# ╔═╡ b05932cc-c7d9-5410-b7f2-b053cfeb4326
md"""
The bin's area, ``t\,P(x)``, is the probability of sending ``x``, so sending it costs

```math
-\log_2\big(t\,P(x)\big) \;=\; -\log_2 P(x) \;+\; \log_2\frac{1}{t} \quad \text{bits}.
```

The first term depends on where ``x`` falls. The second, the *tolerance term*, is the same for every value, and grows as the tolerance gets finer.

Now send a misfit ``r`` this way, with a Gaussian of width ``s`` as ``P`` (§3):

```math
P(r) = \frac{1}{\sqrt{2\pi}\,s}\, e^{-r^2/2s^2} .
```

Plug it in, and the cost splits into the squared error and two terms that don't depend on ``r``:

```math
-\log_2\big(t\,P(r)\big) \;=\; \frac{r^2}{2s^2}\log_2 e \;+\; \log_2\!\big(\sqrt{2\pi}\,s\big) \;+\; \log_2\frac{1}{t} .
```

Add up all ``N`` misfits, ``r_1`` to ``r_N``:

```math
\frac{\log_2 e}{2s^2}\sum_{c=1}^{N} r_c^2 \;+\; N\log_2\!\big(\sqrt{2\pi}\,s\big) \;+\; N\log_2\frac{1}{t} .
```

The sender is free to choose ``s``, and the two terms with ``s`` in them pull against each other. A narrow Gaussian makes the squared error expensive; a wide one spreads its probability thin, and every misfit pays for it. The best width is where the cost stops falling. Drop the terms without ``s``, and use ``\tfrac{d}{ds}\log_2 s = \log_2 e / s``:

```math
\frac{d}{ds}\left[\frac{\log_2 e}{2s^2}\sum_c r_c^2 + N\log_2 s\right] = \log_2 e\left(\frac{N}{s} - \frac{1}{s^3}\sum_c r_c^2\right) = 0 .
```

Multiply through by ``s^3 / \log_2 e`` and solve for ``s^2``:

```math
s^2 = \frac{1}{N}\sum_c r_c^2 = \mathrm{mse} .
```

So the best width is the root-mean-square misfit. Put it back, and the first term becomes ``\tfrac{N}{2}\log_2 e`` and the second ``\tfrac{N}{2}\log_2(2\pi\cdot\mathrm{mse})``. The total is then

```math
\frac{N}{2}\log_2\big(2\pi e \cdot \mathrm{mse}\big) \;+\; N\log_2\frac{1}{t} \quad \text{bits}.
```

Only the mean squared error depends on the network. Halve the misfits, and each one costs a bit less.

The code works in *nats*, the unit of natural logarithms: 1 nat is ``\log_2 e \approx 1.44`` bits, and `bits` converts.
"""

# ╔═╡ 74aab869-7574-5319-9f4a-4d2db701db61
"The misfits `r`, in nats, coded with a Gaussian of width `s`, leaving out the tolerance term."
misfit_cost(r, s) = sum(abs2, r) / (2s^2) + length(r) * (log(s) + log(2π) / 2)

# ╔═╡ 2456446b-4992-5de2-b992-547ad8a1f182
"The same, with `s` set to the root-mean-square misfit: `n` misfits of mean square `mse`."
best_misfit_cost(mse, n) = n / 2 * log(2π * ℯ * mse)

# ╔═╡ d4d41340-f72a-59de-953f-566e994be2e4
md"""
The tolerance term is the same for every network that sends these targets, so it cancels whenever two codes are compared. We compare every code with the simplest one, *no network*: the sender codes the targets themselves, with one Gaussian at their own mean and spread.
"""

# ╔═╡ 5595ab9e-5cb7-55d0-a5fc-78ce8b192ad9
md"""
### Weight decay is a code too

[*Weight decay*](https://papers.nips.cc/paper/1991/hash/8eefcfdf5990e441f0fb6f3fad709e21-Abstract.html) was the standard remedy for an overfitting network, and it is still in use. Add a penalty on the squared weights to the training loss, ``\lambda \sum w^2``. Its gradient, ``2\lambda w``, shrinks every weight by a small fixed fraction at every step, so a weight decays toward 0 unless the data keep pushing it back. The strength ``\lambda`` is a knob, usually tuned by trying values on held-out data. It is the method the paper sets out to beat.

The paper's first move is to show that weight decay is already MDL (§4). Send the weights the same way as the misfits, under a *prior*, a distribution agreed on before any data is seen: a Gaussian of width ``\sigma_P`` centred on 0. They cost ``\tfrac{\log_2 e}{2\sigma_P^2}\sum w^2`` bits, plus terms that don't depend on the weights. Add the two parts, and take out the misfits' factor:

```math
\frac{\log_2 e}{2s^2}\sum_c r_c^2 + \frac{\log_2 e}{2\sigma_P^2}\sum w^2 \;=\; \frac{\log_2 e}{2s^2}\Big[\sum_c r_c^2 + \frac{s^2}{\sigma_P^2}\sum w^2\Big] .
```

The bracket is the summed squared error plus a penalty on the squared weights: weight decay, with its strength set by the two widths, ``\lambda = s^2 / \sigma_P^2``.

So weight decay was a description length all along, and its knob is a ratio of two code widths. The paper reads weight decay's success as "a vindication of this crude MDL approach".

But what makes it crude?

Go back to the bin. A weight ``w``, sent under a prior ``P``, costs ``-\log_2 P(w)`` for the height and ``\log_2(1/t)`` for the width. Weight decay charges for the first and fixes the second: every weight is sent to the same tolerance, fine enough for the most sensitive one.

That wastes bits wherever a weight could be sent coarsely. A weight that can move by 0.5 without changing the network's output needs only a coarse bin; a weight that matters to the fourth decimal needs a fine one. The paper calls ignoring the difference "a terrible waste of bits" (§4). The waste also hides a choice: with the same tax on every network, the bill can never prefer one whose weights may be sent coarsely over one that needs every weight exact.

How can each weight choose its own precision?
"""

# ╔═╡ 4f2d3cd5-aa6a-55c2-b701-d6eb03f48907
md"""
## Expected squared error plus KL

Make each weight a small cloud instead of a point: a Gaussian, ``Q = \mathcal{N}(\mu, \sigma^2)`` (§5). Its mean ``\mu`` is where the weight sits, and its width ``\sigma`` says how precisely. A wide cloud is a weight that doesn't need to be exact. Both are learned.

### Bits back

How do you send a cloud? The sender picks one point from it at random, ``w \sim Q``, and sends that point under the prior, to a fine tolerance. That is the bin from before, with the prior as its distribution:

```math
\text{cost} = -\log_2\big(t\,P(w)\big) \quad \text{bits},
```

and with a fine ``t`` it is a lot.

Then comes a refund (§5.2). Once the receiver has the weights and the misfits, they have the targets, so they can rerun the sender's training, random seeds and all, and arrive at the same cloud ``Q``. Knowing the cloud and the point, they can work out the random bits the sender used to pick the point. The sender didn't need real coin flips for that. They could use the first bits of some other message they owe the receiver anyway, and the receiver reads that message back off the point, a trick since built into [working compressors](https://arxiv.org/abs/1901.04866). Those bits were delivered, so they come off the bill.

How many are there? Picking an outcome of probability ``p`` takes about ``-\log_2 p`` fair coin flips, as many as sending it would: feed coin flips to the decoder of notebook 03's arithmetic coder, and out comes a sample. A bin of width ``t`` has probability ``t\,Q(w)`` under the cloud, so picking it takes

```math
\text{refund} = -\log_2\big(t\,Q(w)\big) \quad \text{bits}.
```

What one pick really costs is the difference:

```math
\text{cost} - \text{refund} \;=\; -\log_2\big(t\,P(w)\big) + \log_2\big(t\,Q(w)\big) \;=\; \log_2\frac{Q(w)}{P(w)} .
```

The tolerance cancels. A finer ``t`` makes the cost and the refund grow together, so precision is no longer a fixed tax: each weight chooses its own, through the width of its cloud. What's left depends on which point was picked, so average it over the pick:

```math
\mathbb{E}_{w \sim Q}\left[\log_2 \frac{Q(w)}{P(w)}\right] = \mathrm{KL}(Q \,\|\, P),
```

the [*Kullback–Leibler divergence*](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), KL for short. This is the information in a weight, the term the paper adds to the loss.

> Think of an address. The prior is a city, and the cloud is the district where the weight must land. Any house in the district will do, so the house number can spell out part of another message. What the sender really pays for is naming the district: about ``\log_2`` of the city's size over the district's.

For a Gaussian cloud and a Gaussian prior ``\mathcal{N}(0, \sigma_P^2)``, the divergence has a closed form (eq. 9). The paper writes it in nats, and so does our code:

```math
\mathrm{KL}(Q \,\|\, P) \;=\; \log\frac{\sigma_P}{\sigma} \;+\; \frac{\sigma^2 + \mu^2}{2\sigma_P^2} \;-\; \frac12 .
```

The ``\mu^2`` term is weight decay again. The ``\log(\sigma_P / \sigma)`` term is new, and it charges for precision. A cloud as wide as the prior costs nothing for it, and each halving of its width costs one more bit.

Notice what the precision is measured against: the prior. A weight's bits count how much the data narrows it *relative to the prior*, so a cloud that looks like the prior, centred where the prior is and about as wide, is nearly free, however narrow both are.
"""

# ╔═╡ 9a217e44-1307-51a8-97b1-2473236dfb1d
"KL(Q ‖ P) in nats, for a cloud Q = N(μ, σ²) and a prior P = N(0, σₚ²): eq. 9."
kl(μ, σ, σₚ) = log(σₚ / σ) + (σ^2 + μ^2) / (2σₚ^2) - 1 / 2

# ╔═╡ 34ac9ced-0eac-5cd2-b82a-6ea260a10ae7
md"""
### The misfits, with noisy weights

The receiver runs the network with the weights it was sent, a random draw from the clouds. So the outputs are random, and so are the misfits and their bill. As with the weights, what counts is the average over the draw. The squared error, averaged that way, splits in two on each case (eq. 12):

```math
\mathbb{E}\big[(d - y)^2\big] = (d - \bar y)^2 + \mathrm{Var}(y) ,
```

where ``d`` is the target, ``y`` the output, and ``\bar y`` its mean over the draws. The first term is the systematic error. The second is the variance the noise in the weights adds to the output, paid for in misfit bits. So a weight can afford a wide cloud only where the output doesn't care about it.

### The loss

Put the two parts together. Sending the targets with a cloud of weights costs, in nats,

```math
\underbrace{\frac{1}{2s^2}\,\mathbb{E}_{w \sim Q}\Big[\sum_c (d_c - y_c)^2\Big]}_{\text{expected squared error}} \;+\; \underbrace{\mathrm{KL}(Q \,\|\, P)}_{\text{information in the weights}} \;+\; N \log\!\big(\sqrt{2\pi}\, s\big),
```

with the tolerance term left out, and the network is trained on it. The last term is the normalizer of the misfit code, from *Squared error is a codelength*. It doesn't depend on the weights, but the code's width ``s`` is learned too, and without the term, a wider ``s`` would make the squared error free.

Widening a cloud shrinks its KL but adds variance to the output; narrowing it does the reverse. Each weight settles where the two balance.

Every choice of clouds gives a code the sender could really use, so the loss is a codelength the sender can achieve. It is not the shortest the prior allows. That would be ``-\log_2 P(\text{targets})``, with the weights averaged over the prior, and the loss reaches it only when ``Q`` is the exact *posterior*, the prior updated on the targets by Bayes' rule. That can take any shape, and a Gaussian per weight is not it. So the loss is an upper bound, and training tightens it.

> The loss has modern names: it is the negative [*evidence lower bound*](https://en.wikipedia.org/wiki/Evidence_lower_bound) (ELBO), and training a network on it with sampled gradients is *Bayes by Backprop* ([Blundell et al., 2015](https://arxiv.org/abs/1505.05424)), which grew out of [Graves (2011)](https://proceedings.neurips.cc/paper/2011/hash/7eb3c8be3d411e8ebfab08eba5f49632-Abstract.html).
"""

# ╔═╡ 6094aea6-a49a-5994-af61-7fa4253be3cd
md"""
### Training on it

The paper computed the expected squared error exactly, with tables for pushing a Gaussian through each hidden unit, which only works for one hidden layer and linear outputs (§5.3). Nowadays we sample instead. Draw ``\varepsilon`` from a standard Gaussian and set ``w = \mu + \sigma \varepsilon``, and the gradient of what that sample costs flows back to ``\mu`` and ``\sigma``. This is the [*reparameterization trick*](https://en.wikipedia.org/wiki/Reparameterization_trick). A few samples per step are enough. For the last 2,000 steps the learning rate drops tenfold, so the jitter from sampling dies down and the ledger reads a settled network.

In code, a cloud is one flat vector `θ` with named parts: the means `θ.μ`, the log-widths `θ.ρ`, the log of the misfit code's width ``s``, and the prior's own parameters.
"""

# ╔═╡ 13dc376e-9c80-51b8-ad55-43405a0bede7
# Lux names the layers of a Chain `layer_1`, `layer_2`, …
LAYERS = (:layer_1, :layer_2)

# ╔═╡ b2bf51b2-ec1d-5779-b33f-ab29f6252e3e
widths(θ) = exp.(θ.ρ)

# ╔═╡ b5a144de-7e9c-5e59-8d36-eeaca5979795
"One draw of the weights from the cloud θ, given standard Gaussian noise ε."
sample(θ, ε) = θ.μ .+ widths(θ) .* ε

# ╔═╡ 6a142169-be9c-551e-971d-f119bd744a7d
"Standard Gaussian noise, shaped like the weights μ."
noise_like(rng, μ) = ComponentArray(randn(rng, length(μ)), getaxes(μ))

# ╔═╡ 5f9babdd-3f19-5094-a970-b54c312a128b
"One Gaussian prior per layer, N(0, σₚ²), stored as log σₚ."
gaussian_prior() = ComponentArray(layer_1 = 0.0, layer_2 = 0.0)

# ╔═╡ 01b6047a-9102-5935-8aa1-81fd0a989d99
md"""
Two more things are learned along with the clouds: the width ``s`` of the misfit code, and the width of the prior for each layer, "letting the data determine the prior" (§6). These are only a few numbers, and the paper ignores what they cost to send. So do we.

Two habits come from the paper's experiment (§9):

- **A warm-up.** With the full weight cost from the start, the paper found that "it is initially very easy to reduce the combined cost function by using weights that contain almost no information, and it is very hard to escape from this poor solution". So the weight cost starts at 5% of its full size and rises to 100% over the first 3,000 steps. We also hold the prior still during the warm-up, so that it doesn't shrink to fit weights that haven't grown yet.
- **Five starts.** Train from five random starts, and keep the one with the shortest description: the one that saves the most. That is MDL's own rule, applied once more, and it never looks at test data.
"""

# ╔═╡ e5095ad0-2394-593c-842a-4c187bbc910f
# Noise draws per training step.
SAMPLES = 4

# ╔═╡ c73bfdc7-f9d3-5945-acf1-dfdaca253077
md"""
## Testing the claim

The claim has two halves, and our task can test both. Trained on the loss, a network should generalize better than one tuned the usual way, and without holding any data out. And among trained networks, the ones with the largest net saving should generalize best.

### The baseline

The paper compared against weight decay, tuned with held-out data. Try strengths ``\lambda`` from ``10^{-4}`` to 1, score each by 5-fold [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_%28statistics%29) on the training cases, and retrain on all of them with the best. Our ``\lambda`` weighs the penalty against the *mean* squared error rather than the sum, so it is the ratio of widths from above divided by the number of cases, ``\lambda = s^2 / N\sigma_P^2``:
"""

# ╔═╡ 604db986-57ab-5229-991d-1ed52f386a91
subset(data, idx) = (x = data.x[:, idx], y = data.y[:, idx])

# ╔═╡ 3a71b0d9-be6c-54b8-9e0d-eae5dfa59dad
md"""
### One Gaussian

Start with the paper's basic code: one Gaussian prior for each layer, its width learned (§5–6). Train from five starts:
"""

# ╔═╡ 11bdb645-bcf7-5c15-a5c6-2b943b4027a8
md"""
Unlike weight decay, it has found all four plateaus, the two at 0 included. What speckle is left sits on the edges of the ridges. There each unit is steep, so the little that its distractor weights still let through is enough to tip a pixel; on the plateaus the units are saturated, and the same leak changes nothing.
"""

# ╔═╡ 4cd46967-e9fc-5a51-b092-7325a7abf661
md"""
### Noise in, nothing out

Back in *Noise in, noise out*, the plain network memorized pure noise. That is overfitting at its purest: there is nothing in the targets to learn, so nothing it learned can carry over to new cases.

A loss that charges for the information in the weights should refuse. Memorizing would save bits on the misfits, but only by first storing them in the weights, and random numbers can't be stored in fewer bits than they save. So on noise the ledger can't come out ahead, except by luck. Train the noisy network on the same targets, from five starts again:
"""

# ╔═╡ 8d48e436-8617-534e-a7e8-495c71f148b7
md"""
### A mixture of Gaussians

The paper's answer is a richer prior: a mixture of Gaussians, from [Nowlan and Hinton (1992)](https://www.cs.toronto.edu/~hinton/absps/sunspots.pdf) (§6.1, §7). Its own mixture had five, with learned means, so that each could settle on a cluster of weights, and its result, a relative test error of 0.286 against weight decay's 0.317, came from that prior. Ours is simpler: two components, both centred on 0, their widths and shares learned, and shared by the weights of both layers. A narrow one, the *spike*, is for the weights that should be 0, and a wide one for the ones that shouldn't. Saying "zero" now means pointing at the spike.

The KL from a cloud to a mixture has no closed form, so we use the paper's bound (eq. 17), where ``\pi_k`` is the share of the component ``P_k``:

```math
\mathrm{KL}(Q \,\|\, P) \;\le\; -\log \sum_k \pi_k \, e^{-\mathrm{KL}(Q \,\|\, P_k)} .
```

The paper shows that the bound is itself the cost of a code (§7), so the loss is still a description length.
"""

# ╔═╡ 14931e07-d199-5611-84ca-0a515a39903c
logsumexp(v) = let m = Zygote.ignore_derivatives(() -> maximum(v)); m + log(sum(exp.(v .- m))) end

# ╔═╡ 781d8d49-01ea-5211-9ecd-a4a0e064c0c8
"A mixture of two zero-mean Gaussians, a narrow spike and a wide one, shared by all the weights: stored as log widths and the logits of their shares."
mixture_prior() = ComponentArray(log_σ = [log(0.05), log(1.0)], logit = [0.0, 0.0])

# ╔═╡ ee6e09f2-2151-5238-8cf9-7cbff70d5d2f
is_mixture(prior) = :logit in propertynames(prior)

# ╔═╡ fafc485f-3338-581b-b8ad-5cfd2bd5a055
"The paper's bound (17) on each weight's KL to a mixture, in nats: −log Σₖ πₖ exp(−KL(Q ‖ Pₖ))."
function mixture_costs(μ, σ, prior)
    σs = exp.(prior.log_σ)
    logπ = prior.logit .- logsumexp(prior.logit)
    terms = [logπ[k] .- kl.(μ, σ, σs[k]) for k in eachindex(σs)]
    m = Zygote.ignore_derivatives(() -> reduce((a, b) -> max.(a, b), terms))
    -(m .+ log.(sum(t -> exp.(t .- m), terms)))
end

# ╔═╡ a2229ad1-11aa-5e0d-b9f3-d610980e2b65
"Each weight's KL in nats, for the means `μ` and widths `σ` of one layer. (Mixture priors come later, in A mixture of Gaussians.)"
costs(μ, σ, prior, layer) = is_mixture(prior) ? mixture_costs(μ, σ, prior) : kl.(μ, σ, exp(getproperty(prior, layer)))

# ╔═╡ a9143211-6207-5f4f-8b09-b97a06ef55b3
"The weights' KL in nats: every weight of every layer."
weight_cost(θ, prior = θ.prior) = sum(l -> sum(costs(getproperty(θ.μ, l), getproperty(widths(θ), l), prior, l)), LAYERS)

# ╔═╡ 80a53f12-49fd-556b-bd19-0a3e6bc35079
md"""
Five starts again:
"""

# ╔═╡ b63350b6-22a5-5904-af65-02b3571cb763
md"""
### The embarrassment

All of this had 100 cases for 22 inputs. The paper had 105 cases for 128 inputs, fewer cases than inputs. Take our cases down to 50, then 30, and train both codes from five starts again:
"""

# ╔═╡ f95b8975-7eb9-5d1c-86d0-5ee0d19c1178
"The truth as network weights: units 1 and 2 are the two ridges, and everything else is 0."
function truth_means()
    μ = zero(weights(Xoshiro(0)))
    μ.layer_1.weight[1, 1:2] .= (2, 1)
    μ.layer_1.weight[2, 1:2] .= (-1, 2)
    μ.layer_2.weight[1, 1:2] .= (1, 1)
    μ
end

# ╔═╡ 1ff92f76-894f-57d4-bfbe-b626480e13cb
md"""
## Appendix: helpers

Counting cases and converting nats, the cloud's predictions and its bills, the plane, number formatting, and the chart style.
"""

# ╔═╡ 3f6cc915-39b8-50cd-9527-64993e41a46b
num_of_cases(data) = size(data.y, 2)

# ╔═╡ 9ccc1f86-94c6-534c-bd09-cad1de2d352c
# The same inputs, with targets that are pure noise.
NOISE_TARGETS = (x = TRAIN.x, y = randn(Xoshiro(101), 1, num_of_cases(TRAIN)));

# ╔═╡ 2aeefb9c-d9f3-5189-9d01-99296749a21b
bits(nats) = nats / log(2)

# ╔═╡ 645418b5-949a-549b-a139-473f5f0dc1bf
"The bits to send `data`'s targets with no network: one Gaussian code, at their own mean and spread."
no_network(data) = bits(
	best_misfit_cost(var(data.y; corrected = false), num_of_cases(data))
)

# ╔═╡ f7728920-be5f-5993-bc72-bbecbcf07d80
# Dense layers keep no state, but Lux asks for it.
STATE = last(Lux.setup(Xoshiro(0), MODEL));

# ╔═╡ d62c9d9b-80f5-5977-90a7-8915a360dd90
"Forward pass."
predict(w, x) = first(MODEL(x, w, STATE))

# ╔═╡ 25420e04-b36b-5938-8fd7-472bead269bc
"Sum of squared errors."
sse(w, data) = sum(abs2, predict(w, data.x) .- data.y)

# ╔═╡ c5f39cf6-2893-5993-9e5b-30cc55ea7bab
"The network trained on squared error alone."
plain_net(data) = train(
	(w, t) -> sse(w, data) / num_of_cases(data),
	weights(Xoshiro(1));
	steps = 5000
)

# ╔═╡ 09eabb3f-704e-59e6-b54d-505eeeb7d0e4
PLAIN = plain_net(TRAIN);

# ╔═╡ 364ba16a-24b2-5296-a9f2-dbc60240312f
MEMORIZER = plain_net(NOISE_TARGETS);

# ╔═╡ 5deb2101-b3e8-50d5-b539-def1b355aa30
"""
The cloud θ's ledger for `data`'s targets, in bits against no network: what it saves on the misfits, with the misfit
code's best width; what it spends on the weights, their KL; and the net saving.
"""
function ledger(θ, data; draws = 500, rng = Xoshiro(8))
    mse = mean(_ -> sse(sample(θ, noise_like(rng, θ.μ)), data), 1:draws) / num_of_cases(data)
    saved = no_network(data) - bits(best_misfit_cost(mse, num_of_cases(data)))
    spent = bits(weight_cost(θ))
    (; saved, spent, net = saved - spent)
end

# ╔═╡ dd526d59-4939-5057-b7f7-033238d6c653
"Of several trained clouds, the one with the shortest description, the largest net saving: the paper's rule for restarts."
shortest(θs, data) = argmax(θ -> ledger(θ, data).net, θs)

# ╔═╡ 034f774e-317b-5b32-a608-d5a2772d5815
"The network trained with weight decay of strength λ."
decayed(λ, data) = train(
	(w, t) -> sse(w, data) / num_of_cases(data) + λ * sum(abs2, w), 
	weights(Xoshiro(1));
	steps = 3000
)

# ╔═╡ 6d5d8243-2bc8-5aa5-ac36-a5d9bb3ec05d
"Weight decay as the paper tuned it: each λ scored by 5-fold cross-validation, then the best retrained on all of `data`."
function cross_validated(data; lambdas = 10 .^ (-4:0.5:0), folds = 5, rng = Xoshiro(6))
    n = num_of_cases(data)
    parts = collect(Iterators.partition(randperm(rng, n), n ÷ folds))
    score(λ) = mean(fold -> sse(decayed(λ, subset(data, setdiff(1:n, fold))), subset(data, fold)), parts)
    λ = argmin(score, lambdas)
    (; λ, net = decayed(λ, data))
end

# ╔═╡ 88329ada-a959-5972-b1fe-a4f2f402a6d2
# About 45 trainings; under a minute.
DECAY = cross_validated(TRAIN);

# ╔═╡ e6407394-aa39-5a0b-96c5-2fa925bfa5d8
"""
The loss in nats: the misfit cost averaged over the noise draws `εs`, plus `β` times the weights' KL.
The prior learns only when `learn_prior` is true.
"""
function objective(θ, εs, β, learn_prior, data)
    s = exp(θ.log_noise)
    prior = learn_prior ? θ.prior : Zygote.dropgrad(θ.prior)
    mean(ε -> misfit_cost(predict(sample(θ, ε), data.x) .- data.y, s), εs) + β * weight_cost(θ, prior)
end

# ╔═╡ eac8d019-c638-5190-8e50-de49099a5c27
"Train a cloud of weights on `data`: weight cost warmed up over `warmup` steps, prior held still until `prior_from`."
function train_noisy_net(rng, data; prior = gaussian_prior(), σ₀ = 1e-2, s₀ = 0.5, warmup = 3000, steps = 8000, prior_from = 3000)
    μ = weights(rng)
    θ = ComponentArray(μ = μ, ρ = fill!(similar(μ), log(σ₀)), log_noise = log(s₀), prior = prior)
    β(t) = min(1.0, 0.05 + 0.95 * t / warmup)
    draws() = [noise_like(rng, μ) for _ in 1:SAMPLES]
    train((θ, t) -> objective(θ, Zygote.ignore_derivatives(draws), β(t), t >= prior_from, data), θ; steps, cooldown = 2000)
end

# ╔═╡ 1e5e51eb-80ec-536a-859c-ce087491ce9c
# Five starts, about 5 seconds each.
ONE_GAUSSIAN = [train_noisy_net(Xoshiro(s), TRAIN; prior = gaussian_prior()) for s in 1:5];

# ╔═╡ 59a6369d-a0f2-5e2d-86db-1f19fdf4181f
BEST_GAUSSIAN = shortest(ONE_GAUSSIAN, TRAIN);

# ╔═╡ fcdbe5e8-df07-5650-89fe-622dfd3f217f
# Five starts, about 5 seconds each.
NOISE_NET = shortest([train_noisy_net(Xoshiro(s), NOISE_TARGETS) for s in 1:5], NOISE_TARGETS);

# ╔═╡ 0f928b74-c921-5f8d-a473-00dbe725af3a
# Five starts, about 5 seconds each.
MIXTURE = [train_noisy_net(Xoshiro(s), TRAIN; prior = mixture_prior()) for s in 1:5];

# ╔═╡ 94cf7183-2d9e-5667-aef3-e328987831d8
BEST_MIXTURE = shortest(MIXTURE, TRAIN);

# ╔═╡ 08c9d2ee-9c4c-57db-ac0b-2e8cc638fd03
Markdown.parse("""
The paper offered its claim as intuition. Six years later, David McAllester proved a precise form of it, the PAC-Bayes bound (1999).

His [2013 tutorial](https://arxiv.org/abs/1307.2118) is the gentler way in: it starts from a bound in bits and works up to clouds of weights. In the bound's best-known form, if the prior is chosen before the data are seen, and the cases are drawn independently, then a network whose weights are drawn from a cloud ``Q`` has a test error at most about ``\\sqrt{\\mathrm{KL}(Q \\,\\|\\, P) / 2N}`` above its training error, on a loss scaled from 0 to 1. The KL is the paper's weight cost, in nats.

The loss of the paper is still with us: over latent variables instead of weights, it is the loss of [variational autoencoders](https://arxiv.org/abs/1312.6114). So is its trap. Variational autoencoders warm up their KL term from near 0 ([Bowman et al., 2016](https://arxiv.org/abs/1511.06349)), for the paper's reason.

Notebook 03 left a twist to carry along: [Blier and Ollivier](https://arxiv.org/abs/1802.07044) found that variational codes, the descendants of this loss, give "surprisingly poor" compression bounds on deep networks. The loss is an upper bound, only as tight as the clouds and the prior allow. Even at $(length(BEST_MIXTURE.μ)) weights, ours needed the right prior, a warm-up, and restarts, and at 30 cases a much shorter description existed that training couldn't find.

The next item, Shane Legg's [*Machine Super Intelligence*](https://www.vetta.org/documents/Machine_Super_Intelligence.pdf), asks what the right prior is, if not one we chose. [Solomonoff's](https://en.wikipedia.org/wiki/Solomonoff%27s_theory_of_inductive_inference) answer weighs every computable hypothesis by ``2^{-K}``: [notebook 02](https://zxygentoo.github.io/the-ninety-percent-list/02.Kolmogorov.Complexity.and.Algorithmic.Randomness.html)'s complexity, back as a prior.
""")

# ╔═╡ e5c4ba91-0cdd-5ba6-ba9a-eddc7e822f8d
# 20 trainings, about 2 minutes.
SCARCE = map((30, 50)) do n
    data = dataset(Xoshiro(4), n)
    (; n, data,
       gaussian = [train_noisy_net(Xoshiro(s), data; prior = gaussian_prior()) for s in 1:5],
       mixture = [train_noisy_net(Xoshiro(s), data; prior = mixture_prior()) for s in 1:5])
end;

# ╔═╡ fb39e911-5cb0-5cc2-ae15-c4563a6f2f45
DECAY_SCARCE = cross_validated(SCARCE[1].data);

# ╔═╡ d4830351-0e00-5b5e-9204-8da9bf420a48
"""
Keep training a cloud at full weight cost, with its prior swapped for `prior`.
With `free_means = false`, the means stay where they are, and only the widths, the misfit code and the prior move.
"""
function refit(θ, data, prior; free_means = true, steps = 6000, rng = Xoshiro(3))
    θ = ComponentArray(μ = θ.μ, ρ = θ.ρ, log_noise = θ.log_noise, prior = prior)
    keep = fill!(similar(θ), 1.0)
    free_means || (keep.μ .= 0)
    held(θ) = keep .* θ .+ (1 .- keep) .* Zygote.dropgrad(θ)
    draws() = [noise_like(rng, θ.μ) for _ in 1:SAMPLES]
    train((θ, t) -> objective(held(θ), Zygote.ignore_derivatives(draws), 1.0, true, data), θ; steps, lr = 3e-3, cooldown = 1000)
end

# ╔═╡ 8e55a733-7d7b-53ff-ab1f-b30e93b7dbbd
"The truth under a code: its means fixed at the true weights, and everything else refitted."
function price_truth(data, prior)
    μ = truth_means()
    start = ComponentArray(μ = μ, ρ = fill!(similar(μ), log(1e-2)), log_noise = log(0.5), prior = prior)
    refit(start, data, prior; free_means = false)
end

# ╔═╡ b5b82864-abe3-5ec4-9b0f-6d18ad2af047
# Six refits, about half a minute.
TRUTH_PRICED = map([(30, SCARCE[1].data), (50, SCARCE[2].data), (num_of_cases(TRAIN), TRAIN)]) do (n, data)
    (; n, data, gaussian = price_truth(data, gaussian_prior()), mixture = price_truth(data, mixture_prior()))
end;

# ╔═╡ 37c8db24-5ecc-51b5-9b61-0ae7f40458a0
# Six refits, about half a minute.
FREED = let r = TRUTH_PRICED[1]
    (gaussian = [refit(r.gaussian, r.data, r.gaussian.prior; rng = Xoshiro(10 + s)) for s in 1:3],
     mixture = [refit(r.mixture, r.data, r.mixture.prior; rng = Xoshiro(10 + s)) for s in 1:3])
end;

# ╔═╡ f1ccd75e-bfdc-5ad2-a02b-c468a14eb0fc
"The cloud's prediction: the network's output averaged over draws of the weights."
predictive_mean(θ, x; draws = 500, rng = Xoshiro(7)) = mean(_ -> predict(sample(θ, noise_like(rng, θ.μ)), x), 1:draws)

# ╔═╡ 077df9e7-ca77-530f-8d6f-931f9e26498c
cloud_error(θ, data) = relative_error(predictive_mean(θ, data.x), data)

# ╔═╡ 84aacc97-52cc-5221-978e-874554ea41a3
"The hidden units whose outgoing weight, and at least one incoming weight, have a mean larger than its width."
function active_units(θ)
    keep = abs.(θ.μ) .>= widths(θ)
    filter(h -> keep.layer_2.weight[1, h] && any(keep.layer_1.weight[h, :]), 1:HIDDEN)
end

# ╔═╡ 21f5451d-01bf-567d-982b-4517de423903
GRID = range(-2.5, 2.5; length = 121)

# ╔═╡ 6b41f06e-f343-5728-817e-eae08696a8e5
# Every pixel of the plane as a test input: (x₁, x₂) from the grid, and fresh distractors.
PLANE = let pixels = vec([(a, b) for a in GRID, b in GRID])
    vcat(first.(pixels)', last.(pixels)', randn(Xoshiro(9), INPUTS - 2, length(pixels)))
end;

# ╔═╡ 3b314357-4a17-5bde-8170-310d7f2ba3ee
on_plane(ŷ) = reshape(ŷ, length(GRID), length(GRID))

# ╔═╡ 0d0339c0-c1ab-5f51-8b5b-e72219150fe7
gauss(w, m, s) = exp(-(w - m)^2 / (2s^2)) / (sqrt(2π) * s)

# ╔═╡ fc1f13fd-db8c-514a-bbc7-a9234145f6d5
"`x` with `d` decimals."
fixed(x, d = 1) = Printf.format(Printf.Format("%.$(d)f"), x)

# ╔═╡ 75189faf-163a-5b1a-87be-a6743c1e23b3
Markdown.parse("""
The map is speckled like TV static. The network fitted the distractors along with the signal, so fresh distractors at every pixel scramble its answer. Its relative test error is $(fixed(relative_error(predict(PLAIN, TEST.x), TEST), 3)), where the truth itself would score $(fixed(FLOOR, 3)).

How far does this go? Give it the same inputs with targets that are pure noise, standard Gaussian numbers with nothing to find, and train it the same way:
""")

# ╔═╡ ce58e9b6-6f20-5b53-a16c-d43e0eb26801
let err = relative_error(predict(MEMORIZER, NOISE_TARGETS.x), NOISE_TARGETS)
    Markdown.parse("""
    Its training error on the noise is $(fixed(err, 3)). It has memorized $(num_of_cases(NOISE_TARGETS)) random numbers.

    That settles what COMP would say about this model. It is Digits again, one level up. Digits fits any string of ``n`` bits perfectly, so its fit costs 0 and its COMP is ``n``: the whole cost of sending the string as it is. A model that can fit any targets is the same. Its fit costs almost nothing, and its COMP is about what the raw targets cost. Targets that can be any real numbers take infinitely many bits to send exactly, and for them COMP is infinite. [Zhang et al. (2017)](https://arxiv.org/abs/1611.03530), item 24 on our list, made the same observation famous for real networks.

    So COMP fails twice: it can't be computed in time, and if it could, it wouldn't help. It is a price on the model, the same for every member, so only the fit decides, and the best fit is memorizing.

    The question has shifted. Notebook 03 asked which model explains a string best. Here we want the member, the setting of the weights, that predicts new cases best, and that needs a price on each member for what it actually learned. The paper charges for exactly that, the bits in the weights after training, and makes the charge its loss.
    """)
end

# ╔═╡ 4a8c2ead-7277-50d5-a17a-08aa9fff050e
Markdown.parse("""
For our $(num_of_cases(TRAIN)) training targets, that is $(fixed(no_network(TRAIN))) bits, plus the tolerance term.

Measured against no network, a network is an investment. On the misfits it *saves* bits, because they are smaller than the targets' own spread. On its weights it *spends* bits. It is worth sending only if it saves more than it spends, and its net saving is how much shorter it makes the description.

The paper's claim compares the weights' information with the targets', but the targets' count includes the tolerance term, which grows without limit as the tolerance gets finer. A fairer measure for the weights is what they save. Read this way, the claim is a ledger: the larger a network's net saving, the better it should generalize. The intuition is the paper's own. Weights that hold few bits can't have memorized much, so what they hold must be structure the cases share, and structure carries over to new cases.
""")

# ╔═╡ d1c1a326-d00d-5bfd-a564-a4ffb65134cd
Markdown.parse("""
Weight decay is calmer than the plain network, but blurred, and still speckled by the distractors. It is close to a tilted plane, high at the top of the map and low at the bottom, and the two plateaus at 0 are missing. The paper saw the same (§9): with small weights, "the hidden units operate in their central linear range, so the whole network is effectively linear".

| | relative test error |
|:--|--:|
| no penalty | $(fixed(relative_error(predict(PLAIN, TEST.x), TEST), 3)) |
| weight decay, ``\\lambda`` by cross-validation | $(fixed(relative_error(predict(DECAY.net, TEST.x), TEST), 3)) |
| the truth itself | $(fixed(FLOOR, 3)) |
""")

# ╔═╡ 22127acf-0b2e-5b6e-ba2e-d39e003c78ee
let runs = sort([(s = s, l = ledger(θ, TRAIN), units = length(active_units(θ)), err = cloud_error(θ, TEST)) for (s, θ) in enumerate(ONE_GAUSSIAN)]; by = r -> -r.l.net)
    rows = ["| $(r.s) | $(fixed(r.l.saved)) | $(fixed(r.l.spent)) | $(fixed(r.l.net)) | $(r.units) | $(fixed(r.err, 3)) |" for r in runs]
    Markdown.parse("""
    Here is each start's ledger, in bits against no network, best first. A hidden unit counts as *used* if its outgoing weight and at least one incoming weight have means larger than their widths.

    | start | saved on the misfits | spent on the weights | net saving | hidden units used | relative test error |
    |:-:|--:|--:|--:|:-:|--:|
    $(join(rows, "\n"))

    Start $(first(runs).s) saves the most, and we keep it.
    """)
end

# ╔═╡ eeb4827a-f390-5f76-8556-b2aeb465afba
let θ = BEST_GAUSSIAN
    units = active_units(θ)
    rows = ["| unit $h | $(fixed(θ.μ.layer_1.weight[h, 1], 2)) | $(fixed(θ.μ.layer_1.weight[h, 2], 2)) | $(fixed(θ.μ.layer_1.bias[h], 2)) | $(fixed(θ.μ.layer_2.weight[1, h], 2)) |" for h in units]
    Markdown.parse("""
    It works. The best start uses $(length(units)) hidden units, and their weights are the truth's two ridges:

    | | ``x_1`` | ``x_2`` | bias | out |
    |:--|--:|--:|--:|--:|
    $(join(rows, "\n"))
    | the truth, first ridge | 2 | 1 | 0 | 1 |
    | the truth, second ridge | −1 | 2 | 0 | 1 |

    The second ridge's minus sign has moved inside, since ``\\tanh`` is odd: ``-\\tanh(x_1 - 2x_2) = \\tanh(-x_1 + 2x_2)``.

    To predict, the cloud averages its outputs over 500 draws of the weights. That way, its relative test error is $(fixed(cloud_error(θ, TEST), 3)), against $(fixed(relative_error(predict(DECAY.net, TEST.x), TEST), 3)) for weight decay, and nothing was held out to get it.

    Here it is on the plane:
    """)
end

# ╔═╡ 5f01405a-c452-5604-8424-569e5925454b
let runs = [(s = s, l = ledger(θ, TRAIN), err = cloud_error(θ, TEST)) for (s, θ) in enumerate(ONE_GAUSSIAN)]
    best, frugal = argmax(r -> r.l.net, runs), argmin(r -> r.l.spent, runs)
    Markdown.parse("""
    Now read the table's net savings against its test errors. Sorted by net saving, the starts' test errors fall into line, up to a near-tie between the top two. That is the claim, across five networks trained on the same cases: the more a network's weights save beyond what they cost, the better it generalizes.

    Spending little is not enough on its own. Start $(frugal.s) spends only $(fixed(frugal.l.spent)) bits on its weights, against $(fixed(best.l.spent)) for start $(best.s), but it fits badly: it saves only $(fixed(frugal.l.saved)) bits on the misfits, and its test error is $(fixed(frugal.err, 3)). What counts is the difference between the two, the net saving, and that is what the loss maximizes.
    """)
end

# ╔═╡ a506b9d7-5a85-549a-bd26-76c9fa947a09
let l = ledger(NOISE_NET, NOISE_TARGETS), ŷ = vec(predictive_mean(NOISE_NET, NOISE_TARGETS.x)), y = vec(NOISE_TARGETS.y)
    Markdown.parse("""
    The best of them uses $(length(active_units(NOISE_NET))) hidden units, and its weights cost $(fixed(l.spent, 2)) bits. Its predictions barely move: across the $(num_of_cases(NOISE_TARGETS)) inputs they spread by $(fixed(std(ŷ), 3)), against the targets' own spread of $(fixed(std(y), 2)), and they sit at about the targets' mean. Its net saving, $(fixed(l.net, 2)) bits, says the same: it is no network at all, give or take half a bit.

    That is the right answer. Both networks saw the same $(num_of_cases(NOISE_TARGETS)) numbers, and neither held any out. The plain network memorized them, because nothing in its loss charged for it. This one priced memorizing them, found the price higher than the saving, and declined. Held-out data would have exposed the plain network afterwards; the ledger caught the problem during training, from the training cases alone.
    """)
end

# ╔═╡ 009641e7-3f55-55bd-969d-c7bb9d207c8a
let l = ledger(BEST_GAUSSIAN, TRAIN), units = active_units(BEST_GAUSSIAN)
    Markdown.parse("""
    ### Where the bits go

    Back to the real targets. The best one-Gaussian network carries the signal in $(3 * length(units)) weights: in each of its $(length(units)) used units, one from ``x_1``, one from ``x_2``, and one to the output. Yet its weights spend $(fixed(l.spent)) bits, $(round(Int, 100 * l.spent / l.saved))% of what they save. The net saving is large, but these are not the lean weights the claim pictures. Where do the bits go?

    Each used unit also has 20 weights coming in from the distractors, and the truth wants every one of them at a precise 0. A distractor weight with a wide cloud adds noise to its unit on every case; twenty of them add up, and the unit's signal drowns. So each must be held much closer to 0 than the prior is wide, and that narrowing is what the KL charges for:
    """)
end

# ╔═╡ 0dcac29e-382b-525a-a70f-6d04979abae5
let rows = map(SCARCE) do (; n, data, gaussian, mixture)
        g, s = ledger(shortest(gaussian, data), data), ledger(shortest(mixture, data), data)
        collapsed(θs) = count(θ -> ledger(θ, data).spent < 1, θs)
        "| $n | $(fixed(g.net)) ($(collapsed(gaussian)) of 5 collapsed) | $(fixed(s.net)) ($(collapsed(mixture)) of 5 collapsed) |"
    end
    m50 = shortest(SCARCE[2].mixture, SCARCE[2].data)
    Markdown.parse("""
    | cases | one Gaussian, best start | mixture, best start |
    |--:|--:|--:|
    $(join(rows, "\n"))
    | $(num_of_cases(TRAIN)) | $(fixed(ledger(BEST_GAUSSIAN, TRAIN).net)) | $(fixed(ledger(BEST_MIXTURE, TRAIN).net)) |

    Net savings in bits. A start has *collapsed* when its weights cost less than 1 bit: it predicts the targets' mean, which is no network at all.

    At 30 cases, every start of both codes collapses. At 50, one Gaussian still collapses every time, and the mixture gets only partway: its best start uses $(length(active_units(m50))) hidden unit, reading ``x_2`` alone, and scores $(fixed(cloud_error(m50, TEST), 3)) on the test set. The paper met the same thing. The shortest description it found "was the solution in which all the weights except the output bias are equal and negative", with a relative error of about 1, and it called this "a serious embarrassment for the Minimum Description Length Principle or for our method of describing the weights".

    There are three suspects: MDL itself, the code, or the search. We can tell them apart, because we know the truth. Fix the means at the true weights, refit everything else, and see what the truth saves under each code:
    """)
end

# ╔═╡ 80694af1-403b-57e3-b836-5cff96c5c3e3
let trained = [[ledger(shortest(r.mixture, r.data), r.data).net for r in SCARCE]; ledger(BEST_MIXTURE, TRAIN).net]
    rows = map(zip(TRUTH_PRICED, trained)) do (r, best)
        "| $(r.n) | $(fixed(ledger(r.gaussian, r.data).net)) | $(fixed(ledger(r.mixture, r.data).net)) | $(fixed(best)) |"
    end
    Markdown.parse("""
    | cases | the truth, one Gaussian | the truth, mixture | mixture, best trained |
    |--:|--:|--:|--:|
    $(join(rows, "\n"))

    Net savings in bits. Under the mixture, the truth saves bits even at 30 cases; training just never found it. Under one Gaussian, at 30 and 50 cases, the truth costs more than it saves.

    At 100 cases, the trained networks beat the truth: the mixture's by a little, and one Gaussian's, at $(fixed(ledger(BEST_GAUSSIAN, TRAIN).net)) bits, by a lot. The mixture's gap is innocent. The truth's means are held at whole numbers, and this sample of noise prefers slightly different ones. One Gaussian's gap is the leak from *Where the bits go*: its trained network fits a little of the noise through its distractor weights, which saves bits on the training cases and costs accuracy on new ones. Under that code, a slightly overfit network is a shorter description than the truth.

    The means were held still, though. Free them too, starting from the priced truth at 30 cases, and train on, three times each with fresh noise:
    """)
end

# ╔═╡ 3f895d6d-2c7a-58c1-924a-ae9830aca097
"The range of `values` with `d` decimals, or a single number when its ends agree."
between(values, d = 1) = let (lo, hi) = extrema(values)
    fixed(lo, d) == fixed(hi, d) ? fixed(lo, d) : "$(fixed(lo, d)) to $(fixed(hi, d))"
end

# ╔═╡ 301dd0cc-6bfa-5964-b6b7-a914f4bb455d
let data = TRUTH_PRICED[1].data
    span(θs) = between([ledger(θ, data).net for θ in θs])
    errors(θs) = between([cloud_error(θ, TEST) for θ in θs], 3)
    Markdown.parse("""
    - **One Gaussian walks away from the truth**, to the trivial network: a net saving of $(span(FREED.gaussian)) bits, with test error $(errors(FREED.gaussian)).
    - **The mixture stays**, and improves to a net saving of $(span(FREED.mixture)) bits, with test error $(errors(FREED.mixture)). Weight decay, on the same $(num_of_cases(data)) cases, gets $(fixed(relative_error(predict(DECAY_SCARCE.net, TEST.x), TEST), 3)).

    So the verdicts split.

    - **One Gaussian: the code.** Even the truth costs more than it saves, and started there, training walks away.
    - **The mixture: the search.** A description that saves bits exists, and it is stable once found, but no random start reaches it. This is the trap the warm-up was meant to avoid: weights that hold almost nothing are easy to reach, and hard to leave.
    - **MDL itself is acquitted**, here. With a good code, the shortest description has the truth's structure.
    """)
end

# ╔═╡ cec8f3c9-b875-5e81-aa2c-df88ac7ae07a
"A whole number, with commas between the thousands."
commas(n::Integer) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => ",")

# ╔═╡ 4ab48f39-82fd-5a3b-8c65-160eb701dc29
let rows = ["| $(r.n) | $(commas(2^r.n)) | $(fixed(r.comp, 2)) | $(fixed(r.seconds, 3)) s |" for r in COMP_TIMINGS]
    slowest = last(COMP_TIMINGS)
    years = slowest.seconds * 2.0^(100 - slowest.n) / (365.25 * 24 * 3600)
    Markdown.parse("""
    | bits ``n`` | strings | COMP (bits) | time |
    |--:|--:|--:|--:|
    $(join(rows, "\n"))

    Each extra bit doubles the work. At this rate, 100 bits would take about ``10^{$(round(Int, log10(years)))}`` years, some ``10^{$(round(Int, log10(years / 1.38e10)))}`` times the age of the universe. Notebook 03 stopped at 20 bits for this reason.

    Bernoulli does have a way out. Its best fit depends on a string only through its count of 1s, so all the strings with the same count can be summed at once: ``n + 1`` terms instead of ``2^n``.

    A network has no such way out. Its fit doesn't come down to a few counts, so computing its COMP takes one training run per dataset. And its targets are real numbers, so the sum becomes an integral over every possible set of targets. Quantize them to a fine tolerance within some range, and the list is finite again, but it grows exponentially with the number of cases, with a far larger base than 2.
    """)
end

# ╔═╡ 947ec45d-2a0e-545e-8c2c-9d41313272af
let n = length(weights(Xoshiro(0)))
    Markdown.parse("""
    That is $n weights for $(num_of_cases(TRAIN)) training cases, with $(commas(num_of_cases(TEST))) more held out for testing. Here is the truth on the ``(x_1, x_2)`` plane, with the training cases on it:
    """)
end

# ╔═╡ 7afb28ea-f81f-5845-9893-09bd63657b65
VIZ = (surface = "#fcfcfb", text = "#0b0b0b", text2 = "#52514e", grid = "#e4e3de")

# ╔═╡ 8ce757c7-3507-5b82-9b64-31a218a2b79b
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# ╔═╡ 04b5296d-d362-5623-9a88-448c87f1f18d
# Blue to red through a neutral grey, for values around 0.
DIVERGING = cgrad([colorant"#184f95", colorant"#5598e7", colorant"#f0efec", colorant"#e34948", colorant"#9c2524"])

# ╔═╡ 386a3f78-ce6a-5434-ab8b-e9fa428aa9a5
# One blue ramp, light to dark, for magnitudes.
SEQUENTIAL = cgrad([colorant"#fcfcfb", colorant"#cde2fb", colorant"#86b6ef", colorant"#3987e5", colorant"#1c5cab", colorant"#0d366b"])

# ╔═╡ f85c9643-35bd-5663-9f04-bab99f8d2154
THEME = Theme(
    fontsize = 13, backgroundcolor = VIZ.surface, textcolor = VIZ.text,
    Axis = (backgroundcolor = VIZ.surface, xgridcolor = VIZ.grid, ygridcolor = VIZ.grid,
            topspinevisible = false, rightspinevisible = false, leftspinecolor = VIZ.text2, bottomspinecolor = VIZ.text2,
            xtickcolor = VIZ.text2, ytickcolor = VIZ.text2, xticklabelcolor = VIZ.text2, yticklabelcolor = VIZ.text2,
            titlealign = :left, titlefont = :regular, titlesize = 14),
    Legend = (framevisible = false, backgroundcolor = VIZ.surface),
);

# ╔═╡ 149bdecd-7728-58b7-b78e-d9f7d1e2871c
with_theme(THEME) do
    fig = Figure(size = (760, 320))
    ax = Axis(fig[1, 1]; xlabel = "value", ylabel = "density", title = "A value's probability: the density's height, times its bin's width")
    w = range(-3, 3; length = 601)
    lines!(ax, w, gauss.(w, 0, 1); color = SERIES_COLORS[1], linewidth = 2)
    v, t = 1.6, 0.3
    h = gauss(v, 0, 1)
    poly!(ax, Rect(v - t / 2, 0, t, h); color = (SERIES_COLORS[1], 0.25), strokecolor = SERIES_COLORS[1], strokewidth = 1)
    text!(ax, v - t / 2, h / 2; text = "height P(x)", align = (:right, :center), offset = (-6, 0), color = VIZ.text2)
    text!(ax, v + t / 2, h; text = "width t", align = (:left, :bottom), offset = (6, 2), color = VIZ.text2)
    ylims!(ax, 0, 0.47)
    fig
end

# ╔═╡ a1887041-35ed-5e9e-a653-25fcc1935d40
plane_axis(position, title) = Axis(position; title, aspect = 1, xlabel = "x₁", ylabel = "x₂", limits = (extrema(GRID), extrema(GRID)))

# ╔═╡ d3f55395-d4ae-5450-9b7d-901134745ecb
"A map of `z` over the plane, with the training cases as rings."
function plane_map!(ax, z; colormap = DIVERGING, colorrange = (-2.2, 2.2))
    hm = heatmap!(ax, GRID, GRID, z; colormap, colorrange)
    scatter!(ax, TRAIN.x[1, :], TRAIN.x[2, :]; color = :transparent, strokecolor = VIZ.text, strokewidth = 1, markersize = 6)
    hm
end

# ╔═╡ 4d023645-bc11-59f7-a488-3169674ff70d
with_theme(THEME) do
    fig = Figure(size = (470, 420))
    ax = plane_axis(fig[1, 1], "The truth, and the $(num_of_cases(TRAIN)) training cases")
    hm = plane_map!(ax, on_plane(truth.(PLANE[1, :], PLANE[2, :])))
    Colorbar(fig[1, 2], hm; label = "f(x₁, x₂)")
    fig
end

# ╔═╡ 07e1afbe-1435-5366-a807-d9d8c1976008
with_theme(THEME) do
    fig = Figure(size = (470, 420))
    hm = plane_map!(plane_axis(fig[1, 1], "No penalty"), on_plane(predict(PLAIN, PLANE)))
    Colorbar(fig[1, 2], hm; label = "prediction")
    fig
end

# ╔═╡ d98a33f9-22fb-5a73-86d1-e560e7cf3ac7
with_theme(THEME) do
    fig = Figure(size = (470, 420))
    hm = plane_map!(plane_axis(fig[1, 1], "Weight decay, λ = $(round(DECAY.λ; sigdigits = 2))"), on_plane(predict(DECAY.net, PLANE)))
    Colorbar(fig[1, 2], hm; label = "prediction")
    fig
end

# ╔═╡ df7eb95a-cb0d-56ce-85d8-09b582fc2d9f
with_theme(THEME) do
    fig = Figure(size = (470, 420))
    hm = plane_map!(plane_axis(fig[1, 1], "One Gaussian: the mean prediction"), on_plane(predictive_mean(BEST_GAUSSIAN, PLANE)))
    Colorbar(fig[1, 2], hm; label = "prediction")
    fig
end

# ╔═╡ e20c620c-1bb9-5b36-a2b6-6e5b90de84ac
with_theme(THEME) do
    fig = Figure(size = (900, 420))
    hm = plane_map!(plane_axis(fig[1, 1], "The truth"), on_plane(truth.(PLANE[1, :], PLANE[2, :])))
    plane_map!(plane_axis(fig[1, 2], "Mixture of Gaussians: the mean prediction"), on_plane(predictive_mean(BEST_MIXTURE, PLANE)))
    Colorbar(fig[1, 3], hm; label = "value")
    fig
end

# ╔═╡ b0455b32-4928-5765-b48c-74cf5c3cbd78
with_theme(THEME) do
    maps = [
        ("The truth", truth.(PLANE[1, :], PLANE[2, :]), FLOOR),
        ("No penalty", predict(PLAIN, PLANE), relative_error(predict(PLAIN, TEST.x), TEST)),
        ("Weight decay", predict(DECAY.net, PLANE), relative_error(predict(DECAY.net, TEST.x), TEST)),
        ("One Gaussian", predictive_mean(BEST_GAUSSIAN, PLANE), cloud_error(BEST_GAUSSIAN, TEST)),
        ("Mixture of Gaussians", predictive_mean(BEST_MIXTURE, PLANE), cloud_error(BEST_MIXTURE, TEST)),
    ]
    fig = Figure(size = (900, 640))
    hms = map(enumerate(maps)) do (i, (title, z, err))
        plane_map!(plane_axis(fig[fldmod1(i, 3)...], "$title: $(fixed(err, 3))"), on_plane(z))
    end
    Colorbar(fig[2, 3], last(hms); label = "prediction", vertical = false, width = Relative(0.8), tellheight = false)
    fig
end

# ╔═╡ 12dfbff1-23b7-5764-9f3b-c011780ccd70
"A mixture's shares, πₖ."
shares(prior) = exp.(prior.logit .- logsumexp(prior.logit))

# ╔═╡ 387c10aa-faf0-5008-8cfc-9c96abd00ff1
"The share of a cloud's narrowest prior component: the spike."
spike_share(θ) = shares(θ.prior)[argmin(exp.(θ.prior.log_σ))]

# ╔═╡ d78a68bb-5df1-5bb6-bdb8-72f4bea340ab
let runs = [(l = ledger(θ, TRAIN), units = length(active_units(θ)), share = spike_share(θ), err = cloud_error(θ, TEST)) for θ in MIXTURE]
    span(group) = between([r.l.net for r in group])
    best = argmax(r -> r.l.net, runs)
    split_ridge = filter(r -> r.units > 2, runs)
    loose = filter(r -> r.units == 2 && r.share < 0.9, runs)
    Markdown.parse("""
    The other starts landed elsewhere. $(length(split_ridge)) of them split one ridge across two units, for a net saving of $(span(split_ridge)) bits. $(length(loose)) found the two ridges again, under a looser mixture whose spike has only a $(round(Int, 100 * minimum(r.share for r in loose))) to $(round(Int, 100 * maximum(r.share for r in loose)))% share, so pointing at it costs more: a net saving of $(span(loose)). The best saves $(fixed(best.l.net)).

    All five score $(between([r.err for r in runs], 3)) on the test set, so here the claim has nothing to predict: all five generalize alike. The ledger still ranks them, by simplicity alone, and its favourite is the simplest: the truth's two units, under the tightest spike.
    """)
end

# ╔═╡ 77df1761-4ebb-5a61-ae36-121f3b299699
ROW_NAMES = ["x₁"; "x₂"; ["d$i" for i in 1:INPUTS-2]; "bias"; "out"]

# ╔═╡ 9afeec6e-205b-5197-b586-6b5cab85d54f
"Bits per weight: hidden units, by their inputs, their bias, and their outgoing weight."
function bit_table(θ)
    first_layer = bits.(costs(θ.μ.layer_1.weight, widths(θ).layer_1.weight, θ.prior, :layer_1))
    bias = bits.(costs(θ.μ.layer_1.bias, widths(θ).layer_1.bias, θ.prior, :layer_1))
    out = bits.(costs(vec(θ.μ.layer_2.weight), vec(widths(θ).layer_2.weight), θ.prior, :layer_2))
    hcat(first_layer, bias, out)
end

# ╔═╡ 54337ec5-d8f6-5ad1-a077-6c1e1ded0573
"What the zeros cost: the distractor weights into the used hidden units, in bits, with their clouds' mean width."
function zero_bill(θ)
    units = active_units(θ)
    b = bit_table(θ)[units, 3:INPUTS]
    (; count = length(b), total = sum(b), low = minimum(b), high = maximum(b), width = mean(widths(θ).layer_1.weight[units, 3:INPUTS]))
end

# ╔═╡ 103fd665-1073-5fd8-9c20-3a82f23ca479
let z = zero_bill(BEST_GAUSSIAN), σₚ = exp(BEST_GAUSSIAN.prior.layer_1), l = ledger(BEST_GAUSSIAN, TRAIN)
    Markdown.parse("""
    - There are $(z.count) of them, held to about ``\\pm $(fixed(z.width, 3))``, under a prior of width ``\\sigma_P = $(fixed(σₚ, 2))``.
    - By the KL formula, a zero held that tightly costs about ``\\log_2(\\sigma_P / \\sigma) - \\tfrac12 \\log_2 e``, or $(fixed(bits(kl(0, z.width, σₚ)))) bits. The actual costs run from $(fixed(z.low)) to $(fixed(z.high)) bits each.
    - Together they cost $(fixed(z.total)) bits: $(round(Int, 100 * z.total / l.spent))% of the $(fixed(l.spent)) the weights spend.

    Here is the whole bill, weight by weight:
    """)
end

# ╔═╡ 280abba4-d2a3-53de-bc26-154763b473f5
Markdown.parse("""
In the two used units, the rows for ``x_1``, ``x_2`` and the output carry the signal. Between them, the distractor rows glow faintly, weight after weight, and that faint glow, $(zero_bill(BEST_GAUSSIAN).count) times over, is most of the bill. The two units' biases are precise zeros too, and glow the same. The output row glows across the idle units as well, for the same reason. Their incoming clouds are as wide as the prior, which costs nothing, so their outputs are pure noise, and each output weight must hold a precise 0 to keep it out: one precise zero instead of $(INPUTS + 1).

The zeros also cost accuracy. One Gaussian charges for their widths but hardly pulls in their means. The pull is the KL's ``\\mu^2 / 2\\sigma_P^2`` term, and for means of a few hundredths under a prior of width $(fixed(exp(BEST_GAUSSIAN.prior.layer_1), 2)), it is almost nothing. So the zeros sit a little off 0, fitting a little of the training noise, and fresh distractors leak through. That leak is the speckle on the map, and it keeps the test error, $(fixed(cloud_error(BEST_GAUSSIAN, TEST), 3)), above the floor of $(fixed(FLOOR, 3)).

The trouble is the prior, not the network. One Gaussian must cover the weights near 2 and the weights at 0 with a single width, and it can't be narrow for the zeros without being narrow for the rest. The bill shows both sides. Its darkest cell, at $(fixed(maximum(bit_table(BEST_GAUSSIAN)))) bits, is the weight of about 2 from ``x_1``, and most of its cost is the ``\\mu^2`` term, for sitting more than four prior widths from 0. The zeros pay the opposite price, for being far narrower than the prior. The paper saw this coming (§6.1): "no single Gaussian prior can provide a good model of a spike around 0 and a spike around 1."
""")

# ╔═╡ ccb5fc61-c16c-5237-b9de-a764149d50a5
let g = BEST_GAUSSIAN, θ = BEST_MIXTURE
    lg, l = ledger(g, TRAIN), ledger(θ, TRAIN)
    zg, z = zero_bill(g), zero_bill(θ)
    σs, πs = exp.(θ.prior.log_σ), shares(θ.prior)
    spike = argmin(σs)
    each(z) = "$(fixed(z.low, 2))–$(fixed(z.high, 2))"
    Markdown.parse("""
    The best start learns a spike of width $(fixed(σs[spike], 3)) with a $(round(Int, 100 * πs[spike]))% share, and a wide component of width $(fixed(σs[3 - spike], 2)) for the rest. Side by side with one Gaussian, in bits against no network:

    | | one Gaussian | mixture of Gaussians |
    |:--|--:|--:|
    | each zero | $(each(zg)) | $(each(z)) |
    | all $(z.count) zeros | $(fixed(zg.total)) | $(fixed(z.total)) |
    | spent on the weights | $(fixed(lg.spent)) | $(fixed(l.spent)) |
    | saved on the misfits | $(fixed(lg.saved)) | $(fixed(l.saved)) |
    | net saving | $(fixed(lg.net)) | $(fixed(l.net)) |
    | relative test error | $(fixed(cloud_error(g, TEST), 3)) | $(fixed(cloud_error(θ, TEST), 3)) |

    A zero now costs a fraction of a bit. The mixture spends $(round(Int, 100 * l.spent / lg.spent))% as much on its weights and saves more on the misfits, and its test error is next to the truth's own, $(fixed(FLOOR, 3)). Its weights now cost $(round(Int, 100 * l.spent / l.saved))% of what they save. The ledger and the claim agree again.

    The bill, on the same scale as before:
    """)
end

# ╔═╡ 0dedf7d3-174f-567c-a37f-f461d954ff28
let z = zero_bill(BEST_MIXTURE), zg = zero_bill(BEST_GAUSSIAN)
    Markdown.parse("""
    The distractor rows have gone dark, but not because their clouds are wide. They are about $(fixed(z.width, 3)) wide: as narrow as the spike, and narrower than under one Gaussian ($(fixed(zg.width, 3))). They are cheap because they look like the prior, as the KL formula said. A prior that expects most weights to be almost exactly 0 makes an exact 0 nearly free, and pulls the zeros' means in too, so the leak is gone.

    And the network on the plane, beside the truth:
    """)
end

# ╔═╡ be298eb2-41b5-5448-9b78-c0f3e0d252c3
let decay = relative_error(predict(DECAY.net, TEST.x), TEST)
    Markdown.parse("""
    ## Looking back

    The paper claimed that a network generalizes when its weights hold much less information than its targets, so it should be trained on the expected squared error plus the information in its weights.

    On our task, the claim held. Trained on that loss, with nothing held out, the network beat weight decay tuned by cross-validation: $(fixed(cloud_error(BEST_GAUSSIAN, TEST), 3)) with one Gaussian, and $(fixed(cloud_error(BEST_MIXTURE, TEST), 3)) with a mixture of Gaussians, against $(fixed(decay, 3)). Among the networks training found, those whose weights saved the most beyond their cost generalized best, and on pure noise the loss sent nothing. How much a network saves depends on the prior it is coded with: one Gaussian paid $(fixed(zero_bill(BEST_GAUSSIAN).low)) bits or more for every zero, and a mixture of Gaussians said zero almost for free. With scarce data, the search gave out before MDL did.

    So how many bits does a network need? Not the 64 a computer stores for each weight: the mixture network sent all $(length(BEST_MIXTURE.μ)) of its weights in $(fixed(ledger(BEST_MIXTURE, TRAIN).spent)) bits, net of the refund, most of them in the six that carry the ridges.

    Here is every network in this notebook on the plane, each titled with its relative test error:
    """)
end

# ╔═╡ b845009d-ef89-598a-a612-39e96be14606
# One colour scale for both bills.
BITS_LIMIT = maximum(maximum ∘ bit_table, (BEST_GAUSSIAN, BEST_MIXTURE))

# ╔═╡ adf3104b-6068-5973-a5c9-c8b315f1fcd8
"Bits per weight as a heatmap: hidden units across, inputs down."
function bits_chart(table, title)
    with_theme(THEME) do
        fig = Figure(size = (470, 500))
        ax = Axis(fig[1, 1]; title, xlabel = "hidden unit", yreversed = true, xticks = 1:HIDDEN,
                  yticks = (1:length(ROW_NAMES), ROW_NAMES), yticklabelsize = 10)
        hm = heatmap!(ax, 1:HIDDEN, 1:length(ROW_NAMES), table; colormap = SEQUENTIAL, colorrange = (0, BITS_LIMIT))
        Colorbar(fig[1, 2], hm; label = "bits")
        fig
    end
end

# ╔═╡ 895efa0c-63f2-561e-abd4-19742c36dc32
bits_chart(bit_table(BEST_GAUSSIAN), "One Gaussian: bits per weight")

# ╔═╡ ee7073ef-2ccc-585b-be04-b6ea7a6c4c61
bits_chart(bit_table(BEST_MIXTURE), "Mixture of Gaussians: bits per weight")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
CairoMakie = "~0.15.17"
ComponentArrays = "~0.15.50"
Lux = "~1.31.5"
Optimisers = "~0.4.11"
PlutoUI = "~0.7.84"
Zygote = "~0.7.15"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.3"
manifest_format = "2.1"
project_hash = "69bd66d0fc539414e60075dcced56472d6d475f1"

[[deps.ADTypes]]
deps = ["PrecompileTools"]
git-tree-sha1 = "629de23e1c16911b439dabd2303c08af9575b226"
registries = "General"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.24.0"
weakdeps = ["ChainRulesCore", "ConstructionBase", "EnzymeCore"]

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

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
git-tree-sha1 = "8c0bc12397dabf5d2dfc4b5763ec463854cc44d3"
registries = "General"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.46"

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
git-tree-sha1 = "86d0ae34bc46373170064ef33c12a303994ab736"
registries = "General"
uuid = "4f381ef7-9af0-4cbe-99d4-cf36d7b0f233"
version = "0.3.12"

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

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "1aec1ff0dcaa83a484e7f72097562fa3102c6722"
registries = "General"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.30.2"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceAMDGPUExt = "AMDGPU"
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceFillArraysExt = "FillArrays"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceGPUArraysCoreTrackerExt = ["GPUArraysCore", "Tracker"]
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "0aa371c875e24cf773c87b2f50ab03e4545748af"
registries = "General"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.7.0"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDACore = "bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"

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

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "5f866fe1b4890785975a933a452946507660a9d6"
registries = "General"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.6.4"

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

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Preferences", "Static"]
git-tree-sha1 = "f3a21d7fc84ba618a779d1ed2fcca2e682865bab"
registries = "General"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.7"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "0db159493e83ecb2d122f9c40138d89b70d43963"
registries = "General"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.1.1"

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
deps = ["CRC32c", "Cairo", "Cairo_jll", "Colors", "FileIO", "FreeType", "FreeType2_jll", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "2d9e0c818e6e5af309192a5b3638c8848c7119b5"
registries = "General"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.15.17"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "0cfb779e24fdced77a05df1dbe99200821a9f563"
registries = "General"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.8+1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays"]
git-tree-sha1 = "7f6cff0df4f383cd200dc2608b70df1d9b1f22c9"
registries = "General"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.74.8"

[[deps.ChainRulesCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "295fa9b898db109bc56df62c0fc20bd105cec00e"
registries = "General"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.8"
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
git-tree-sha1 = "060adaf7a5c2e14bca025edf0eea0206ffdebd87"
registries = "General"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.32.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "af523e59de94de05b9b367a2ca4d63233aacd222"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.5"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "96fce5c80e0834e96ab6d24fb384fd945c9cb2d4"
registries = "General"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.2"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "0737a3e59f9d94957f5b177fa2ca40901636b85b"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.4"

[[deps.CommonSolve]]
deps = ["PrecompileTools"]
git-tree-sha1 = "6c389fa857f6ca5a95474b52a52023fd77f24cb7"
registries = "General"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.14"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
registries = "General"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.CommonWorldInvalidations]]
deps = ["PrecompileTools"]
git-tree-sha1 = "7c4ea0adfb7238bf4678aa36325863fab6163f6f"
registries = "General"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.2.2"

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

[[deps.ComponentArrays]]
deps = ["Adapt", "ArrayInterface", "ChainRulesCore", "ConstructionBase", "Functors", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrayInterface", "StaticArraysCore"]
git-tree-sha1 = "648077c72493a3ee81f5da4e5803ad1b8e05af58"
registries = "General"
uuid = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
version = "0.15.50"

    [deps.ComponentArrays.extensions]
    ComponentArraysGPUArraysExt = "GPUArrays"
    ComponentArraysKernelAbstractionsExt = "KernelAbstractions"
    ComponentArraysMooncakeExt = "Mooncake"
    ComponentArraysOptimisersExt = "Optimisers"
    ComponentArraysReactantExt = "Reactant"
    ComponentArraysRecursiveArrayToolsExt = "RecursiveArrayTools"
    ComponentArraysReverseDiffExt = "ReverseDiff"
    ComponentArraysSciMLBaseExt = ["SciMLBase", "SymbolicIndexingInterface"]
    ComponentArraysTrackerExt = "Tracker"
    ComponentArraysZygoteExt = "Zygote"

    [deps.ComponentArrays.weakdeps]
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SciMLBase = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
    SymbolicIndexingInterface = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

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
git-tree-sha1 = "b56df0c58a9d37b4a926f84f3f4f283a5987680e"
registries = "General"
uuid = "95dc2771-c249-4cd0-9c9f-1f3b4330693c"
version = "0.1.9"

[[deps.ConcreteStructs]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a72c3b5ce6d2a477f55b5c9b8756e91e695c67f6"
registries = "General"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.8"

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

[[deps.CpuId]]
deps = ["Markdown", "PrecompileTools"]
git-tree-sha1 = "5dbd5150b96f29553fc3b553c6bfb0eadf7ac204"
registries = "General"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.3"

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

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
registries = "General"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "79a2aca180a85c690c58a020d47b426954b590f8"
registries = "General"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.16.0"

[[deps.DispatchDoctor]]
deps = ["MacroTools", "Preferences"]
git-tree-sha1 = "278d0120d7c22ef36bd8d22f349868d6a782a949"
registries = "General"
uuid = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
version = "0.4.29"
weakdeps = ["ChainRulesCore", "EnzymeCore"]

    [deps.DispatchDoctor.extensions]
    DispatchDoctorChainRulesCoreExt = "ChainRulesCore"
    DispatchDoctorEnzymeCoreExt = "EnzymeCore"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "Roots", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "903fc89ebdfd518ef02fa121ec82c4ea70c62eb9"
registries = "General"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.133"

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

[[deps.EnzymeCore]]
git-tree-sha1 = "04c4269956f90d8f5c9acc73b774c5293481f4d7"
registries = "General"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.26"
weakdeps = ["Adapt", "ChainRulesCore"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"
    EnzymeCoreChainRulesCoreExt = "ChainRulesCore"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "83231673ea4d3d6008ac74dc5079e77ab2209d8f"
registries = "General"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f2659a735f9360fedfa29925812380f322679e93"
registries = "General"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.10.0+0"

[[deps.ExpressionExplorer]]
git-tree-sha1 = "fdcc9d4a253e0041d37c95487f9ae027c3ec002d"
registries = "General"
uuid = "21656369-7473-754a-2065-74616d696c43"
version = "1.1.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d9d3cd382f2c999f684e6bfa4039fcc16c153b6f"
registries = "General"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "9.0.2+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "16aa62b6b2c5c65a642155bd1bce90633b8d3130"
registries = "General"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.2"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
registries = "General"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

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
git-tree-sha1 = "831f86fde78ab7f6470d6ba4d1161a11fb6b4ec0"
registries = "General"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.18.2"
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
git-tree-sha1 = "e1e9ba3a8f5d71e6ae698f138a1362aeabb42e9f"
registries = "General"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.18.3+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
registries = "General"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a55669aa77ae8bbdc7310c4027112500ff5ec874"
registries = "General"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.4.7"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

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
git-tree-sha1 = "e894be022a2eba58463939f1bf1f7f7006f27f52"
registries = "General"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.19+0"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "416b521e1158dd93cd99b3adcd872085726b36a1"
registries = "General"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.4"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "f60436514e6d73f8ee069ae4ee0ff038fe96b69d"
registries = "General"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.2"

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
git-tree-sha1 = "f700c32c1e9e11b576592043f270e6b33868a56f"
registries = "General"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.15000.0+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
registries = "General"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

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
git-tree-sha1 = "c28dceeadc1c9c699656a3ac9d63f79cfaa17472"
registries = "General"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.31"

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

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "eb7dbaba9b60b122c6526b369291695d25e4128a"
registries = "General"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.25"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
registries = "General"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

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
weakdeps = ["ForwardDiff", "Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

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
git-tree-sha1 = "d22e8bdb3f03c300a103ff23c0dd5a654c6701de"
registries = "General"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.7"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
registries = "General"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "123311cf9834c6ee6a8347bba9af68f255615391"
registries = "General"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.11.0"

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
git-tree-sha1 = "06d90f699285d055e34f0b014636a75a17ff2252"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.13.4"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]
    JSONSerializationExt = ["Serialization"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
registries = "General"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e06b451acfb1c6fa5ea753b5042207226271f455"
registries = "General"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.1+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "e51d16f9dcff134c4ecf79e7cfa764c59d2d648e"
registries = "General"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.45"
weakdeps = ["EnzymeCore", "LinearAlgebra", "SparseArrays"]

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

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
git-tree-sha1 = "6d7c65411dda4bfa0beb7490476122b4f805cf31"
registries = "General"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.3.0+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "PrecompileTools", "Preferences", "Printf", "ScopedValues", "Unicode"]
git-tree-sha1 = "41c2ce7ec5a14c89b93dc6a5853bf2a91d7a8583"
registries = "General"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "10.4.3"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "TOML"]
git-tree-sha1 = "39def0b4521e0ef2bf26960d8965729f496bb4ec"
registries = "General"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.56+0"

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

[[deps.Lux]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "ChainRulesCore", "ConcreteStructs", "DiffResults", "DispatchDoctor", "EnzymeCore", "FastClosures", "ForwardDiff", "Functors", "GPUArraysCore", "LinearAlgebra", "LuxCore", "LuxLib", "MLDataDevices", "MacroTools", "Markdown", "NNlib", "Optimisers", "PrecompileTools", "Preferences", "Random", "ReactantCore", "Reexport", "SciMLPublic", "Setfield", "Static", "StaticArraysCore", "Statistics", "UUIDs", "WeightInitializers"]
git-tree-sha1 = "3b8f9abca8323aea081c597c11f1f15f0c5dcb04"
registries = "General"
uuid = "b2108857-7c20-44ae-9111-449ecde12c47"
version = "1.31.5"

    [deps.Lux.extensions]
    ComponentArraysExt = "ComponentArrays"
    EnzymeExt = "Enzyme"
    FluxExt = "Flux"
    GPUArraysExt = "GPUArrays"
    LossFunctionsExt = "LossFunctions"
    MLUtilsExt = "MLUtils"
    MPIExt = "MPI"
    MPINCCLExt = ["CUDA", "MPI", "NCCL"]
    MooncakeExt = "Mooncake"
    ReactantExt = ["Enzyme", "Reactant"]
    ReverseDiffExt = ["FunctionWrappers", "ReverseDiff"]
    SimpleChainsExt = "SimpleChains"
    TrackerExt = "Tracker"
    ZygoteExt = "Zygote"

    [deps.Lux.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
    FunctionWrappers = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    LossFunctions = "30fc2ffe-d236-52d8-8643-a9d8f7c094a7"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    NCCL = "3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SimpleChains = "de6bee2f-e2f4-4ec7-b6ed-219cc6f6e9e5"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LuxCore]]
deps = ["DispatchDoctor", "Random", "SciMLPublic"]
git-tree-sha1 = "a2636fa690d539c3035d248f268ca39fa9900369"
registries = "General"
uuid = "bb33d45b-7691-41d6-9220-0943567d0623"
version = "1.5.4"

    [deps.LuxCore.extensions]
    ArrayInterfaceReverseDiffExt = ["ArrayInterface", "ReverseDiff"]
    ArrayInterfaceTrackerExt = ["ArrayInterface", "Tracker"]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    FluxExt = "Flux"
    FunctorsExt = "Functors"
    MLDataDevicesExt = ["Adapt", "MLDataDevices"]
    ReactantExt = "Reactant"
    SetfieldExt = "Setfield"

    [deps.LuxCore.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ArrayInterface = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
    Functors = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
    MLDataDevices = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Setfield = "efcf1570-3423-57d1-acb7-fd33fddbac46"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.LuxLib]]
deps = ["ArrayInterface", "CPUSummary", "ChainRulesCore", "DispatchDoctor", "EnzymeCore", "FastClosures", "Functors", "KernelAbstractions", "LinearAlgebra", "LuxCore", "MLDataDevices", "Markdown", "NNlib", "Preferences", "Random", "Reexport", "SciMLPublic", "Static", "StaticArraysCore", "Statistics", "UUIDs"]
git-tree-sha1 = "6d24be90ca806a65de138456ba41f9c861b9816c"
registries = "General"
uuid = "82251201-b29d-42c6-8e01-566dec8acb11"
version = "1.15.10"

    [deps.LuxLib.extensions]
    AppleAccelerateExt = "AppleAccelerate"
    BLISBLASExt = "BLISBLAS"
    CUDAExt = "CUDA"
    CUDAForwardDiffExt = ["CUDA", "ForwardDiff"]
    EnzymeExt = "Enzyme"
    ForwardDiffExt = "ForwardDiff"
    LoopVectorizationExt = ["LoopVectorization", "Polyester"]
    MKLExt = "MKL"
    OctavianExt = ["Octavian", "LoopVectorization"]
    OneHotArraysExt = ["OneHotArrays"]
    ReactantExt = ["Reactant", "ReactantCore"]
    ReverseDiffExt = "ReverseDiff"
    SLEEFPiratesExt = "SLEEFPirates"
    TrackerAMDGPUExt = ["AMDGPU", "Tracker"]
    TrackerExt = "Tracker"
    cuDNNExt = ["CUDA", "cuDNN"]

    [deps.LuxLib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    AppleAccelerate = "13e28ba4-7ad8-5781-acae-3021b1ed3924"
    BLISBLAS = "6f275bd8-fec0-4d39-945b-7e95a765fa1e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890"
    MKL = "33e6dc65-8f57-5167-99aa-e5a354878fb2"
    Octavian = "6fd5a793-0b7e-452c-907f-f8bfe9c57db4"
    OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    ReactantCore = "a3311ec8-5e00-46d5-b541-4f83e724a433"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SLEEFPirates = "476501e8-09a2-5ece-8869-fb82de89a1fa"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MLDataDevices]]
deps = ["Adapt", "Functors", "Preferences", "Random", "SciMLPublic"]
git-tree-sha1 = "0d8704d7609a51d6f2717b5e26762dbe6e2bec7c"
registries = "General"
uuid = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
version = "1.18.0"

    [deps.MLDataDevices.extensions]
    AMDGPUExt = "AMDGPU"
    CUDAExt = "CUDA"
    ChainRulesCoreExt = "ChainRulesCore"
    ChainRulesExt = "ChainRules"
    ComponentArraysExt = "ComponentArrays"
    FillArraysExt = "FillArrays"
    GPUArraysSparseArraysExt = ["GPUArrays", "SparseArrays"]
    MLUtilsExt = "MLUtils"
    MetalExt = ["GPUArrays", "Metal"]
    OneHotArraysExt = "OneHotArrays"
    OpenCLExt = ["GPUArrays", "OpenCL"]
    ReactantExt = "Reactant"
    RecursiveArrayToolsExt = "RecursiveArrayTools"
    ReverseDiffExt = "ReverseDiff"
    SparseArraysExt = "SparseArrays"
    TrackerExt = "Tracker"
    ZygoteExt = "Zygote"
    cuDNNExt = ["CUDA", "cuDNN"]
    oneAPIExt = ["GPUArrays", "oneAPI"]

    [deps.MLDataDevices.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
registries = "General"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "ComputePipeline", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FastInterpolations", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "Pkg", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "8ff1cd11162828e9f517e32d83b49d2af63391c6"
registries = "General"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.24.17"

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
git-tree-sha1 = "d9d20d6eb2a3290ba5c70a1c013b45305cc39c9e"
registries = "General"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.1"

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

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "BFloat16s", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "PrecompileTools", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "4e0d1973b9d72c797640263497a8f92cc726cfc3"
registries = "General"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.58"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreCUDNNExt = ["EnzymeCore", "CUDA", "cuDNN"]
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibMetalExt = "Metal"
    NNlibMooncakeExt = "Mooncake"
    NNlibReactantExt = "Reactant"
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "676c0f1c1c34224d6869675047a54b42fdc7b8c5"
registries = "General"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.5"

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
git-tree-sha1 = "750d0f2b5aac238d50dfee5f7a44fae3175a3305"
registries = "General"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.18.0"
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
git-tree-sha1 = "afa66dc1246b28b68600d053715e00a379268980"
registries = "General"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.5.3+0"

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

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Compat", "ConstructionBase", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "ce02e5e508795e3df43be2f6a2220e3fd0aae206"
registries = "General"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.4.11"

    [deps.Optimisers.extensions]
    OptimisersAdaptExt = ["Adapt"]
    OptimisersEnzymeCoreExt = "EnzymeCore"
    OptimisersReactantExt = "Reactant"
    OptimisersReactantMLDataDevicesExt = ["Reactant", "MLDataDevices"]

    [deps.Optimisers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    MLDataDevices = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

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
git-tree-sha1 = "91503bbab255aec9374145accd1b09c7e108cd54"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.5"

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
git-tree-sha1 = "9869376035f47756b8b8e711fd0d06a06d51da6f"
registries = "General"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.84"

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
git-tree-sha1 = "ce5ba3edfdeba107a721748d53b5f5887f415853"
registries = "General"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.13.0"

    [deps.ProgressMeter.extensions]
    ProgressMeterDistributedExt = "Distributed"
    ProgressMeterIJuliaExt = "IJulia"

    [deps.ProgressMeter.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

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

[[deps.ReactantCore]]
deps = ["ExpressionExplorer", "MacroTools"]
git-tree-sha1 = "1fe4e616752a0c4818ae88a504fc59b0fb016659"
registries = "General"
uuid = "a3311ec8-5e00-46d5-b541-4f83e724a433"
version = "0.1.26"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
registries = "General"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

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
git-tree-sha1 = "c5e8a0a52c0d5d043214de04f11f4d9c2c6ccdb8"
registries = "General"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "3.0.13"

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

[[deps.SciMLPublic]]
deps = ["PrecompileTools"]
git-tree-sha1 = "74685afb51732a464fbce79a72708f7c4203ceb7"
registries = "General"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.3.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "67a144433c4ce877ee6d1ada69a124d6b1ecf7be"
registries = "General"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
registries = "General"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "89880a2cca5cabc68c9023f58e544beda1d2fc1f"
registries = "General"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.2.0"

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

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "eec446511ab8c3293dd846c61c15128392fceed5"
registries = "General"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.3"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "88ef0d98204515bc53dd4aaea5f7788882d1b66c"
registries = "General"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
registries = "General"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "474a5283ad435618090122872eea6a8165ea6bcf"
registries = "General"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.4.6"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "SciMLPublic", "Static"]
git-tree-sha1 = "2a635e15d5035c53b345077c947f31ff91744078"
registries = "General"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.10.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "8a8a52f887f38d44014e24ee46b8a7311c625f6d"
registries = "General"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.24"
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
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    AdaptExt = "Adapt"
    GPUArraysCoreExt = "GPUArraysCore"
    KernelAbstractionsExt = "KernelAbstractions"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"
    StaticArraysExt = "StaticArrays"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "b04bfab37484defe300b2f9ed64469abab2431ff"
registries = "General"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.12.0"

    [deps.StructUtils.extensions]
    StructUtilsDBInterfaceExt = ["DBInterface", "Tables"]
    StructUtilsDataFramesExt = ["DataFrames"]
    StructUtilsLazilyInitializedFieldsExt = ["LazilyInitializedFields"]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    DBInterface = "a10d1c49-ce27-4219-8d33-6db1a4562965"
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
git-tree-sha1 = "eb3b85ca904a1d1647b15acc091b113b8874b648"
registries = "General"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.3"

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

[[deps.UnsafeAtomics]]
deps = ["LLVM"]
git-tree-sha1 = "74ac5e0e702972da953ffc981094617070be582d"
registries = "General"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.4.0"

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsGPUCompilerExt = "GPUCompiler"

    [deps.UnsafeAtomics.weakdeps]
    GPUCompiler = "61eb1bfa-7361-4325-ad38-22787b887f55"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
registries = "General"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WeightInitializers]]
deps = ["ConcreteStructs", "GPUArraysCore", "LinearAlgebra", "Random", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "66fcded521d4effa1917d6131cf5cce7caf66eb9"
registries = "General"
uuid = "d49dbf32-c5c2-4618-8acc-27bb2598ef2d"
version = "1.3.5"

    [deps.WeightInitializers.extensions]
    AMDGPUExt = "AMDGPU"
    CUDAExt = "CUDA"
    ChainRulesCoreExt = "ChainRulesCore"
    GPUArraysExt = "GPUArrays"
    ReactantExt = "Reactant"

    [deps.WeightInitializers.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

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

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "b6bb65f9d868585a33d76917cf81a222c69f1d30"
registries = "General"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.7.15"

    [deps.Zygote.extensions]
    ZygoteAtomExt = "Atom"
    ZygoteCUDAExt = "CUDA"
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Atom = "c52e3926-4ff0-5f6e-af25-54175e0327b1"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "c6a86c133861234450ab260dee01b42abd604095"
registries = "General"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.8"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
registries = "General"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "21e05cef1411bfe6b22a1ca8cbdc64a297d4b447"
registries = "General"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.15.3+0"

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
git-tree-sha1 = "11c07205bc3b3b7833b8fb36beb4e4764d0126a7"
registries = "General"
uuid = "46979653-d7f6-5232-b59e-dd310c4598de"
version = "1.26.0+1"

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
git-tree-sha1 = "21cbc8fc2ac0bb182ed461e556d9c062ae6a3164"
registries = "General"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.24.1+0"

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
# ╟─b3dce1ff-f7f7-5a3b-97e3-a19f8dcaca7a
# ╠═65cc8b96-25fe-5676-aac0-d4715c84314e
# ╟─ca4d9ebf-335b-573f-8a78-32072021bcc1
# ╟─0e7627a5-f61c-5b54-a1ad-df392620ffb1
# ╠═73b6b9b6-1ba8-5d35-a029-3329024fbf64
# ╠═32501cfd-cdf6-5491-98be-191229948e55
# ╠═b14d1afc-72b0-5118-bdab-1944df3afb78
# ╟─4ab48f39-82fd-5a3b-8c65-160eb701dc29
# ╟─2d3bba55-ba86-57a8-a905-659e57b6bb77
# ╠═0e2eeebd-951a-5eea-b995-04fd66b688dd
# ╠═883df49d-a6a1-5194-b5e5-90c916d52d50
# ╠═e6b75c28-b9ba-5ee8-a231-a6fca1845bef
# ╠═0a7ef374-adca-5b9b-8159-1f85b9197338
# ╟─947ec45d-2a0e-545e-8c2c-9d41313272af
# ╟─4d023645-bc11-59f7-a488-3169674ff70d
# ╟─ce3d4e6b-baa9-4627-8cb7-7d262ff86f52
# ╠═f0cc85b0-16f3-5503-b720-3983d7094b28
# ╠═fcb787e3-5f33-57a4-a550-fd6e5f3ddf17
# ╠═d62c9d9b-80f5-5977-90a7-8915a360dd90
# ╠═25420e04-b36b-5938-8fd7-472bead269bc
# ╠═71c1f97f-0824-5767-8a69-03e6955c0682
# ╟─5e8b963b-a5d0-536a-b53f-968e33d67bae
# ╠═c5f39cf6-2893-5993-9e5b-30cc55ea7bab
# ╠═09eabb3f-704e-59e6-b54d-505eeeb7d0e4
# ╟─837af198-c6af-58f0-be18-50d0030689c8
# ╠═5a4add4f-3828-5aca-b76c-3960b85c880d
# ╠═92788641-8d09-5fd1-b9c2-4c7e1c087ec3
# ╟─74a4508b-259f-541d-ba00-7aff49ee6f0f
# ╟─07e1afbe-1435-5366-a807-d9d8c1976008
# ╟─75189faf-163a-5b1a-87be-a6743c1e23b3
# ╠═9ccc1f86-94c6-534c-bd09-cad1de2d352c
# ╠═364ba16a-24b2-5296-a9f2-dbc60240312f
# ╟─ce58e9b6-6f20-5b53-a16c-d43e0eb26801
# ╟─eccb9cb7-c861-5a93-a115-86f10262b9ec
# ╟─149bdecd-7728-58b7-b78e-d9f7d1e2871c
# ╟─b05932cc-c7d9-5410-b7f2-b053cfeb4326
# ╠═74aab869-7574-5319-9f4a-4d2db701db61
# ╠═2456446b-4992-5de2-b992-547ad8a1f182
# ╟─d4d41340-f72a-59de-953f-566e994be2e4
# ╠═645418b5-949a-549b-a139-473f5f0dc1bf
# ╟─4a8c2ead-7277-50d5-a17a-08aa9fff050e
# ╟─5595ab9e-5cb7-55d0-a5fc-78ce8b192ad9
# ╟─4f2d3cd5-aa6a-55c2-b701-d6eb03f48907
# ╠═9a217e44-1307-51a8-97b1-2473236dfb1d
# ╟─34ac9ced-0eac-5cd2-b82a-6ea260a10ae7
# ╟─6094aea6-a49a-5994-af61-7fa4253be3cd
# ╠═13dc376e-9c80-51b8-ad55-43405a0bede7
# ╠═b2bf51b2-ec1d-5779-b33f-ab29f6252e3e
# ╠═b5a144de-7e9c-5e59-8d36-eeaca5979795
# ╠═6a142169-be9c-551e-971d-f119bd744a7d
# ╠═5f9babdd-3f19-5094-a970-b54c312a128b
# ╠═a2229ad1-11aa-5e0d-b9f3-d610980e2b65
# ╠═a9143211-6207-5f4f-8b09-b97a06ef55b3
# ╠═e6407394-aa39-5a0b-96c5-2fa925bfa5d8
# ╟─01b6047a-9102-5935-8aa1-81fd0a989d99
# ╠═e5095ad0-2394-593c-842a-4c187bbc910f
# ╠═eac8d019-c638-5190-8e50-de49099a5c27
# ╠═5deb2101-b3e8-50d5-b539-def1b355aa30
# ╠═dd526d59-4939-5057-b7f7-033238d6c653
# ╟─c73bfdc7-f9d3-5945-acf1-dfdaca253077
# ╠═034f774e-317b-5b32-a608-d5a2772d5815
# ╠═604db986-57ab-5229-991d-1ed52f386a91
# ╠═6d5d8243-2bc8-5aa5-ac36-a5d9bb3ec05d
# ╠═88329ada-a959-5972-b1fe-a4f2f402a6d2
# ╟─d98a33f9-22fb-5a73-86d1-e560e7cf3ac7
# ╟─d1c1a326-d00d-5bfd-a564-a4ffb65134cd
# ╟─3a71b0d9-be6c-54b8-9e0d-eae5dfa59dad
# ╠═1e5e51eb-80ec-536a-859c-ce087491ce9c
# ╠═59a6369d-a0f2-5e2d-86db-1f19fdf4181f
# ╟─22127acf-0b2e-5b6e-ba2e-d39e003c78ee
# ╟─eeb4827a-f390-5f76-8556-b2aeb465afba
# ╟─df7eb95a-cb0d-56ce-85d8-09b582fc2d9f
# ╟─11bdb645-bcf7-5c15-a5c6-2b943b4027a8
# ╟─5f01405a-c452-5604-8424-569e5925454b
# ╟─4cd46967-e9fc-5a51-b092-7325a7abf661
# ╠═fcdbe5e8-df07-5650-89fe-622dfd3f217f
# ╟─a506b9d7-5a85-549a-bd26-76c9fa947a09
# ╟─009641e7-3f55-55bd-969d-c7bb9d207c8a
# ╟─103fd665-1073-5fd8-9c20-3a82f23ca479
# ╟─895efa0c-63f2-561e-abd4-19742c36dc32
# ╟─280abba4-d2a3-53de-bc26-154763b473f5
# ╟─8d48e436-8617-534e-a7e8-495c71f148b7
# ╠═14931e07-d199-5611-84ca-0a515a39903c
# ╠═781d8d49-01ea-5211-9ecd-a4a0e064c0c8
# ╠═ee6e09f2-2151-5238-8cf9-7cbff70d5d2f
# ╠═fafc485f-3338-581b-b8ad-5cfd2bd5a055
# ╟─80a53f12-49fd-556b-bd19-0a3e6bc35079
# ╠═0f928b74-c921-5f8d-a473-00dbe725af3a
# ╠═94cf7183-2d9e-5667-aef3-e328987831d8
# ╟─ccb5fc61-c16c-5237-b9de-a764149d50a5
# ╟─ee7073ef-2ccc-585b-be04-b6ea7a6c4c61
# ╟─0dedf7d3-174f-567c-a37f-f461d954ff28
# ╟─e20c620c-1bb9-5b36-a2b6-6e5b90de84ac
# ╟─d78a68bb-5df1-5bb6-bdb8-72f4bea340ab
# ╟─b63350b6-22a5-5904-af65-02b3571cb763
# ╠═e5c4ba91-0cdd-5ba6-ba9a-eddc7e822f8d
# ╟─0dcac29e-382b-525a-a70f-6d04979abae5
# ╠═f95b8975-7eb9-5d1c-86d0-5ee0d19c1178
# ╠═d4830351-0e00-5b5e-9204-8da9bf420a48
# ╠═8e55a733-7d7b-53ff-ab1f-b30e93b7dbbd
# ╠═b5b82864-abe3-5ec4-9b0f-6d18ad2af047
# ╟─80694af1-403b-57e3-b836-5cff96c5c3e3
# ╠═37c8db24-5ecc-51b5-9b61-0ae7f40458a0
# ╠═fb39e911-5cb0-5cc2-ae15-c4563a6f2f45
# ╟─301dd0cc-6bfa-5964-b6b7-a914f4bb455d
# ╟─be298eb2-41b5-5448-9b78-c0f3e0d252c3
# ╟─b0455b32-4928-5765-b48c-74cf5c3cbd78
# ╟─08c9d2ee-9c4c-57db-ac0b-2e8cc638fd03
# ╟─1ff92f76-894f-57d4-bfbe-b626480e13cb
# ╠═3f6cc915-39b8-50cd-9527-64993e41a46b
# ╠═2aeefb9c-d9f3-5189-9d01-99296749a21b
# ╠═f7728920-be5f-5993-bc72-bbecbcf07d80
# ╠═f1ccd75e-bfdc-5ad2-a02b-c468a14eb0fc
# ╠═077df9e7-ca77-530f-8d6f-931f9e26498c
# ╠═84aacc97-52cc-5221-978e-874554ea41a3
# ╠═21f5451d-01bf-567d-982b-4517de423903
# ╠═6b41f06e-f343-5728-817e-eae08696a8e5
# ╠═3b314357-4a17-5bde-8170-310d7f2ba3ee
# ╠═0d0339c0-c1ab-5f51-8b5b-e72219150fe7
# ╠═fc1f13fd-db8c-514a-bbc7-a9234145f6d5
# ╠═3f895d6d-2c7a-58c1-924a-ae9830aca097
# ╠═cec8f3c9-b875-5e81-aa2c-df88ac7ae07a
# ╠═7afb28ea-f81f-5845-9893-09bd63657b65
# ╠═8ce757c7-3507-5b82-9b64-31a218a2b79b
# ╠═04b5296d-d362-5623-9a88-448c87f1f18d
# ╠═386a3f78-ce6a-5434-ab8b-e9fa428aa9a5
# ╠═f85c9643-35bd-5663-9f04-bab99f8d2154
# ╠═a1887041-35ed-5e9e-a653-25fcc1935d40
# ╠═d3f55395-d4ae-5450-9b7d-901134745ecb
# ╠═12dfbff1-23b7-5764-9f3b-c011780ccd70
# ╠═387c10aa-faf0-5008-8cfc-9c96abd00ff1
# ╠═77df1761-4ebb-5a61-ae36-121f3b299699
# ╠═9afeec6e-205b-5197-b586-6b5cab85d54f
# ╠═54337ec5-d8f6-5ad1-a077-6c1e1ded0573
# ╠═b845009d-ef89-598a-a612-39e96be14606
# ╠═adf3104b-6068-5973-a5c9-c8b315f1fcd8
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
