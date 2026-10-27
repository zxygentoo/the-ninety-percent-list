### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ 627356e1-392e-4624-a04a-b0f0cffb05f0
begin
    using CairoMakie, PlutoUI
    using Printf, Random
end

# ╔═╡ e5caf670-59e8-4ffa-8652-8c4fa3d2945b
md"""
# Kolmogorov Complexity and Algorithmic Randomness

*We know what the best compressor is. Can we find it?*

[Notebook 01](https://zxygentoo.github.io/the-ninety-percent-list/01.An.Observation.on.Generalization.html) named the best compressor there is: ``K(x)``, the length of the shortest program that prints ``x``. This notebook asks the obvious next question. Given a string, can we find its shortest program?

**Prerequisites**

- [Notebook 01](https://zxygentoo.github.io/the-ninety-percent-list/01.An.Observation.on.Generalization.html), or the idea that a compressed file is a description of the original.
- What a program, a loop and a counter are.
- Logarithms in base 2.
"""

# ╔═╡ 1a06f904-cebb-43a2-8bd3-dcea80cf58ba
TableOfContents()

# ╔═╡ 80c26549-7387-493b-a7fa-0111fe12d4cd
md"""
## ``K``, the best compressor

A compressor turns a string into something shorter, from which the original can be rebuilt. The best compressor imaginable would find, for every string, the shortest program that prints it.

Some strings have programs much shorter than themselves:

- `1010101010101010` is `print("10" ^ 8)`: "10", eight times. A program that prints "10" a million times is barely longer: only the number grows.
- The first million digits of ``\pi`` look patternless, yet a few lines of code compute them.

Others seem not to:

- `0110100010111001`, sixteen coin flips, seems to have no program shorter than `print("0110100010111001")`. Whether it really has none is the question this notebook runs into.

The length of the shortest program is the string's *Kolmogorov complexity*, ``K(x)``. It measures structure. A string with a lot of it has a short recipe and a small ``K``. A string with none costs about its own length.

!!! info "Definition"
    To make "shortest program" exact, fix a *decompressor* ``D``: a program that reads a bit string and, if it stops, outputs a string. Any input ``d`` with ``D(d) = x`` is a *description* of ``x``, and the complexity of ``x`` is the length of its shortest description:

    ```math
    K_D(x) = \min \,\{\, |d| \;:\; D(d) = x \,\}
    ```

    - ``d`` is any bit string, a candidate input to the decompressor;
    - ``D(d) = x`` keeps the candidates that ``D`` turns into ``x``: the descriptions of ``x``;
    - ``|d|`` is the length of a description, in bits;
    - ``\min`` picks the shortest.

    The book also calls ``D`` a *description mode*, and writes ``C`` where we write ``K``: it saves ``K`` for a variant, prefix complexity, in chapter 4. Notebook 01 already used ``C`` for a compressor's output.

If ``D`` is the interpreter of a programming language, a description is a program, and ``K_D(x)`` is the length of the shortest program that prints ``x``. That is the best compressor: the shortest program is the compressed file, and running it gives back ``x``.

The subscript ``D`` is there because the answer depends on the decompressor. The next section builds a small one, and from then on we drop the subscript and write ``K(x)`` for complexity on it. The real ``K`` is defined on a far more capable decompressor, and a note there says how the two relate.

The definition says what ``K(x)`` is, but not how to find it.
"""

# ╔═╡ ecdc07f2-09b3-4776-b9e2-1c95157bcb7c
md"""
## A little machine to find ``K``

The obvious way to find the shortest description is to try them all, shortest first, and stop at the first that prints the string. For that we need a decompressor simple enough to run millions of times.

### The machine

The machine has an output, which starts empty, and a counter, which starts at zero. It knows six instructions:

| instruction | what it does | steps |
|:-:|:--|:-:|
| `0`, `1` | append that bit to the output | 1 |
| `r` | repeat the output: ``x`` becomes ``xx`` | one per bit copied, at least 1 |
| `+` | add one to the counter | 1 |
| `[` | if the counter is 0, skip past the matching `]` | 1 |
| `]` | take one off the counter, which never goes below 0; if it is still above 0, go back to just after the matching `[` | 1 |

A program halts when it runs off its end. A program whose brackets don't match describes nothing.

Let's play the machine and run two example programs.

First, `1rr`:

| instruction | output | steps so far |
|:-:|:--|--:|
| `1` | `1` | 1 |
| `r` | `11` | 2 |
| `r` | `1111` | 4 |

The program halts and prints `1111`.

Now `+[+]`:

| instruction | counter | what happens |
|:-:|:-:|:--|
| `+` | 1 | |
| `[` | 1 | the counter isn't 0, so enter the loop |
| `+` | 2 | |
| `]` | 1 | take one off; still above 0, loop back |
| `+` | 2 | |
| `]` | 1 | go back again |
| … | 2, 1, 2, … | |


This one loops forever.

All the machine can express is repetition. That makes it a poor compressor, but one whose every short program we can try.
"""

# ╔═╡ d99b524c-072b-42dd-b5e7-0f8c5585178f
INSTRUCTIONS = ('0', '1', 'r', '+', '[', ']')

# ╔═╡ 4baf4325-231a-4ebb-82ee-47209fb631e6
"For each bracket in program `p`, the position of the bracket it matches."
function match_brackets(p)
    jump, open = zeros(Int, length(p)), Int[]
    for (i, c) in enumerate(p)
        if c == '['
            push!(open, i)
        elseif c == ']'
            j = pop!(open)
            jump[i], jump[j] = j, i
        end
    end
    jump
end

# ╔═╡ c659b873-9c05-439e-a2e6-9f383cc80825
"""
    run_instruction(p, jump, state)

The machine's state after one more instruction of program `p`, whose brackets pair up as in `jump`.
A state holds the position in the program `pc`, the output `out`, the `counter`, and the `steps` taken.
"""
function run_instruction(p, jump, (; pc, out, counter, steps))
    c = p[pc]
    c in "01" ? (; pc = pc + 1, out = out * c, counter, steps = steps + 1) :
    c == 'r'  ? (; pc = pc + 1, out = repeat(out, 2), counter, steps = steps + max(length(out), 1)) :
    c == '+'  ? (; pc = pc + 1, out, counter = counter + 1, steps = steps + 1) :
    c == '['  ? (; pc = counter == 0 ? jump[pc] + 1 : pc + 1, out, counter, steps = steps + 1) :
    #= ] =#     (; pc = counter > 1 ? jump[pc] + 1 : pc + 1, out, counter = max(counter - 1, 0), steps = steps + 1)
end

# ╔═╡ f4f4ed71-574d-4d31-90ef-c32c73ee526f
"""
    run_program(p; cap, fits = out -> true)

Run program `p` for at most `cap` steps. Returns its last state, and how the run `ended`:

- `:halted`: it ran off the end of the program;
- `:capped`: its next instruction would take it past `cap` steps;
- `:dropped`: its output stopped passing `fits`. The machine only adds to the end of its output, so a search can drop it there.
"""
function run_program(p; cap, fits = out -> true)
    jump = match_brackets(p)
    state = (; pc = 1, out = "", counter = 0, steps = 0)
    while state.pc <= length(p)
        next = run_instruction(p, jump, state)
        next.steps > cap && return (; state..., ended = :capped)
        fits(next.out) || return (; next..., ended = :dropped)
        state = next
    end
    (; state..., ended = :halted)
end

# ╔═╡ 7071dd05-8185-4a4b-9810-824ede1c58d6
CAP = 1_000

# ╔═╡ 7c823376-ece1-40bf-ad51-536cea674917
examples = map(["1rr", "00rrr", "+++[01]", "1+++[r]", "+[+]"]) do program
    (; program, run_program(program; cap = CAP)...)
end;

# ╔═╡ 6e89b6a2-a2d2-4e7f-a084-3319f9e258fc
md"""
> The rigorous ``K`` is defined on a [*universal* decompressor](https://en.wikipedia.org/wiki/Universal_Turing_machine), one that can run any program, like a Python interpreter. There are many of them, and which one hardly matters: each can run another's programs once it is handed an interpreter for them, so their ``K``s differ by about the length of that interpreter, a fixed constant (Theorem 1 in the book).

> Our machine is far from universal; all it can do is repeat. But anything it prints, a universal decompressor can print too, so its ``K`` is an upper bound on the real one, up to that constant. And a search that just runs programs meets the same two problems on it as on any machine: it can't tell a slow program from one that never stops, and the number of programs to try explodes.
"""

# ╔═╡ b1783191-4952-47be-8897-e857111035fd
md"""
### Counting in bits

Six instructions need 3 bits each to encode (3 bits give eight codes, so two go unused), and a program of ``m`` instructions is ``3m`` bits long.

Printing a string with `0` and `1` costs three bits for every bit (they're instructions too), so a string with no pattern would cost three times its own length. We need an escape hatch for this: a flag bit at the front of every description.

| first bit | the rest is | length of the description |
|:-:|:--|:--|
| `0` | the string itself, a *literal* | ``1 + n`` bits, for a string of ``n`` bits |
| `1` | a *program* | ``1 + 3m`` bits, for ``m`` instructions |

Every string now has a description only one bit longer than itself, so ``K(x) \le n + 1``. A program wins only when ``1 + 3m < 1 + n``, that is, when it has fewer than ``n / 3`` instructions. For sixteen 0s, `00rrr` takes ``1 + 15 = 16`` bits, one fewer than the literal's 17.
"""

# ╔═╡ fb9ea17e-0c5d-4b84-a6e9-4d6bf9d6ddb1
md"""
### The search

The search runs in rounds. Round ``m`` tries every program of ``m`` instructions, in dictionary order (`0`, `1`, `r`, `+`, `[`, `]`), and runs each one. The first round in which some program prints the target gives the answer, and the first program in it that does is the shortest description.

Here it is looking for `0000`:

| round | programs | what happens |
|:-:|:--|:--|
| 0 | the empty program | prints nothing |
| 1 | `0`, `1`, `r`, `+` | `0` prints `0`; `1` goes wrong at once; `r` and `+` print nothing |
| 2 | 17 programs, from `00` to `[]` | none prints more than two bits |
| 3 | `000`, `001`, `00r`, … | `000` prints `000`; `001` goes wrong; **`00r` prints `0000`** |

Two things keep the rounds small:

- **Brackets must match.** `[` and `]` on their own describe nothing, so round 1 has four programs, not six, and round 2 has 17, not 36.
- **A run that goes wrong is dropped.** `1` has printed `1`, which can never become `0000`, because the machine only adds to the end of its output. So the search stops running it at once.

Is `00r` the shortest description of `0000`? Not on our machine. As a description it costs ``1 + 3 \times 3 = 10`` bits, and the literal only 5. A program has to beat the literal, so the search only needs rounds with fewer than ``n / 3`` instructions. For `0000` that means rounds 0 and 1, which find nothing, so ``K(0000) = 5``. Programs start to pay off at 16 bits: for sixteen 0s the search runs rounds 0 to 5, and finds `00rrr` in round 5.

Longer searches meet `+[+]` in round 4, and waiting for it to finish means waiting forever. This is what the step cap is for: a program still running at the cap is set aside, as if it had printed nothing.
"""

# ╔═╡ f85e3a2a-1fd2-4ac8-88b8-fb35151d4ae9
"Are the brackets of program `p` balanced? Its nesting depth must never drop below zero, and must end at zero."
function balanced(p)
    depth = cumsum([(c == '[') - (c == ']') for c in p])
    all(≥(0), depth) && (isempty(depth) || last(depth) == 0)
end

# ╔═╡ 6c5acb92-ae5c-439c-8140-9f107f034ea4
"Every program of `m` instructions whose brackets match, in dictionary order: `0`, `1`, `r`, `+`, `[`, `]`."
function programs(m)
    every = (join(reverse(t)) for t in Iterators.product(ntuple(_ -> INSTRUCTIONS, m)...))  # reversed, so the last position varies fastest
    collect(Iterators.filter(balanced, every))
end

# ╔═╡ 847462de-fdd3-4b4c-99cc-8f942905bfbe
# All programs of up to 9 instructions, made once and shared by both searches.
PROGRAMS = Dict(m => programs(m) for m in 0:9);

# ╔═╡ 23fe3b5b-27c3-4f33-9e30-5f6bf3259878
"Does program `p` print `x` within `cap` steps?"
prints(p, x; cap) = let r = run_program(p; cap, fits = out -> startswith(x, out))
    r.ended == :halted && r.out == x
end

# ╔═╡ 04f7df8b-b97a-41f7-bba1-9dbf22f05bc8
"""
    shortest(x; cap)

The shortest description of `x` found when each program may run for at most `cap` steps:
the first program, shortest first, that prints `x`, or else the literal. Also counts the programs tried.
"""
function shortest(x; cap)
    tried = 0
    for m in 0:(length(x) - 1) ÷ 3    # programs shorter than the literal: 1 + 3m < 1 + length(x)
        hits = filter(p -> prints(p, x; cap), PROGRAMS[m])
        tried += length(PROGRAMS[m])
        isempty(hits) || return (; bits = 1 + 3m, description = first(hits), tried)
    end
    (; bits = 1 + length(x), description = "literal", tried)
end

# ╔═╡ 88b2c83a-e48a-42ff-aebe-f1d73634225f
md"""
## ``K`` is not computable

Suppose some program did compute ``K``. We could use it to hunt for complex strings: pick a number ``N``, and write

> go through all strings in order, and print the first ``x`` with ``K(x) > N``.

The hunt always succeeds. Only finitely many strings have a description of ``N`` bits or fewer, and there are infinitely many strings.

Now raise ``N``. The string the hunt prints must get ever more complex, since none of its descriptions is ``N`` bits or shorter. The hunt itself barely changes. The program that computes ``K`` stays the same, and so does the loop around it: together, a fixed ``c`` bits. Only the number ``N`` written into it grows, and writing ``N`` down takes about ``\log_2 N`` bits: doubling ``N`` adds a single bit. All told, the hunt is about ``c + \log_2 N`` bits long.
"""

# ╔═╡ 1f5a563d-4d0d-4fea-8088-38ac258f8fb7
md"""
So ``N`` races ahead of the hunt's own length, and sooner or later passes it: ``c + \log_2 N < N``. Take an ``N`` past that point, and look at the string ``x`` the hunt prints. Two things are true of it.

The hunt printed it, so the hunt is a description of ``x``, and the shortest description can be no longer:

```math
K(x) \le c + \log_2 N
```

And the hunt printed it because it passed the hunt's test:

```math
K(x) > N
```

Chain them through ``c + \log_2 N < N``:

```math
K(x) \;\le\; c + \log_2 N \;<\; N \;<\; K(x)
```

``K(x)`` would be less than itself.

The loop is ordinary, and the counting is sound. What gave way is the one thing we supposed: a program that computes ``K``. There is none.

This is [Berry's paradox](https://en.wikipedia.org/wiki/Berry_paradox), made precise (Theorem 6 in the book). "The first string with no description of ``N`` bits or fewer" is itself a description of that string, and a short one.

> One caveat about our machine. The argument needs a *universal* machine, one that can run the hunt as a program. Ours is too weak, which is also why we could see through `+[+]`. ``K`` is defined on universal machines, and there no amount of searching can certify the answer.

So the search for ``K`` is stuck for good. There is always a shorter program it can't rule out, because it can't know whether that program is slow or will never stop. What can we do now?
"""

# ╔═╡ a25b45c3-a2f3-431b-aec6-31e1fa36106b
md"""
## ``Kt``, the cost of running things

If we can't tell slow from never, stop waiting: make the waiting cost something. A slow description then loses to a fast one, the way a long description loses to a short one. We give up a little compression, since the shortest description might be a slow one. In exchange, the search ends.

Every description now has two costs: how long it is, and how long it runs. Here are the strings from the start of this notebook again, each with a description:

| string | description | its length | its running time |
|:--|:--|:--|:--|
| `1010101010101010` | `print("10" ^ 8)` | short | a moment |
| the first million digits of ``\pi`` | a few lines of code | short | trillions of steps, for a simple program |
| `0110100010111001`, sixteen coin flips | `print("0110100010111001")` | as long as the string | about a step per bit |

The first is short and fast, the second short but slow, the third fast but long. Are trillions of steps a fair price for saving millions of bits? To say, we need an exchange rate between steps and bits.

Leonid Levin's ``Kt`` sets one: a description pays for its length, plus the *logarithm* of its running time.

!!! info "Definition"
    Fix a decompressor ``D``, as for ``K``. If ``D`` turns a description ``d`` into ``x`` in ``t`` steps, the description costs ``|d| + \log_2 t`` bits, and ``Kt(x)`` is the cheapest:

    ```math
    Kt(x) = \min \,\{\, |d| + \log_2 t \;:\; D(d) = x \text{ in } t \text{ steps} \,\}
    ```

    - ``|d|`` is the length of the description, in bits, as in ``K``;
    - ``t`` is the number of steps ``D`` takes to turn ``d`` into ``x``;
    - ``\log_2 t`` is the cost of that time, in bits;
    - ``\min`` picks the cheapest description.

The rate is one bit per doubling: running twice as long costs the same as being one bit longer. At that rate, time is cheap for all three strings:

- `print("10" ^ 8)` pays a few bits.
- The ``\pi`` program pays about 40 bits for its trillions of steps, against the 3.3 million bits of the digits themselves.
- The literal of the coin flips pays ``\log_2 16 = 4`` bits.

The formula shows the price we pay: ``Kt(x) \ge K(x)``, since ``\log_2 t`` is never negative. If the shortest description is fast, ``Kt`` is only a few bits worse than ``K``. If it is slow, ``Kt`` settles for a longer one that runs faster. On a universal machine a short program can run for longer than the age of the universe, and then the gap can be large. On our machine it never comes to that: for every string of up to 24 bits, ``Kt``'s best description is the shortest one, so the price is just the ``\log_2 t``.

``Kt`` is not in the book, which calls resource-bounded complexity "outside the scope of our book" (p. 377). Levin defined it in a [1984 paper](https://www.cs.bu.edu/fac/lnd/pdf/inf.pdf), building on his 1973 *universal search*; he used natural logarithms, and this notebook uses base 2 throughout.

### Levin search

The logarithm turns the step cap from a guess into a rule. **Levin search** runs in phases ``k = 1, 2, 3, \ldots``, and in phase ``k`` a description of ``\ell`` bits may run for ``2^{k - \ell}`` steps. That is exactly enough for every description with ``\ell + \log_2 t \le k``. So:

- When a phase ends, every description has either printed, or run out of steps, in which case it scores above ``k`` even if it prints ``x`` later.
- The first phase in which some description prints ``x`` is ``\lceil Kt(x) \rceil``, and the best description in it gives ``Kt(x)`` exactly.

`+[+]` gets a budget like every other program. When it runs out, the search moves on. A program that never halts can no longer hold up the answer.
"""

# ╔═╡ 6ba46b6f-3c23-46f2-a5f0-fb07e4c8e379
"""
    phase(x, k)

Phase `k` of Levin search for `x`: every description of l ≤ k bits runs for 2^(k - l) steps.
Returns the descriptions that printed `x`, each scored by l + log2(steps), and how many programs ran.
"""
function phase(x, k)
    n = length(x)
    # Of all the literals, only x's own prints x: 1 + n bits, printed in n steps.
    literal = 1 + n <= k && n <= 2^(k - 1 - n) ? [(Kt = 1 + n + log2(n), description = "literal")] : []
    runs = [(p, run_program(p; cap = 2^(k - 1 - 3m), fits = out -> startswith(x, out))) for m in 0:(k - 1) ÷ 3 for p in PROGRAMS[m]]
    printed = [(Kt = 1 + 3length(p) + log2(r.steps), description = p) for (p, r) in runs if r.ended == :halted && r.out == x]
    (; found = [literal; printed], tried = length(runs))
end

# ╔═╡ d35209ee-0a3e-4d30-9d19-e4b94579edb5
"""
    levin(x)

`Kt(x)` by Levin search: run phases 1, 2, 3, … until one prints `x`. Returns the best description
in that phase with its `Kt`, the phase, and the programs tried over all phases.
"""
function levin(x)
    tried = 0
    for k in Iterators.countfrom(1)
        result = phase(x, k)
        tried += result.tried
        isempty(result.found) || return (; argmin(f -> f.Kt, result.found)..., phase = k, tried)
    end
end

# ╔═╡ 4face28f-83ae-4385-b7f3-5d0f72fe6841
md"""
## Nothing is cheap

``Kt``'s search always comes back, which ``K``'s never can. But it pays in the same currency as the search for ``K``: programs tried. Here is the cost of both, for coin-flip strings of every length from 4 to 24 bits.
"""

# ╔═╡ 9a0e8619-395a-4c83-a4eb-cb01e219458e
LENGTHS = 4:24

# ╔═╡ 9b5f388b-b0ea-45d6-a51c-e57228f310ca
md"""
## Structure is rare, randomness is the norm

Look at what a search must do before it can stop. To return a description of ``x``, it first has to rule out every shorter one. A string with structure has a short description, so the search stops early. A string without structure has none shorter than its literal, so the search must try every program shorter than that: all of them.

Here are three strings of the same length, 24 bits: two with structure, one without.
"""

# ╔═╡ cf49ed6a-21fc-4a87-b44b-5fe8b3377b3d
md"""
Almost none. This is the simplest fact in chapter 1 (Theorem 5), and it takes nothing but counting.

There are ``2^0 + 2^1 + \cdots + 2^{m-1} = 2^m - 1`` bit strings shorter than ``m`` bits, so fewer than ``2^m`` descriptions shorter than ``m`` bits, and each describes at most one string. Of the ``2^n`` strings of length ``n``, then, fewer than ``2^{n-k}`` have a description shorter than ``n - k`` bits:

```math
\frac{\#\{\, x \text{ of length } n \;:\; K(x) < n - k \,\}}{2^n} \;<\; 2^{-k}
```

Fewer than half of all strings have a description shorter than ``n - 1`` bits, and fewer than one in a thousand one shorter than ``n - 10``. Counting doesn't care what the machine is, so this holds for ``Kt``, which is never below ``K``, for zstd, and for every neural network.

The book turns this into a bet (p. 8). Toss a coin 80,000 times and save the result as a 10,000-byte file. No compressor that existed before the tosses will shrink it by more than 10 bytes: the chance that any one compressor does is below ``2^{-80}``, and there aren't that many compressors.
"""

# ╔═╡ 671cfa11-e1d1-4340-8474-fee239bb1697
# For every output of up to 24 bits: the first, and so shortest, program of up to 7 instructions that prints it.
# A program of 8 instructions is 25 bits, no shorter than the literal of any string up to 24 bits.
simplest = let runs = [(p, run_program(p; cap = CAP, fits = out -> length(out) <= 24)) for m in 0:7 for p in PROGRAMS[m]]
    Dict(unique(first, [r.out => p for (p, r) in runs if r.ended == :halted]))
end;

# ╔═╡ ad737d1a-f371-48ec-835b-25fb565b39cd
md"""
## Appendix: helpers

The coin flips, number formatting, and the chart style. The two series use the first two slots of a colour-blind-safe categorical palette.
"""

# ╔═╡ 6155e11b-8a04-462c-9590-05a5bd4f9d1f
"A string of `n` coin flips, the same on every run."
coin(n) = join(rand(Xoshiro(n), ('0', '1'), n))

# ╔═╡ ad4216da-7792-4ae7-8bbc-7318a9387799
# The sixteen coin flips are the ones from the introduction.
TARGETS = ["0"^16, "01"^8, "0"^24, "0110100010111001", coin(20)]

# ╔═╡ e9fd5fb3-2a61-41ba-86bd-5ba50268fcc7
k_table = [(; x, shortest(x; cap = CAP)...) for x in TARGETS]

# ╔═╡ c31f8dd3-af67-4898-93ca-69baf74a9ba5
let zeros16 = k_table[1], per_bit = (length(PROGRAMS[9]) / length(PROGRAMS[8]))^(1 / 3)
    Markdown.parse("""
    ## What this shows

    1. **``K`` is the best compressor, but a search for it doesn't settle.** What a search finds within a step cap is only an upper bound: a program the cap set aside might be shorter, as `$(zeros16.description)` would have been with a cap of 10.
    2. **``K`` is not computable.** A program still running might be slow or might never stop, and Berry's paradox shows that no algorithm can compute ``K``.
    3. **``Kt`` charges for running time, and its search ends.** It adds ``\\log_2 t`` bits for running time, every program gets a deadline, and Levin search returns an exact answer.
    4. **Nothing is cheap.** Both searches grow by about ×$(round(per_bit; digits = 2)) per bit of string, and a 128-bit string is already out of reach.
    5. **Structure is rare, randomness is the norm.** A search must rule out every description shorter than the best one, and for most strings the best is the string itself: fewer than one string in ``2^k`` has a description ``k`` bits shorter, on any machine.

    Practical compressors live with this by narrowing the search. zstd looks only for back-references; a neural network looks only for what its weights can express, found by gradient descent rather than one program at a time. Both give up the guarantee of the shortest description to get *a* description, in time. ``K`` is the yardstick they are measured against, and ``Kt`` shows that even a computable version of the yardstick is out of reach.
    """)
end

# ╔═╡ 3043a622-07a7-4fc4-a9ab-504ec6d495df
kt_table = [(; x, levin(x)...) for x in TARGETS]

# ╔═╡ e997e6d3-2ba9-455a-8347-b30394e242fb
costs = map(LENGTHS) do n
    x = coin(n)
    (; n, K = shortest(x; cap = CAP).tried, Kt = levin(x).tried)
end;

# ╔═╡ 552af09a-a1a1-4638-975e-475c1bf9b094
contrast = [(; x, K = shortest(x; cap = CAP), Kt = levin(x)) for x in ["0"^24, "001"^8, coin(24)]]

# ╔═╡ 3ba9b3cf-f3f6-4cfc-8909-e4f70cbde016
commas(n::Integer) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => ",")

# ╔═╡ ced6454d-9de0-4281-a249-5d51c309081a
let rows = map(c -> "| `$(c.x)` | $(c.K.bits) | $(commas(c.K.tried)) | $(@sprintf("%.2f", c.Kt.Kt)) | $(commas(c.Kt.tried)) |", contrast)
    structured, flips = first(contrast), last(contrast)
    Markdown.parse("""
    | string | ``K``, best found | programs tried | ``Kt`` | programs tried |
    |:--|--:|--:|--:|--:|
    $(join(rows, "\n"))

    The coin flips cost the search for ``K`` $(round(Int, flips.K.tried / structured.K.tried)) times as many programs as the 0s, and Levin search $(round(Int, flips.Kt.tried / structured.Kt.tried)) times as many. Each search runs until it reaches the string's best description, and there are up to ``2^\\ell`` descriptions shorter than ``\\ell`` bits. So the cost of a search grows exponentially with the string's complexity, not its length.

    So the question becomes: how many strings have structure?
    """)
end

# ╔═╡ e2108ad6-af44-4f61-bb16-b2a7c6644024
let rows = map((16, 20, 24)) do n
        K = [1 + 3length(p) for (x, p) in simplest if length(x) == n]
        "| $n | $(commas(2^n)) | $(count(<(n), K)) | $(count(<(n - 4), K)) | $(commas(2^(n - 4) - 1)) |"
    end
    rare = sort([(x, p) for (x, p) in simplest if length(x) == 24 && 1 + 3length(p) < 20]; by = last)
    Markdown.parse("""
    On our machine, structure is rarer still. Here is every program of up to 7 instructions, run for up to $(commas(CAP)) steps, against every string of 16, 20 and 24 bits:

    | bits | strings | with a description shorter than the string | shorter by more than 4 bits | counting allows up to |
    |--:|--:|--:|--:|--:|
    $(join(rows, "\n"))

    The $(length(rare)) strings of 24 bits that compress by more than 4 bits are each a three-bit pattern, repeated:

    $(join(("- `$x`, printed by `$p`" for (x, p) in rare), "\n"))

    Counting leaves room for $(commas(2^20 - 1)) of them. Our machine finds $(length(rare)), because the only structure it can express is repetition. A richer machine would find more. None can find more than counting allows.

    So a typical string is like the coin flips. It has no structure to find, and the search for its best description has to try every shorter description first. Structure is rare, and that is why searching for it is expensive.
    """)
end

# ╔═╡ 8b33ab7e-dd6e-4ee6-a219-b69a1338ed17
"`x` in scientific notation, as inline math for markdown."
function sci(x)
    e = floor(Int, log10(x))
    "``$(round(x / 10.0^e; digits = 1)) \\times 10^{$e}``"
end

# ╔═╡ 01503ebf-ebc7-4472-ac4c-2eb23d39ffcd
"A number of years, in words."
duration(years) = years < 1 ? "about $(round(Int, 12years)) months" :
                  years < 1_000 ? "about $(round(Int, years)) years" : "about $(sci(years)) years"

# ╔═╡ 678586b7-b147-483c-8546-82ed6b95048c
let per_instruction = length(PROGRAMS[9]) / length(PROGRAMS[8])
    per_bit = per_instruction^(1 / 3)
    programs_at(n) = last(costs).Kt * per_bit^(n - last(LENGTHS))
    years_at(n) = programs_at(n) / 1e9 / (365.25 * 24 * 3600)    # at a billion programs a second
    Markdown.parse("""
    Both lines climb at the same rate. Each extra instruction multiplies the number of programs by about $(round(per_instruction; digits = 1)): six choices, less the ones whose brackets don't match. An instruction is three bits, so that is about ×$(round(per_bit; digits = 2)) per bit of string. The ``Kt`` search tries more programs at every length, because it reruns them in each phase and, with ``\\log_2 t`` in its score, reaches a little further. Its slope is the same.

    Extend the line. A 64-bit coin flip would take Levin search about $(sci(programs_at(64))) programs: at a billion programs a second, $(duration(years_at(64))). A 128-bit one would take $(duration(years_at(128))), some $(commas(round(Int, round(years_at(128) / 13.8e9; sigdigits = 2)))) times the age of the universe.

    So what separates ``K`` from ``Kt`` is not the cost:

    > **``K`` and ``Kt`` need the same exponential search. Only ``Kt``'s search ends.**

    Why does it have to be this expensive?
    """)
end

# ╔═╡ 8f8c94ee-da76-4db7-8743-4aa5f4a3623b
"How a run ended, in words."
ending(ended) = ended == :halted ? "halted" : ended == :capped ? "still running at the cap" : "dropped"

# ╔═╡ 17b8298f-1257-4915-a1f4-b61583668c7e
Markdown.parse("""
Five programs, run by the code above. `+[+]` would never stop, so each run gets a *step cap*: after $(commas(CAP)) steps, a run still going is stopped.

| program | output | steps | ended |
|:--|:--|--:|:--|
$(join(("| `$(e.program)` | $(isempty(e.out) ? "(nothing)" : "`$(e.out)`") | $(commas(e.steps)) | $(ending(e.ended)) |" for e in examples), "\n"))

In `+++[01]` and `1+++[r]`, three `+`s make the loop body run three times.
""")

# ╔═╡ 130bc65b-f1fa-49a3-9161-f8ffddd0d225
"A description, for a markdown table: a program as code, or the literal."
shown(description) = description == "literal" ? "the literal" : "`$description`"

# ╔═╡ 956f5a70-d173-4ce9-9bb6-4da6f55af675
let rows = ("| `$(r.x)` | $(r.bits) | $(shown(r.description)) | $(commas(r.tried)) |" for r in k_table)
    Markdown.parse("""
    The best we can find with $(commas(CAP)) steps per program:

    | string | best found (bits) | description | programs tried |
    |:--|--:|:--|--:|
    $(join(rows, "\n"))
    """)
end

# ╔═╡ 04cd4ff9-a1ab-4965-b246-ad81b9ff7b11
let zeros16 = k_table[1], flips = k_table[end], stingy = shortest(zeros16.x; cap = 10)
    Markdown.parse("""
    The table looks like answers. It isn't, quite.

    - **A cap can hide the answer.** `$(zeros16.description)` needs $(run_program(zeros16.description; cap = 10^6).steps) steps. With a cap of 10 it would have been set aside like `+[+]`, and the search would have settled for $(shown(stingy.description)), $(stingy.bits) bits. Nothing guarantees that $(commas(CAP)) steps is enough either.
    - **The literal is only what we found.** For the $(length(flips.x)) coin flips, no program beat the $(flips.bits)-bit literal. To call the literal *shortest*, we would have to rule out every shorter program, including the ones the cap set aside.

    `+[+]` is one of those. Run it for a thousand steps, or a billion, and all we learn is that it hasn't stopped *yet*. Is it slow, or will it never stop? Running it can't tell us.

    Here we can read the program and see that it loops forever, and on a machine this simple, a cleverer search could do that for every program. On a universal machine, it can't. That is a theorem.
    """)
end

# ╔═╡ e3b6d93f-6b15-45ba-8572-529d26388999
let
    # Length in bits, and steps to print x, of a description.
    cost(x, d) = d == "literal" ? (bits = 1 + length(x), steps = length(x)) : (bits = 1 + 3length(d), steps = run_program(d; cap = 10^6).steps)
    rows = map(kt_table) do r
        (; bits, steps) = cost(r.x, r.description)
        "| `$(r.x)` | $(@sprintf("%.2f", r.Kt)) | $(shown(r.description)) | $bits | $steps | $(r.phase) | $(commas(r.tried)) |"
    end
    zeros16, flips = kt_table[1], kt_table[end]
    z, f = cost(zeros16.x, zeros16.description), cost(flips.x, flips.description)
    Markdown.parse("""
    | string | ``Kt`` | best description | its bits | its steps | phase | programs tried |
    |:--|--:|:--|--:|--:|--:|--:|
    $(join(rows, "\n"))

    Every search ends, and every answer is exact.

    - For sixteen 0s, `$(zeros16.description)` is $(z.bits) bits long and takes $(z.steps) steps: ``Kt = $(z.bits) + \\log_2 $(z.steps) = $(@sprintf("%.2f", zeros16.Kt))``. When the program is fast, ``Kt`` is ``K`` plus a few bits.
    - For $(length(flips.x)) coin flips, the best is still the literal: $(f.bits) bits, printed in $(f.steps) steps, so ``Kt = $(f.bits) + \\log_2 $(f.steps) = $(@sprintf("%.2f", flips.Kt))``.

    No step cap had to be chosen, and nothing was left running. But look at the last column.
    """)
end

# ╔═╡ 0e07f66b-9847-4b9e-a483-4b5cb5287675
VIZ = (surface = "#fcfcfb", text = "#0b0b0b", text2 = "#52514e", grid = "#e4e3de")

# ╔═╡ 2b8c2c43-2710-49d9-a277-073cbeffc017
SERIES_COLORS = ["#2a78d6", "#eb6834"]

# ╔═╡ fef76558-f6a4-4edd-b417-3f74594b6e3d
THEME = Theme(
    fontsize = 13, backgroundcolor = VIZ.surface, textcolor = VIZ.text,
    Axis = (backgroundcolor = VIZ.surface, xgridcolor = VIZ.grid, ygridcolor = VIZ.grid,
            topspinevisible = false, rightspinevisible = false, leftspinecolor = VIZ.text2, bottomspinecolor = VIZ.text2,
            xtickcolor = VIZ.text2, ytickcolor = VIZ.text2, xticklabelcolor = VIZ.text2, yticklabelcolor = VIZ.text2,
            titlealign = :left, titlefont = :regular, titlesize = 14),
    Legend = (framevisible = false, backgroundcolor = VIZ.surface),
)

# ╔═╡ 0e12e9c0-9fc0-4d2e-8f55-733161e149aa
let sizes = (10^3, 10^5, 10^7), Ns = exp10.(range(0, 9; length = 200))
    # Where N passes a hunt of c fixed bits: the N with N = c + log2(N), found by iterating.
    passes(c) = foldl((N, _) -> c + log2(N), 1:10; init = float(c))
    blue, orange = SERIES_COLORS
    with_theme(THEME) do
        fig = Figure(size = (760, 440))
        ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "N", ylabel = "bits",
                  title = "However long the hunt starts, N passes it")
        lines!(ax, Ns, Ns; color = orange, linewidth = 2, label = "N: the string has no description this short")
        for c in sizes
            lines!(ax, Ns, c .+ log2.(Ns); color = blue, linewidth = 2, label = "the hunt: c + log₂ N bits")
            text!(ax, 1, c; text = "c = $(commas(c))", align = (:left, :bottom), offset = (6, 4), color = VIZ.text2)
            scatter!(ax, [passes(c)], [passes(c)]; color = VIZ.text, markersize = 10, strokecolor = VIZ.surface, strokewidth = 2)
        end
        text!(ax, passes(first(sizes)), passes(first(sizes)); text = "past each dot, the hunt is shorter than N",
              align = (:left, :top), offset = (10, -8), color = VIZ.text2)
        Legend(fig[2, 1], ax; orientation = :horizontal, merge = true)
        fig
    end
end

# ╔═╡ 7baaaf58-7317-47d3-ae15-c4e5f921184a
with_theme(THEME) do
    fig = Figure(size = (760, 440))
    ax = Axis(fig[1, 1]; yscale = log10, xticks = 4:4:24,
              xlabel = "length of the coin-flip string (bits)", ylabel = "programs tried",
              title = "Both searches grow exponentially with the length of the string")
    series = [(:K, "K", "search for K, $(commas(CAP)) steps per program", :circle),
              (:Kt, "Kt", "Levin search for Kt", :rect)]
    for ((key, short, label, marker), color) in zip(series, SERIES_COLORS)
        values = getproperty.(costs, key)
        scatterlines!(ax, LENGTHS, values; color, marker, markersize = 9, linewidth = 2, label)
        text!(ax, last(LENGTHS), last(values); text = short, align = (:left, :center), offset = (10, 0), color = VIZ.text, fontsize = 13)
    end
    xlims!(ax, first(LENGTHS) - 1, last(LENGTHS) + 2)
    Legend(fig[2, 1], ax; orientation = :horizontal)
    fig
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

julia_version = "1.13.2"
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
# ╟─e5caf670-59e8-4ffa-8652-8c4fa3d2945b
# ╠═627356e1-392e-4624-a04a-b0f0cffb05f0
# ╟─1a06f904-cebb-43a2-8bd3-dcea80cf58ba
# ╟─80c26549-7387-493b-a7fa-0111fe12d4cd
# ╟─ecdc07f2-09b3-4776-b9e2-1c95157bcb7c
# ╠═d99b524c-072b-42dd-b5e7-0f8c5585178f
# ╠═4baf4325-231a-4ebb-82ee-47209fb631e6
# ╠═c659b873-9c05-439e-a2e6-9f383cc80825
# ╠═f4f4ed71-574d-4d31-90ef-c32c73ee526f
# ╠═7071dd05-8185-4a4b-9810-824ede1c58d6
# ╠═7c823376-ece1-40bf-ad51-536cea674917
# ╟─17b8298f-1257-4915-a1f4-b61583668c7e
# ╟─6e89b6a2-a2d2-4e7f-a084-3319f9e258fc
# ╟─b1783191-4952-47be-8897-e857111035fd
# ╟─fb9ea17e-0c5d-4b84-a6e9-4d6bf9d6ddb1
# ╠═f85e3a2a-1fd2-4ac8-88b8-fb35151d4ae9
# ╠═6c5acb92-ae5c-439c-8140-9f107f034ea4
# ╠═847462de-fdd3-4b4c-99cc-8f942905bfbe
# ╠═23fe3b5b-27c3-4f33-9e30-5f6bf3259878
# ╠═04f7df8b-b97a-41f7-bba1-9dbf22f05bc8
# ╠═ad4216da-7792-4ae7-8bbc-7318a9387799
# ╠═e9fd5fb3-2a61-41ba-86bd-5ba50268fcc7
# ╟─956f5a70-d173-4ce9-9bb6-4da6f55af675
# ╟─04cd4ff9-a1ab-4965-b246-ad81b9ff7b11
# ╟─88b2c83a-e48a-42ff-aebe-f1d73634225f
# ╟─0e12e9c0-9fc0-4d2e-8f55-733161e149aa
# ╟─1f5a563d-4d0d-4fea-8088-38ac258f8fb7
# ╟─a25b45c3-a2f3-431b-aec6-31e1fa36106b
# ╠═6ba46b6f-3c23-46f2-a5f0-fb07e4c8e379
# ╠═d35209ee-0a3e-4d30-9d19-e4b94579edb5
# ╠═3043a622-07a7-4fc4-a9ab-504ec6d495df
# ╟─e3b6d93f-6b15-45ba-8572-529d26388999
# ╟─4face28f-83ae-4385-b7f3-5d0f72fe6841
# ╠═9a0e8619-395a-4c83-a4eb-cb01e219458e
# ╠═e997e6d3-2ba9-455a-8347-b30394e242fb
# ╟─7baaaf58-7317-47d3-ae15-c4e5f921184a
# ╟─678586b7-b147-483c-8546-82ed6b95048c
# ╟─9b5f388b-b0ea-45d6-a51c-e57228f310ca
# ╠═552af09a-a1a1-4638-975e-475c1bf9b094
# ╟─ced6454d-9de0-4281-a249-5d51c309081a
# ╟─cf49ed6a-21fc-4a87-b44b-5fe8b3377b3d
# ╠═671cfa11-e1d1-4340-8474-fee239bb1697
# ╟─e2108ad6-af44-4f61-bb16-b2a7c6644024
# ╟─c31f8dd3-af67-4898-93ca-69baf74a9ba5
# ╟─ad737d1a-f371-48ec-835b-25fb565b39cd
# ╟─6155e11b-8a04-462c-9590-05a5bd4f9d1f
# ╟─3ba9b3cf-f3f6-4cfc-8909-e4f70cbde016
# ╟─8b33ab7e-dd6e-4ee6-a219-b69a1338ed17
# ╟─01503ebf-ebc7-4472-ac4c-2eb23d39ffcd
# ╟─8f8c94ee-da76-4db7-8743-4aa5f4a3623b
# ╟─130bc65b-f1fa-49a3-9161-f8ffddd0d225
# ╟─0e07f66b-9847-4b9e-a483-4b5cb5287675
# ╟─2b8c2c43-2710-49d9-a277-073cbeffc017
# ╟─fef76558-f6a4-4edd-b417-3f74594b6e3d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
