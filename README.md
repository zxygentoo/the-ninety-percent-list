# The Ninety Percent List

> So I asked Ilya Sutskever, OpenAI's chief scientist, for a reading list. He gave me a list of like 40 research papers and said, "If you really learn all of these, you'll know 90% of what matters today." And I did.
>
> John Carmack, in a [2023 interview with Dallas Innovates](https://dallasinnovates.com/exclusive-qa-john-carmacks-different-path-to-artificial-general-intelligence/)

A rebuild of that reading list, for anyone who wants the *why*: the theory of generalization, the lineage that explains each design, the current recipe, and the phenomena nobody can yet explain. Read it the way Carmack did: by reimplementing it.

Each item is worked through as a **Build**: the smallest thing to implement that forces the idea to be understood, usually with the one measurement that shows it worked. Builds are [Pluto](https://plutojl.org) notebooks in Julia, which read like essays and run like experiments, and each is linked under its item.

## Notebooks

[All notebooks](https://zxygentoo.github.io/the-ninety-percent-list/)

Progress: **2 / 30** items built `██░░░░░░░░░░░░░░░░░░░░░░░░░░░░`

| Item | Notebook | Status |
|--:|:--|:-:|
| 1 | [01 · An Observation on Generalization](https://zxygentoo.github.io/the-ninety-percent-list/01.An.Observation.on.Generalization.html) | ✅ |
| 2 | [02 · Kolmogorov Complexity and Algorithmic Randomness](https://zxygentoo.github.io/the-ninety-percent-list/02.Kolmogorov.Complexity.and.Algorithmic.Randomness.html) | ✅ |

### Running a notebook

1. Install Julia by following the [official instructions](https://julialang.org/downloads/). The notebooks are tested with 1.13.
2. Install Pluto and start it:

   ```julia
   julia> using Pkg; Pkg.add("Pluto")
   julia> using Pluto; Pluto.run()
   ```

3. In the browser tab that opens, enter the path of a notebook in `notebooks/` and open it.

Each notebook carries its own package environment, so its first run installs what it needs, which takes a few minutes. Downloaded data is cached in `notebooks/data/`, which git ignores.

## The list

Thirty items in five sections.

| | Section | Items |
|:-:|:--|:-:|
| A | [Why learning works at all](#a--why-learning-works-at-all) | 1–6 |
| B | [The lineage](#b--the-lineage) | 7–15 |
| C | [The modern recipe](#c--the-modern-recipe) | 16–22 |
| D | [Phenomena without an explanation](#d--phenomena-without-an-explanation) | 23–28 |
| E | [The engineering of scale](#e--the-engineering-of-scale) | 29–30 |

An item earns its slot by carrying a *why* you can't get elsewhere, or a phenomenon that still lacks an explanation. Being important or widely used is not enough.

- `Ilya` marks items on the original 2019–20 list, as reconstructed (see [Provenance](#provenance)).
- `implement` marks items whose point is building them in full.

### A · Why learning works at all

The worldview. Prediction is compression, and a network is an approximation to the ideal compressor. Everything in the original list that looked out of place is here.

1. **[An Observation on Generalization](https://simons.berkeley.edu/news/observation-generalization)**  
   Sutskever · Simons Institute, Aug 2023 · talk · [YouTube recording](https://www.youtube.com/watch?v=AKMuA_TVz3A) (~1 h)

   Watch first. Unsupervised learning as conditional Kolmogorov complexity, in the author's own words. It tells you why section A exists.

   **Build:** [01 · An Observation on Generalization](https://zxygentoo.github.io/the-ninety-percent-list/01.An.Observation.on.Generalization.html)

2. **[Kolmogorov Complexity and Algorithmic Randomness](https://www.lirmm.fr/~ashen/kolmbook-eng-scan.pdf)** `Ilya`  
   Shen, Uspensky, Vereshchagin · AMS 2017 · book (free PDF) · chapters 1–2 only

   The ideal compressor. Every learning method is an approximation to it.

   **Build:** [02 · Kolmogorov Complexity and Algorithmic Randomness](https://zxygentoo.github.io/the-ninety-percent-list/02.Kolmogorov.Complexity.and.Algorithmic.Randomness.html)

3. **[A Tutorial Introduction to the Minimum Description Length Principle](https://arxiv.org/abs/math/0406077)** `Ilya`  
   Grünwald · 2004 · arXiv:math/0406077

   The bridge from the uncomputable ideal to actual model selection.

4. **[Keeping Neural Networks Simple by Minimizing the Description Length of the Weights](https://www.cs.toronto.edu/~hinton/absps/colt93.pdf)** `Ilya`  
   Hinton, van Camp · COLT 1993

   Compression applied to a network's weights. The seed of variational methods.

5. **[Machine Super Intelligence](https://www.vetta.org/documents/Machine_Super_Intelligence.pdf)** `Ilya`  
   Legg · PhD thesis, 2008 · chapters 1–3

   Intelligence defined as sequence prediction plus reward. The AIXI frame all of this sits inside.

6. **[The First Law of Complexodynamics](https://scottaaronson.blog/?p=762) + [The Coffee Automaton](https://arxiv.org/abs/1405.6903)** `Ilya`  
   Aaronson · 2011 · blog post; Aaronson, Carroll, Ouellette · 2014 · arXiv:1405.6903

   What "interesting structure" is, and why it lives between order and noise. The most idiosyncratic items on the list, and an open question: why complexity rises and then falls while entropy only grows.

### B · The lineage

History that carries the why. Each design here is the answer to a failure of the one before it; read cold, the Transformer looks arbitrary. Condensed from the original, not cut.

7. **[ImageNet Classification with Deep Convolutional Neural Networks](https://www.cs.toronto.edu/~hinton/absps/imagenet.pdf)** `Ilya`  
   Krizhevsky, Sutskever, Hinton · NeurIPS 2012

   Read as the first scaling argument: depth, data, GPUs, and the belief it would keep working.

8. **[The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) + [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)** `Ilya`  
   Karpathy · 2015 · blog; Olah · 2015 · blog

   Sequence modeling as the universal interface; gating as the fix for vanishing gradients. The same problem residual connections solve later. The RNN side is back ([Mamba](https://arxiv.org/abs/2312.00752)), so this history is live.

9. **[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) + [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)** `Ilya`  
   Sutskever, Vinyals, Le · 2014 · arXiv:1409.3215; Bahdanau, Cho, Bengio · 2014 · arXiv:1409.0473

   The fixed-vector bottleneck, then attention invented to remove it. This pair is the why for attention.

10. **[Pointer Networks](https://arxiv.org/abs/1506.03134) + [Neural Turing Machines](https://arxiv.org/abs/1410.5401)** `Ilya`  
   Vinyals, Fortunato, Jaitly · 2015 · arXiv:1506.03134; Graves, Wayne, Danihelka · 2014 · arXiv:1410.5401

   Attention as addressing; differentiable memory. "Attention is a soft dictionary lookup" stops being a metaphor here. The learned-program thread that reasoning models reopened.

11. **[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) + [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)** `Ilya`  
   He, Zhang, Ren, Sun · 2015 · arXiv:1512.03385; 2016 · arXiv:1603.05027

   Why depth works. Notice it is the LSTM gating fix applied to depth instead of time.

12. **[Attention Is All You Need](https://arxiv.org/abs/1706.03762) + [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)** `Ilya` `implement`  
   Vaswani et al. · 2017 · arXiv:1706.03762; Rush et al. (Harvard NLP) · 2018, revised 2022

   Items 8–11, distilled: attention and residual connections kept, recurrence dropped, and positional encodings added to supply the order that recurrence gave for free. Build it in full, asking of each piece which earlier failure it answers.

13. **[Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)** `Ilya`  
   Gilmer, Schoenholz, Riley, Vinyals, Dahl · 2017 · arXiv:1704.01212

   Stands in for the original's three set/relational papers. A transformer is message passing on a complete graph; the question of which structure to hard-code versus learn is still open.

14. **[Deep Speech 2](https://arxiv.org/abs/1512.02595)** `Ilya`  
   Amodei et al. (Baidu) · 2015 · arXiv:1512.02595

   The first paper that reads as a scaling recipe (data, HPC, engineering) rather than an architecture.

15. **[Variational Lossy Autoencoder](https://arxiv.org/abs/1611.02731) + [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)** `Ilya`  
   Chen et al. · 2016 · arXiv:1611.02731; Luo · 2022 · arXiv:2208.11970

   The compression view of generative models, then diffusion derived from it. Skip Luo if you already have diffusion at that depth.

### C · The modern recipe

2020 to 2025. Scaling, in-context learning, preference learning, and the one new paradigm since the original list: reinforcement learning on verifiable rewards.

16. **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) + [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)** `Ilya`  
   Kaplan et al. · 2020 · arXiv:2001.08361; Hoffmann et al. (Chinchilla) · 2022 · arXiv:2203.15556

   Read as a pair; the second corrects the first. The laws are empirical, and data allocation was the thing everyone got wrong.

17. **[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)**  
   Brown et al. (GPT-3) · 2020 · arXiv:2005.14165

   In-context learning as a phenomenon. It made the field, and nobody fully explains it (see section D).

18. **[Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) + [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)**  
   Christiano et al. · 2017 · arXiv:1706.03741; Ouyang et al. (InstructGPT) · 2022 · arXiv:2203.02155

   Preference learning as the bridge from a predictor to an assistant.

19. **[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)**  
   Sutton, Barto · 2nd ed., 2018 · book (free PDF) · chapters 1–6 and 13

   The one textbook. RL is the pillar a deep-learning background often lacks, and every item after this assumes it.

20. **[Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270) + [AlphaZero](https://arxiv.org/abs/1712.01815)**  
   Silver et al. · Nature 2017; Silver et al. · 2017 · arXiv:1712.01815

   Search plus learning plus self-play. The clearest case of synthetic data from a verifier.

21. **[Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) + [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) + [DeepSeek-R1](https://arxiv.org/abs/2501.12948)**  
   Wei et al. · 2022 · arXiv:2201.11903; Lightman et al. · 2023 · arXiv:2305.20050; DeepSeek · 2025 · arXiv:2501.12948

   The reasoning line: test-time compute, process versus outcome reward, RL on verifiable rewards. R1 is item 20's idea applied to language.

22. **[The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)**  
   Sutton · 2019 · essay

   Two pages. The premise of the scaling era, and the thing Sutskever has since called incomplete.

### D · Phenomena without an explanation

Where discovery lives. Most of these were found by a small team on a small model, and each is reachable at laptop scale.

23. **[Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530)**  
   Zhang, Bengio, Hardt, Recht, Vinyals · ICLR 2017 · arXiv:1611.03530

   Networks memorize random labels yet generalize on real ones. Classical generalization theory does not apply.

24. **[Deep Double Descent](https://arxiv.org/abs/1912.02292)**  
   Nakkiran, Kaplun, Bansal, Yang, Barak, Sutskever · 2019 · arXiv:1912.02292

   More parameters, or more data, can hurt and then help. The bias–variance story is incomplete.

25. **[Grokking](https://arxiv.org/abs/2201.02177) + [Progress Measures for Grokking via Mechanistic Interpretability](https://arxiv.org/abs/2301.05217)**  
   Power et al. · 2022 · arXiv:2201.02177; Nanda et al. · ICLR 2023 · arXiv:2301.05217

   Delayed generalization, then the demonstration that you can reverse-engineer what the network learned (modular addition via Fourier features).

26. **[In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)**  
   Olsson et al. · 2022 · Transformer Circuits Thread

   A mechanistic account of item 17. Interpretability as the microscope for "why".

27. **[Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682) + [Are Emergent Abilities of LLMs a Mirage?](https://arxiv.org/abs/2304.15004)**  
   Wei et al. · 2022 · arXiv:2206.07682; Schaeffer, Miranda, Koyejo · NeurIPS 2023 · arXiv:2304.15004

   Read together: a lesson in how a phenomenon claim can be a metric artifact.

28. **[The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)**  
   Huh, Cheung, Wang, Isola · 2024 · arXiv:2405.07987

   Representations converging across models and modalities. A provocation that ties directly back to section A.

### E · The engineering of scale

Where the systems meet the math.

29. **[GPipe](https://arxiv.org/abs/1811.06965) + [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)** `Ilya`  
   Huang et al. · 2018 · arXiv:1811.06965; Google DeepMind · 2025 · online book

   Pipeline parallelism, then the modern roofline-and-sharding view of what actually bounds training and inference.

30. **[Let's Build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) + [nanoGPT](https://github.com/karpathy/nanoGPT) + [llm.c](https://github.com/karpathy/llm.c)** `implement`  
   Karpathy · 2023 video (~2 h) · 2022 repo · 2024 repo

   Stands in for the original's CS231n course. Not reading: implementing. Use them as references to check your own build against, not to copy from.

## Provenance

- **The original list was never published.** Carmack asked Sutskever for it around 2019 ("about four years ago", he said in the 2023 interview), and his copy is gone: "The email including them got lost to Meta's two-year auto-delete policy by the time I went back to look for it last year," as quoted in [this Hacker News thread](https://news.ycombinator.com/item?id=34641359).
- **The reconstruction.** The roughly 27-item reconstruction that has circulated since 2024 ([30papers.com](https://30papers.com), [gist](https://gist.github.com/matijagrcic/ae8353eb1e6be84a7c85d9fdc2f9631f)) is plausible but unverified, and it has no reinforcement learning or meta-learning section. Items marked `Ilya` follow it.
- **Context for the revisions.** Sutskever's November 2025 interview with Dwarkesh Patel, "from the age of scaling to the age of research" ([highlights](https://forum.effectivealtruism.org/posts/iuKa2iPg7vD9BdZna/highlights-from-ilya-sutskever-s-november-2025-interview)).

The list was compiled in October 2026.

## Publishing

On every push to `main`, a [GitHub Action](.github/workflows/pages.yml) runs every notebook, exports each one as a static HTML page with an index, and publishes the result on GitHub Pages. If any cell errors, the export fails and nothing is published.

To build the site locally:

```sh
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'
julia --project=scripts scripts/export.jl     # writes site/
```

## License

[MIT](LICENSE)
