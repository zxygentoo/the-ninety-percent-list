# The Ninety Percent List

> So I asked Ilya Sutskever, OpenAI's chief scientist, for a reading list. He gave me a list of like 40 research papers and said, "If you really learn all of these, you'll know 90% of what matters today." And I did.
>
> John Carmack, in a [2023 interview with Dallas Innovates](https://dallasinnovates.com/exclusive-qa-john-carmacks-different-path-to-artificial-general-intelligence/)

A rebuild of that reading list, for anyone who wants the *why*: the theory of generalization, the lineage that explains each design, the current recipe, and the phenomena nobody can yet explain. Read it the way Carmack did: by reimplementing it.

Each item is worked through as a **Build**: the smallest thing to implement that forces the idea to be understood, usually with the one measurement that shows it worked. Builds are [Pluto](https://plutojl.org) notebooks in Julia, which read like essays and run like experiments.

## Notebooks

[All notebooks](https://zxygentoo.github.io/the-ninety-percent-list/)

| Item | Notebook | Status |
|--:|:--|:-:|
| 1 | [01 · An Observation on Generalization](https://zxygentoo.github.io/the-ninety-percent-list/01.An.Observation.on.Generalization.html) | ✅ |
| 2 | [02 · Kolmogorov Complexity and Algorithmic Randomness](https://zxygentoo.github.io/the-ninety-percent-list/02.Kolmogorov.Complexity.and.Algorithmic.Randomness.html) | ✅ |

Progress: **2 / 30** items built `██░░░░░░░░░░░░░░░░░░░░░░░░░░░░`

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

- `Ilya` marks items on the original 2019–20 list, as reconstructed (see [Provenance](#provenance)).
- `implement` marks items whose point is building them in full.

### A · Why learning works at all

The worldview. Prediction is compression, and a network is an approximation to the ideal compressor. Everything in the original list that looked out of place is here.

1. **[An Observation on Generalization](https://simons.berkeley.edu/news/observation-generalization)**  
   Sutskever · Simons Institute, Aug 2023 · talk · [YouTube recording](https://www.youtube.com/watch?v=AKMuA_TVz3A) (~1 h)

2. **[Kolmogorov Complexity and Algorithmic Randomness](https://www.lirmm.fr/~ashen/kolmbook-eng-scan.pdf)** `Ilya`  
   Shen, Uspensky, Vereshchagin · AMS 2017 · book (free PDF) · chapters 1–2 only

3. **[A Tutorial Introduction to the Minimum Description Length Principle](https://arxiv.org/abs/math/0406077)** `Ilya`  
   Grünwald · 2004 · arXiv:math/0406077

4. **[Keeping Neural Networks Simple by Minimizing the Description Length of the Weights](https://www.cs.toronto.edu/~hinton/absps/colt93.pdf)** `Ilya`  
   Hinton, van Camp · COLT 1993

5. **[Machine Super Intelligence](https://www.vetta.org/documents/Machine_Super_Intelligence.pdf)** `Ilya`  
   Legg · PhD thesis, 2008 · chapters 1–3

6. **[The First Law of Complexodynamics](https://scottaaronson.blog/?p=762) + [The Coffee Automaton](https://arxiv.org/abs/1405.6903)** `Ilya`  
   Aaronson · 2011 · blog post; Aaronson, Carroll, Ouellette · 2014 · arXiv:1405.6903

### B · The lineage

History that carries the why. Each design here is the answer to a failure of the one before it; read cold, the Transformer looks arbitrary. Condensed from the original, not cut.

7. **[ImageNet Classification with Deep Convolutional Neural Networks](https://www.cs.toronto.edu/~hinton/absps/imagenet.pdf)** `Ilya`  
   Krizhevsky, Sutskever, Hinton · NeurIPS 2012

8. **[The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) + [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)** `Ilya`  
   Karpathy · 2015 · blog; Olah · 2015 · blog

9. **[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) + [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)** `Ilya`  
   Sutskever, Vinyals, Le · 2014 · arXiv:1409.3215; Bahdanau, Cho, Bengio · 2014 · arXiv:1409.0473

10. **[Pointer Networks](https://arxiv.org/abs/1506.03134) + [Neural Turing Machines](https://arxiv.org/abs/1410.5401)** `Ilya`  
   Vinyals, Fortunato, Jaitly · 2015 · arXiv:1506.03134; Graves, Wayne, Danihelka · 2014 · arXiv:1410.5401

11. **[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) + [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)** `Ilya`  
   He, Zhang, Ren, Sun · 2015 · arXiv:1512.03385; 2016 · arXiv:1603.05027

12. **[Attention Is All You Need](https://arxiv.org/abs/1706.03762) + [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)** `Ilya` `implement`  
   Vaswani et al. · 2017 · arXiv:1706.03762; Rush et al. (Harvard NLP) · 2018, revised 2022

13. **[Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)** `Ilya`  
   Gilmer, Schoenholz, Riley, Vinyals, Dahl · 2017 · arXiv:1704.01212

14. **[Deep Speech 2](https://arxiv.org/abs/1512.02595)** `Ilya`  
   Amodei et al. (Baidu) · 2015 · arXiv:1512.02595

15. **[Variational Lossy Autoencoder](https://arxiv.org/abs/1611.02731) + [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)** `Ilya`  
   Chen et al. · 2016 · arXiv:1611.02731; Luo · 2022 · arXiv:2208.11970

### C · The modern recipe

2020 to 2025. Scaling, in-context learning, preference learning, and the one new paradigm since the original list: reinforcement learning on verifiable rewards.

16. **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) + [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)** `Ilya`  
   Kaplan et al. · 2020 · arXiv:2001.08361; Hoffmann et al. (Chinchilla) · 2022 · arXiv:2203.15556

17. **[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)**  
   Brown et al. (GPT-3) · 2020 · arXiv:2005.14165

18. **[Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) + [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)**  
   Christiano et al. · 2017 · arXiv:1706.03741; Ouyang et al. (InstructGPT) · 2022 · arXiv:2203.02155

19. **[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)**  
   Sutton, Barto · 2nd ed., 2018 · book (free PDF) · chapters 1–6 and 13

20. **[Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270) + [AlphaZero](https://arxiv.org/abs/1712.01815)**  
   Silver et al. · Nature 2017; Silver et al. · 2017 · arXiv:1712.01815

21. **[Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) + [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) + [DeepSeek-R1](https://arxiv.org/abs/2501.12948)**  
   Wei et al. · 2022 · arXiv:2201.11903; Lightman et al. · 2023 · arXiv:2305.20050; DeepSeek · 2025 · arXiv:2501.12948

22. **[The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)**  
   Sutton · 2019 · essay

### D · Phenomena without an explanation

Where discovery lives. Most of these were found by a small team on a small model, and each is reachable at laptop scale.

23. **[Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530)**  
   Zhang, Bengio, Hardt, Recht, Vinyals · ICLR 2017 · arXiv:1611.03530

24. **[Deep Double Descent](https://arxiv.org/abs/1912.02292)**  
   Nakkiran, Kaplun, Bansal, Yang, Barak, Sutskever · 2019 · arXiv:1912.02292

25. **[Grokking](https://arxiv.org/abs/2201.02177) + [Progress Measures for Grokking via Mechanistic Interpretability](https://arxiv.org/abs/2301.05217)**  
   Power et al. · 2022 · arXiv:2201.02177; Nanda et al. · ICLR 2023 · arXiv:2301.05217

26. **[In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)**  
   Olsson et al. · 2022 · Transformer Circuits Thread

27. **[Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682) + [Are Emergent Abilities of LLMs a Mirage?](https://arxiv.org/abs/2304.15004)**  
   Wei et al. · 2022 · arXiv:2206.07682; Schaeffer, Miranda, Koyejo · NeurIPS 2023 · arXiv:2304.15004

28. **[The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)**  
   Huh, Cheung, Wang, Isola · 2024 · arXiv:2405.07987

### E · The engineering of scale

Where the systems meet the math.

29. **[GPipe](https://arxiv.org/abs/1811.06965) + [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)** `Ilya`  
   Huang et al. · 2018 · arXiv:1811.06965; Google DeepMind · 2025 · online book

30. **[Let's Build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) + [nanoGPT](https://github.com/karpathy/nanoGPT) + [llm.c](https://github.com/karpathy/llm.c)** `implement`  
   Karpathy · 2023 video (~2 h) · 2022 repo · 2024 repo

## Provenance

- **The original was never published.** Carmack got it around 2019, and his copy was later lost to Meta's two-year email auto-delete ([Hacker News](https://news.ycombinator.com/item?id=34641359)).
- **The reconstruction** of roughly 27 items that has circulated since 2024 ([30papers.com](https://30papers.com), [gist](https://gist.github.com/matijagrcic/ae8353eb1e6be84a7c85d9fdc2f9631f)) is plausible but unverified, and has no RL or meta-learning. Items marked `Ilya` follow it.
- **Context for the revisions:** Sutskever's November 2025 interview with Dwarkesh Patel, "from the age of scaling to the age of research" ([highlights](https://forum.effectivealtruism.org/posts/iuKa2iPg7vD9BdZna/highlights-from-ilya-sutskever-s-november-2025-interview)).

Compiled in October 2026.

## Publishing

On every push to `main`, a [GitHub Action](.github/workflows/pages.yml) runs every notebook, exports each one as a static HTML page with an index, and publishes the result on GitHub Pages. If any cell errors, the export fails and nothing is published.

To build the site locally:

```sh
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'
julia --project=scripts scripts/export.jl     # writes site/
```

## License

[MIT](LICENSE)
