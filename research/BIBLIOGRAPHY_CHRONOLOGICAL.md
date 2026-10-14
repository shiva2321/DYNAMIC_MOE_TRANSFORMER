# Annotated chronological bibliography — VSA geometry, associative memory, binding, neural networks, general-intelligence frameworks

Compiled 2026-10-03 for `research/VSA_GEOMETRY_TO_INTELLIGENCE.md`. Ordered oldest → latest.

**Verification labels (be strict about these):**

| Tag | Meaning |
|---|---|
| **[V]** | Existence, authors, venue and headline claim checked this session on an index/publisher/arXiv page (abstract level — **not** full text). |
| **[V-t]** | Only the title/venue was confirmed (search hit or listing); the claim was not read. |
| **[V*]** | Verified, but one detail (noted) is as reported by a search summary and should be double-checked. |
| **[R]** | Classic cited **from memory**. Author/year/venue believed right; not re-verified. Check before citing in a paper. |
| **[W]** | Withdrawn / retracted / flagged by its own authors. Listed so you do not cite it as evidence. |

arXiv IDs are given where known (`arXiv:YYMM.NNNNN`). Items marked *(theory only)* contain no experiments in the version I read. Anything claimed as a result is the **authors' claim**, not independently reproduced here.

---

## Era 0 — 1943–1969: neurons, information, holography, random expansion

- **1943** McCulloch & Pitts. *A logical calculus of the ideas immanent in nervous activity.* Bull. Math. Biophys. 5:115–133. **[R]** Threshold units as logic; origin of connectionist computation.
- **1948** Shannon. *A mathematical theory of communication.* Bell Syst. Tech. J. 27. **[R]** Entropy and channel capacity — the language later used for hypervector capacity.
- **1948** Gabor. *A new microscopic principle.* Nature 161:777. **[R]** Invention of holography; source of the "holographic memory" metaphor.
- **1949** Hebb. *The Organization of Behavior.* Wiley. **[R]** Co-activation learning; basis of outer-product memories.
- **1949** Shannon. *Communication in the presence of noise.* Proc. IRE 37:10–21. **[R]** Signals as points in a high-dimensional space with noise spheres; geometric origin of random-codebook decoding (= VSA cleanup).
- **1950** Hamming. *Error detecting and error correcting codes.* Bell Syst. Tech. J. 29:147–160. **[R]** Hamming distance — the metric of binary VSAs.
- **1958** Rosenblatt. *The perceptron.* Psychol. Rev. 65:386–408. **[R]** Random sparse retina→association connections + learned readout: an early random-feature / HDC-shaped architecture.
- **1960** Widrow & Hoff. *Adaptive switching circuits.* IRE WESCON Conv. Rec. 4:96–104. **[V]** LMS / delta rule; ancestor of error-correcting associative writes (DeltaNet, 2024).
- **1961** Steinbuch. *Die Lernmatrix.* Kybernetik 1:36–45. **[R]** Early binary associative matrix memory.
- **1962** Rosenblatt. *Principles of Neurodynamics.* Spartan. **[R]**
- **1964** Solomonoff. *A formal theory of inductive inference I, II.* Inf. & Control 7. **[R]** Universal prior; prediction ⇔ compression.
- **1965** Cover. *Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition.* IEEE Trans. Electron. Comput. EC-14:326–334. **[V]** Function-counting theorem: a random high-dimensional expansion makes patterns linearly separable with high probability.
- **1965** Kolmogorov. *Three approaches to the quantitative definition of information.* Probl. Inf. Transm. 1:1–7. **[R]**
- **1966** Chaitin. *On the length of programs for computing finite binary sequences.* J. ACM 13:547–569. **[R]**
- **1968** Longuet-Higgins. *Holographic model of temporal recall.* Nature 217. **[V\*]** Convolution/correlation ("holographic") associative memory. (Page number differs between sources; check.)
- **1969** Willshaw, Buneman, Longuet-Higgins. *Non-holographic associative memory.* Nature 222:960–962. **[V]** Sparse binary correlation memory; ancestor of Willshaw nets / SDM.
- **1969** Marr. *A theory of cerebellar cortex.* J. Physiol. 202:437–470. **[R]** Granule-cell expansion + sparse coding = pattern separation.
- **1969** Minsky & Papert. *Perceptrons.* MIT Press. **[R]**
- **1969** Pribram. *The neurophysiology of remembering.* Sci. Am. 220(1):73–86. **[R]** Holographic-brain hypothesis (historical context only).

## Era 1 — 1970–1989: associative memory meets statistical physics; the systematicity challenge

- **1970** Longuet-Higgins, Willshaw, Buneman. *Theories of associative recall.* Q. Rev. Biophys. 3:223–244. **[V-t]**
- **1971** Marr. *Simple memory: a theory for archicortex.* Phil. Trans. R. Soc. B 262:23–81. **[R]**
- **1971** Albus. *A theory of cerebellar function.* Math. Biosci. 10:25–61. **[R]**
- **1972** Kohonen. *Correlation matrix memories.* IEEE Trans. Comput. C-21:353–359. **[R]** Outer-product memory = linear-attention state = tensor-product binding.
- **1972** Anderson. *A simple neural network generating an interactive memory.* Math. Biosci. 14:197–220. **[R]** Linear associator.
- **1973** Kohonen & Ruohonen. *Representation of associated data by matrix operators.* IEEE Trans. Comput. C-22:701–702. **[R]** Error-correcting matrix memory (delta-rule style).
- **1974** Little. *The existence of persistent states in the brain.* Math. Biosci. 19:101–120. **[R]**
- **1978** Rissanen. *Modeling by shortest data description.* Automatica 14:465–471. **[R]** Minimum description length.
- **1980** Palm. *On associative memory.* Biol. Cybern. 36:19–31. **[R]**
- **1980** Treisman & Gelade. *A feature-integration theory of attention.* Cogn. Psychol. 12:97–136. **[R]** The perceptual binding problem.
- **1981** von der Malsburg. *The correlation theory of brain function.* MPI Biophys. Chem. Internal Report 81-2. **[R]** Binding by temporal correlation (synchrony).
- **1981** Hinton & Anderson (eds.). *Parallel Models of Associative Memory.* Erlbaum. **[R]**
- **1982** Hopfield. *Neural networks and physical systems with emergent collective computational abilities.* PNAS 79:2554–2558. **[R]** Energy-based attractor memory.
- **1982** Murdock. *A theory for the storage and retrieval of item and associative information.* Psychol. Rev. 89:609–626. **[R]** TODAM — convolution-based memory model (psychology's HRR).
- **1982** Eich. *A composite holographic associative recall model.* Psychol. Rev. 89:627–661. **[R]** CHARM.
- **1983** Gentner. *Structure-mapping: a theoretical framework for analogy.* Cogn. Sci. 7:155–170. **[R]**
- **1984** Johnson & Lindenstrauss. *Extensions of Lipschitz mappings into a Hilbert space.* Contemp. Math. 26:189–206. **[R]** Random projections preserve distances.
- **1985** Amit, Gutfreund, Sompolinsky. *Storing infinite numbers of patterns in a spin-glass model of neural networks.* Phys. Rev. Lett. 55:1530–1533. **[R]** Hopfield capacity ≈ 0.138 N.
- **1985** Ackley, Hinton, Sejnowski. *A learning algorithm for Boltzmann machines.* Cogn. Sci. 9:147–169. **[R]**
- **1986** Rumelhart, Hinton, Williams. *Learning representations by back-propagating errors.* Nature 323:533–536. **[R]**
- **1986** Hinton, McClelland, Rumelhart. *Distributed representations.* In *Parallel Distributed Processing* vol. 1, ch. 3, pp. 77–109. MIT Press. **[V]**
- **1987** McEliece, Posner, Rodemich, Venkatesh. *The capacity of the Hopfield associative memory.* IEEE Trans. Inf. Theory 33:461–482. **[V]** Capacity ∼ N / (2 ln N) — the same D / ln N scaling that appears in VSA bundling.
- **1987** Hinton & Plaut. *Using fast weights to deblur old memories.* CogSci. **[R]**
- **1987** Laird, Newell, Rosenbloom. *SOAR: an architecture for general intelligence.* Artif. Intell. 33:1–64. **[R]**
- **1988** Fodor & Pylyshyn. *Connectionism and cognitive architecture: a critical analysis.* Cognition 28:3–71. **[V]** The systematicity/compositionality challenge that VSA/TPR were built to answer.
- **1988** Gardner. *The space of interactions in neural network models.* J. Phys. A 21:257–270. **[R]** Storage capacity of attractor networks by statistical mechanics.
- **1988** Kanerva. *Sparse Distributed Memory.* MIT Press. **[V]** Hamming-sphere geometry of memory; keys as addresses; origin of "hyperdimensional" thinking.
- **1988** Touretzky & Hinton. *A distributed connectionist production system.* Cogn. Sci. 12:423–466. **[V]** Early distributed binding of triples.
- **1989** Falkenhainer, Forbus, Gentner. *The structure-mapping engine.* Artif. Intell. 41:1–63. **[R]**

## Era 2 — 1990–1999: the binding solutions (TPR, RAAM, HRR, spatter codes)

- **1990** Smolensky. *Tensor product variable binding and the representation of symbolic structures in connectionist systems.* Artif. Intell. 46:159–216. **[V]** TPR: role⊗filler outer products — the full group-algebra binding.
- **1990** Pollack. *Recursive distributed representations.* Artif. Intell. 46:77–105. **[V]** RAAM: fixed-width codes for trees.
- **1990** Hinton. *Mapping part-whole hierarchies into connectionist networks.* Artif. Intell. 46:47–75. **[R]**
- **1991** Plate. *Holographic reduced representations: convolution algebra for compositional distributed representations.* IJCAI-91:30–35. **[R]**
- **1992** Schmidhuber. *Learning to control fast-weight memories: an alternative to dynamic recurrent networks.* Neural Comput. 4:131–139. **[R]** Outer-product fast weights (the 1991–92 ancestor of linear attention).
- **1993** Shastri & Ajjanagadde. *From simple associations to systematic reasoning.* Behav. Brain Sci. 16:417–494. **[R]** SHRUTI: binding by synchrony.
- **1993** Kainen & Kůrková. *Quasiorthogonal dimension of Euclidean spaces.* Appl. Math. Lett. 6(3):7–10. **[R]** Exponentially many nearly-orthogonal directions.
- **1994** Hofstadter & Mitchell. *The Copycat project.* In *Advances in Connectionist and Neural Computation Theory* vol. 2. **[R]** Fluid analogy.
- **1995** Plate. *Holographic reduced representations.* IEEE Trans. Neural Netw. 6(3):623–641. **[V]** Circular-convolution binding with approximate inverse.
- **1995** McClelland, McNaughton, O'Reilly. *Why there are complementary learning systems in the hippocampus and neocortex.* Psychol. Rev. 102:419–457. **[V]** Fast sparse episodic store + slow distributed learner (CLS).
- **1995** Dayan, Hinton, Neal, Zemel. *The Helmholtz machine.* Neural Comput. 7:889–904. **[R]** Wake-sleep.
- **1995** Igelnik & Pao. *Stochastic choice of basis functions in adaptive function approximation and the functional-link net.* IEEE Trans. Neural Netw. 6:1320–1329. **[R]** Random expansion + linear readout.
- **1996** Kanerva. *Binary spatter-coding of ordered K-tuples.* ICANN'96, LNCS 1112:869–873. **[V]** XOR binding / majority bundling (BSC).
- **1996** Olshausen & Field. *Emergence of simple-cell receptive field properties by learning a sparse code for natural images.* Nature 381:607–609. **[R]**
- **1996** Lund & Burgess. *Producing high-dimensional semantic spaces from lexical co-occurrence.* Behav. Res. Methods 28:203–208. **[R]** HAL.
- **1997** Hummel & Holyoak. *Distributed representations of structure: a theory of analogical access and mapping.* Psychol. Rev. 104:427–466. **[V]** LISA: binding by synchrony; relational reasoning.
- **1997** Landauer & Dumais. *A solution to Plato's problem: the latent semantic analysis theory.* Psychol. Rev. 104:211–240. **[R]**
- **1997** Kanerva. *Fully distributed representation.* Real World Computing Symposium. **[V-t]**
- **1998** Gayler. *Multiplicative binding, representation operators & analogy.* In *Advances in Analogy Research.* **[R]**
- **1998** Halford, Wilson, Phillips. *Processing capacity defined by relational complexity.* Behav. Brain Sci. 21:803–831. **[R]** Human relational capacity ≈ 4 bindings.
- **1999** Rao & Ballard. *Predictive coding in the visual cortex.* Nat. Neurosci. 2:79–87. **[V]**

## Era 3 — 2000–2009: "Vector Symbolic Architecture" named; cognitive models; random features

- **2000** Kanerva, Kristoferson, Holst. *Random indexing of text samples for latent semantic analysis.* CogSci. **[R]**
- **2000** Donoho. *High-dimensional data analysis: the curses and blessings of dimensionality.* AMS lecture. **[R]**
- **2001** Rachkovskij & Kussul. *Binding and normalization of binary sparse distributed representations by context-dependent thinning.* Neural Comput. 13:411–452. **[V]** Binding that preserves sparsity.
- **2001** Eliasmith & Thagard. *Integrating structure and meaning: a distributed model of analogical mapping.* Cogn. Sci. 25:245–286. **[R]**
- **2001** Marcus. *The Algebraic Mind.* MIT Press. **[R]** Variable binding as the missing ingredient.
- **2001** Jaeger. *The "echo state" approach to analysing and training recurrent neural networks.* GMD Report 148. **[R]**
- **2002** Maass, Natschläger, Markram. *Real-time computing without stable states.* Neural Comput. 14:2531–2560. **[R]** Reservoir computing.
- **2003** Gayler. *Vector symbolic architectures answer Jackendoff's challenges for cognitive neuroscience.* ICCS/ASCS (arXiv cs/0412059). **[V]** Names the family "VSA".
- **2003** Eliasmith & Anderson. *Neural Engineering.* MIT Press. **[R]** NEF.
- **2003** Plate. *Holographic Reduced Representation: Distributed Representation for Cognitive Structures.* CSLI. **[R]**
- **2003** Hummel & Holyoak. *A symbolic-connectionist theory of relational inference and generalization.* Psychol. Rev. 110:220–264. **[V-t]**
- **2003/06** Schmidhuber. *Gödel machines: self-referential universal problem solvers making provably optimal self-improvements.* **[R]**
- **2004** Widdows. *Geometry and Meaning.* CSLI. **[R]** Vector geometry of meaning; quantum-logic connections.
- **2005** Hutter. *Universal Artificial Intelligence: Sequential Decisions Based on Algorithmic Probability.* Springer. **[V]** AIXI.
- **2006** Smolensky & Legendre. *The Harmonic Mind.* MIT Press. **[R]** TPR + Optimality Theory.
- **2006** Donoho. *Compressed sensing.* IEEE Trans. Inf. Theory 52:1289–1306. **[R]**
- **2006** Candès, Romberg, Tao. *Robust uncertainty principles.* IEEE Trans. Inf. Theory 52:489–509. **[R]**
- **2006** Huang, Zhu, Siew. *Extreme learning machine.* Neurocomputing 70:489–501. **[R]**
- **2007** Legg & Hutter. *Universal intelligence: a definition of machine intelligence.* Minds & Machines 17:391–444. **[V]**
- **2007** Jones & Mewhort. *Representing word meaning and order information in a composite holographic lexicon.* Psychol. Rev. 114:1–37. **[V]** BEAGLE: convolution-based lexicon fits human semantic data.
- **2007** Rahimi & Recht. *Random features for large-scale kernel machines.* NIPS. **[V]** Bochner/RFF — the kernel view of hypervector encodings.
- **2008** Gayler. *Vector symbolic architectures: a new building material for artificial general intelligence.* AGI-08. **[V-t]**
- **2008** Sahlgren, Holst, Kanerva. *Permutations as a means to encode order in word space.* CogSci. **[R]**
- **2008** Doumas, Hummel, Sandhofer. *A theory of the discovery and predication of relational concepts.* Psychol. Rev. 115:1–43. **[R]** DORA.
- **2009** Kanerva. *Hyperdimensional computing: an introduction to computing in distributed representation with high-dimensional random vectors.* Cogn. Comput. 1:139–159. **[V]**
- **2009** Aerts, Czachor, De Moor. *Geometric analogue of holographic reduced representation.* J. Math. Psychol. 53(5):389–398 (arXiv:0710.2611). **[V]** Clifford (geometric) product as binding; exact inverses.
- **2009** Gayler & Levy. *A distributed basis for analogical mapping.* In *New Frontiers in Analogy Research.* **[V-t]**
- **2009** Schmidhuber. *Driven by compression progress.* In *Anticipatory Behavior in Adaptive Learning Systems.* **[R]**

## Era 4 — 2010–2015: Neural Engineering / Spaun; word vectors; first memory-augmented nets

- **2010** Friston. *The free-energy principle: a unified brain theory?* Nat. Rev. Neurosci. 11:127–138. **[V]**
- **2010** Kanerva. *What we mean when we say "What's the dollar of Mexico?"* AAAI Fall Symp. **[R]** Mapping vectors: analogy as group-like binding.
- **2011** Veness, Ng, Hutter, Uther, Silver. *A Monte-Carlo AIXI approximation.* J. Artif. Intell. Res. 40:95–142. **[R]**
- **2012** Eliasmith, Stewart, Choo, Bekolay, DeWolf, Tang, Rasmussen. *A large-scale model of the functioning brain.* Science 338:1202–1205. **[V]** Spaun: 2.5 M spiking neurons, Semantic Pointer Architecture, multiple tasks.
- **2012** Krizhevsky, Sutskever, Hinton. *ImageNet classification with deep convolutional neural networks.* NIPS. **[R]**
- **2013** Eliasmith. *How to Build a Brain.* OUP. **[R]**
- **2013** Stewart & Eliasmith. *Realistic neurons can compute the operations needed by quantum probability theory and other vector symbolic architectures.* Behav. Brain Sci. 36(3):307–308. **[V]**
- **2013** Rigotti, Barak, Warden, Wang, Daw, Miller, Fusi. *The importance of mixed selectivity in complex cognitive tasks.* Nature 497:585–590. **[V]** High-dimensional random-like mixing in prefrontal cortex.
- **2013** Mikolov, Chen, Corrado, Dean. *Efficient estimation of word representations in vector space.* arXiv:1301.3781. **[R]**; Mikolov, Yih, Zweig. *Linguistic regularities in continuous space word representations.* NAACL. **[R]** Analogy as vector arithmetic on learned embeddings.
- **2013** Bordes, Usunier, García-Durán, Weston, Yakhnenko. *Translating embeddings for modeling multi-relational data.* NIPS. **[R]** TransE (additive binding).
- **2013** Wang. *Non-Axiomatic Logic.* World Scientific. **[R]** NARS.
- **2014** Babadi & Sompolinsky. *Sparseness and expansion in sensory representations.* Neuron 83:1213–1226. **[V]**
- **2014** Graves, Wayne, Danihelka. *Neural Turing machines.* arXiv:1410.5401. **[R]**
- **2015** Weston, Chopra, Bordes. *Memory networks.* ICLR (arXiv:1410.3916). **[R]**
- **2014** Kanerva. *Computing with 10,000-bit words.* Allerton. **[R]**
- **2015** Widdows & Cohen. *Reasoning with vectors: a continuous model for fast robust inference.* Logic J. IGPL 23:141–173. **[R]**

## Era 5 — 2016–2019: memory-augmented nets, dense associative memory, grid-like codes, fractional binding

- **2016** Krotov & Hopfield. *Dense associative memory for pattern recognition.* NeurIPS. **[V]** Polynomial/exponential interaction → capacity ≫ N.
- **2016** Nickel, Rosasco, Poggio. *Holographic embeddings of knowledge graphs.* AAAI. **[V]** HolE: circular correlation as relational binding in KG embeddings.
- **2016** Danihelka, Wayne, Uria, Kalchbrenner, Graves. *Associative long short-term memory.* ICML (arXiv:1602.03032). **[V]** Holographic redundant copies inside an RNN memory.
- **2016** Ba, Hinton, Mnih, Leibo, Ionescu. *Using fast weights to attend to the recent past.* NIPS (arXiv:1610.06258). **[R]**
- **2016** Graves et al. *Hybrid computing using a neural network with dynamic external memory.* Nature 538:471–476. **[R]** DNC.
- **2016** Arjovsky, Shah, Bengio. *Unitary evolution recurrent neural networks.* ICML, pp. 1120–1128. **[V]** Norm-preserving (unitary) recurrence.
- **2016** Wisdom, Powers, Hershey, Le Roux, Atlas. *Full-capacity unitary recurrent neural networks.* NeurIPS (arXiv:1611.00035). **[V]**
- **2016** Trouillon, Welbl, Riedel, Gaussier, Bouchard. *Complex embeddings for simple link prediction.* ICML. **[R]** ComplEx (phasor-like).
- **2016** Rahimi, Kanerva, Rabaey. *A robust and energy-efficient classifier using brain-inspired hyperdimensional computing.* ISLPED. **[R]**
- **2016** Santoro et al. *Meta-learning with memory-augmented neural networks.* ICML. **[R]**
- **2016** Constantinescu, O'Reilly, Behrens. *Organizing conceptual knowledge in humans with a gridlike code.* Science 352:1464–1468. **[V]**
- **2017** Demircigil, Heusel, Löwe, Upgang, Vermet. *On a model of associative memory with huge storage capacity.* J. Stat. Phys. 168:288–299. **[V]** Exponential capacity.
- **2017** Dasgupta, Stevens, Navlakha. *A neural algorithm for a fundamental computing problem.* Science 358:793–796. **[V]** Fly olfaction = sparse random expansion + WTA = LSH variant.
- **2017** Vaswani et al. *Attention is all you need.* NIPS. **[R]**
- **2017** Silver et al. *Mastering the game of Go without human knowledge.* Nature 550:354–359. **[R]** Self-play with an exact truth channel (game rules).
- **2017** Lake, Ullman, Tenenbaum, Gershman. *Building machines that learn and think like people.* Behav. Brain Sci. 40:e253. **[R]**
- **2017** Sabour, Frosst, Hinton. *Dynamic routing between capsules.* NIPS. **[R]**
- **2017** Litwin-Kumar, Harris, Axel, Sompolinsky, Abbott. *Optimal degrees of synaptic connectivity.* Neuron 93:1153–1164. **[V]** Dimension of granule/Kenyon-cell codes is maximised at observed sparse connectivity.
- **2017–18** Gorban & Tyukin. *Stochastic separation theorems.* Neural Netw. 94 **[V-t]**; *Blessing of dimensionality: mathematical foundations of the statistical physics of data.* Phil. Trans. R. Soc. A 376:20170237 (arXiv:1801.03421). **[V]**
- **2018** Frady, Kleyko, Sommer. *A theory of sequence indexing and working memory in recurrent neural networks.* Neural Comput. 30:1449–1513 (arXiv:1803.00412). **[V]** Random input weights + orthogonal recurrence implements VSA sequence coding; crosstalk-limited capacity.
- **2018** Banino et al. *Vector-based navigation using grid-like representations in artificial agents.* Nature 557:429–433. **[V]** Grid-like codes *emerge* from training for path integration.
- **2018** Cueva & Wei. *Emergence of grid-like representations by training recurrent neural networks to perform spatial localization.* ICLR. **[R]**
- **2018** Wu, Wayne, Graves, Lillicrap. *The Kanerva machine: a generative distributed memory.* ICLR. **[R]**
- **2018** Ha & Schmidhuber. *World models.* arXiv:1803.10122. **[R]**
- **2018** Masse, Grant, Freedman. *Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization.* PNAS 115:E10467. **[V]**
- **2018/19** Hawkins, Lewis, Klukas, Purdy, Ahmad. *A framework for intelligence and cortical function based on grid cells in the neocortex.* Front. Neural Circuits 12:121. **[V]**
- **2019** Ahmad & Scheinkman. *How can we be so dense? The benefits of using highly sparse representations.* arXiv:1903.11257. **[V]**
- **2019** Cheung, Terekhov, Chen, Agrawal, Olshausen. *Superposition of many models into one.* NeurIPS (arXiv:1902.05522). **[V]** Nearly-orthogonal context keys superimpose many networks in one weight set (Kanerva hetero-association).
- **2019** Komer, Stewart, Voelker, Eliasmith. *A neural representation of continuous space using fractional binding.* CogSci. **[V]** Spatial Semantic Pointers.
- **2019** Voelker, Kajić, Eliasmith. *Legendre memory units: continuous-time representation in recurrent neural networks.* NeurIPS. **[V]** Predecessor/special case of HiPPO-LegT → S4.
- **2019** Sorscher, Mel, Ganguli, Ocko. *A unified theory for the origin of grid cells through the lens of pattern formation.* NeurIPS (journal version Neuron 2023). **[V]** Non-negativity breaks symmetry toward hexagonal codes.
- **2019** Mitrokhin, Sutor, Fermüller, Aloimonos. *Learning sensorimotor control with neuromorphic sensors: toward hyperdimensional active perception.* Sci. Robot. 4(30). **[V]**
- **2019** Gosmann & Eliasmith. *Vector-derived transformation binding.* Neural Comput. 31(5):849–869. **[V]** VTB.
- **2019** Frady & Sommer. *Robust computation with rhythmic spike patterns.* PNAS 116(36) (arXiv:1901.07718). **[V]** Threshold phasor associative memory in spiking nets.
- **2019** Zeng, Chen, Cui, Yu. *Continual learning of context-dependent processing in neural networks.* Nat. Mach. Intell. 1:364–372. **[V]** Orthogonal weight modification.
- **2019** Gao, Xie, Zhu, Wu. *Learning grid cells as vector representation of self-position coupled with matrix representation of self-motion.* ICLR (arXiv:1810.05597). **[V]** Vector codes + matrix (group) action.
- **2019** Sun, Deng, Nie, Tang. *RotatE: knowledge graph embedding by relational rotation in complex space.* ICLR. **[R]** Relations as unit-modulus complex rotations = FHRR binding.
- **2019** Neubert, Schubert, Protzel. *An introduction to hyperdimensional computing for robotics.* KI – Künstl. Intell. 33. **[R]**
- **2019** Schlag, Smolensky, Fernandez, Jojic, Schmidhuber, Gao. *Enhancing the transformer with explicit relational encoding for math problem solving.* arXiv:1910.06611. **[R]** TP-Transformer.
- **2019** Lample, Sablayrolles, Ranzato, Denoyer, Jégou. *Large memory layers with product keys.* NeurIPS. **[R]** Factored addressing.
- **2019** Chollet. *On the measure of intelligence.* arXiv:1911.01547. **[V]** Intelligence = skill-acquisition efficiency under priors; introduces ARC.
- **2019** Sutton. *The bitter lesson.* (essay) **[R]**
- **2019** Belkin, Hsu, Ma, Mandal. *Reconciling modern machine-learning practice and the classical bias–variance trade-off.* PNAS 116:15849. **[R]**

## Era 6 — 2020–2022: attention = associative memory; resonators; superposition; grokking; surveys

- **2020** Kaplan et al. *Scaling laws for neural language models.* arXiv:2001.08361. **[V]**
- **2020** Frady, Kent, Olshausen, Sommer. *Resonator networks, 1.* Neural Comput. 32(12):2311–2331. **[V]**; Kent, Frady, Sommer, Olshausen. *Resonator networks, 2.* Neural Comput. 32(12):2332–2388 (preprint arXiv:1906.11684). **[V]** Factorisation by iterated unbind + cleanup; operational capacity grows ∝ D².
- **2020** Karunaratne, Le Gallo, Cherubini, Benini, Rahimi, Sebastian. *In-memory hyperdimensional computing.* Nat. Electron. 3. **[V]** 760,000 phase-change devices.
- **2020** Greff, van Steenkiste, Schmidhuber. *On the binding problem in artificial neural networks.* arXiv:2012.05208. **[V]** Segregation / representation / composition.
- **2020** Wang & Isola. *Understanding contrastive representation learning through alignment and uniformity on the hypersphere.* ICML (arXiv:2005.10242). **[V]**
- **2020** Papyan, Han, Donoho. *Prevalence of neural collapse during the terminal phase of deep learning training.* PNAS 117(40):24652–24663 (arXiv:2008.08186). **[V]** Class means converge to a simplex equiangular tight frame — the optimal quasi-orthogonal codebook.
- **2020** Whittington et al. *The Tolman-Eichenbaum machine.* Cell 183:1249–1263. **[V]**
- **2020** Bernardi, Benna, Rigotti, Munuera, Fusi, Salzman. *The geometry of abstraction in the hippocampus and prefrontal cortex.* Cell 183. **[V]**
- **2020** Schlegel, Neubert, Protzel. *A comparison of vector symbolic architectures.* Artif. Intell. Rev. (arXiv:2001.11797). **[V]**
- **2020** Locatello et al. *Object-centric learning with slot attention.* NeurIPS. **[R]**
- **2020–21** *Variable binding for sparse distributed representations: theory and applications.* (arXiv:2009.06734) **[V-t]**
- **2021** Ramsauer et al. *Hopfield networks is all you need.* ICLR. **[V]** Softmax attention = modern-Hopfield update; exponential capacity.
- **2021** Schlag, Irie, Schmidhuber. *Linear transformers are secretly fast weight programmers.* ICML (PMLR 139:9355–9366). **[V]** Linear attention = additive outer-product programming (TPR).
- **2021** Bricken & Pehlevan. *Attention approximates sparse distributed memory.* NeurIPS (arXiv:2111.05498). **[V]**
- **2021** Webb, Sinha, Cohen. *Emergent symbols through binding in external memory.* ICLR (arXiv:2012.14601). **[V]** ESBN.
- **2021** Thomas, Dasgupta, Rosing. *A theoretical perspective on hyperdimensional computing.* J. Artif. Intell. Res. 72:215–249. **[V]** HDC ≈ random-feature kernels.
- **2021–23** Kleyko, Rachkovskij, Osipov, Rahimi. *A survey on hyperdimensional computing aka vector symbolic architectures, Part I* ACM Comput. Surv. 55(6) (arXiv:2111.06077) **[V]** and *Part II* ACM Comput. Surv. (arXiv:2112.15424; doi:10.1145/3558000) **[V]**.
- **2021** Frady, Kleyko, Kymn, Olshausen, Sommer. *Computing on functions using randomized vector representations.* arXiv:2109.03429. **[V]** Fractional power encoding ↔ kernels.
- **2021** Ganesan et al. *Learning with holographic reduced representations.* NeurIPS (arXiv:2109.02157). **[V]**
- **2021** Geva, Schuster, Berant, Levy. *Transformer feed-forward layers are key-value memories.* EMNLP, pp. 5484–5495. **[V]**
- **2021** Elhage et al. *A mathematical framework for transformer circuits.* Transformer Circuits Thread. **[V]**
- **2021** Su, Lu, Pan, Murtadha, Wen, Liu. *RoFormer: enhanced transformer with rotary position embedding.* arXiv:2104.09864. **[V]** Per-pair phase rotation e^{imθ} (FHRR-style position binding).
- **2021** Choromanski et al. *Rethinking attention with Performers.* ICLR (arXiv:2009.14794). **[V]** Random-feature (RFF-family) attention.
- **2021** Bronstein, Bruna, Cohen, Veličković. *Geometric deep learning: grids, groups, graphs, geodesics, and gauges.* arXiv:2104.13478. **[V]**
- **2021** Ellis et al. *DreamCoder: bootstrapping inductive program synthesis with wake-sleep library learning.* PLDI, pp. 835–850. **[V]**
- **2021** Kleyko et al. *Vector symbolic architectures as a computing framework for emerging hardware.* Proc. IEEE (arXiv:2106.05268). **[V]**
- **2021** Hinton. *How to represent part-whole hierarchies in a neural network.* arXiv:2102.12627. **[R]** GLOM.
- **2021** VanRullen & Kanai. *Deep learning and the global workspace theory.* Trends Neurosci. 44(9):692–704. **[V]**
- **2022** Whittington, Warren, Behrens. *Relating transformers to models and neural representations of the hippocampal formation.* ICLR (arXiv:2112.04035). **[V]** Transformer + recurrent position codes yields place/grid-like units.
- **2022** Goyal et al. *Coordination among neural modules through a shared global workspace.* ICLR (arXiv:2103.01197). **[V]**
- **2022** Gu, Goel, Ré. *Efficiently modeling long sequences with structured state spaces.* ICLR (arXiv:2111.00396). **[V]**
- **2022** Olsson et al. *In-context learning and induction heads.* arXiv:2209.11895. **[V]**
- **2022** Elhage et al. *Toy models of superposition.* arXiv:2209.10652. **[V]** Features stored as nearly-orthogonal directions in superposition.
- **2022** Power, Burda, Edwards, Babuschkin, Misra. *Grokking: generalization beyond overfitting on small algorithmic datasets.* arXiv:2201.02177. **[V]**
- **2022** Hoffmann et al. *Training compute-optimal large language models.* arXiv:2203.15556. **[V]**
- **2022** Wei et al. *Emergent abilities of large language models.* TMLR (arXiv:2206.07682). **[V]**
- **2022** Meng, Bau, Andonian, Belinkov. *Locating and editing factual associations in GPT.* NeurIPS. **[V]**
- **2022** Hersche, Karunaratne, Cherubini, Benini, Sebastian, Rahimi. *Constrained few-shot class-incremental learning.* CVPR. **[V]** Hyperdimensional embedding; 423 novel classes over 1,200 base with < 1.6% accuracy drop (authors' claim).
- **2022** Heddes et al. *Torchhd: an open source Python library to support research on hyperdimensional computing and vector symbolic architectures.* JMLR 2023 (arXiv:2205.09208). **[V]**
- **2022** Smolensky, McCoy, Fernandez, Goldrick, Gao. *Neurocompositional computing: from the central paradox of cognition to a new generation of AI systems.* AI Magazine 43(3):308–322 (arXiv:2205.01128). **[V]**
- **2022** Delétang et al. *Neural networks and the Chomsky hierarchy.* ICLR 2023 (arXiv:2207.02098). **[V]** 20,910 models; some architectures never generalise out of distribution regardless of data.
- **2022** Ito, Klinger, Schultz, Murray, Cole, Rigotti. *Compositional generalization through abstract representations in human and artificial neural networks.* NeurIPS (arXiv:2209.07431). **[V]**
- **2022** Renner et al. *Neuromorphic visual scene understanding with resonator networks.* arXiv:2208.12880. **[V]**
- **2022** Gardner, Hermansen, Pachitariu, Burak, Dunn, Moser, Moser. *Toroidal topology of population activity in grid cells.* Nature 602. **[V]**
- **2022** LeCun. *A path towards autonomous machine intelligence.* v0.9.2. **[V]** JEPA / world-model position paper.
- **2022** Flesch, Juechems, Dumbalska, Saxe, Summerfield. *Orthogonal representations for robust context-dependent task performance in brains and neural networks.* Neuron 110(24):4212–4219. **[V]**
- **2022–23** Whittington, Dorrell, Ganguli, Behrens. *Disentanglement with biological constraints: a theory of functional cell types.* ICLR (arXiv:2210.01768). **[V-t]**
- **2022** Xu, Gao, Wu et al. *Conformal isometry of Lie group representation in recurrent network of grid cells.* arXiv:2210.02684. **[V-t]**

## Era 7 — 2023–2024: networks rediscover algebra; VSA enters attention; capacity laws; neuro-vector-symbolic systems

- **2023** Nanda, Chan, Lieberum, Smith, Steinhardt. *Progress measures for grokking via mechanistic interpretability.* ICLR (arXiv:2301.05217). **[V]** Modular addition solved with Fourier features (phases).
- **2023** Chughtai, Chan, Nanda. *A toy model of universality: reverse engineering how networks learn group operations.* ICML (arXiv:2302.03025). **[V]** Group composition via representation theory.
- **2023** Zhong, Liu, Tegmark, Andreas. *The clock and the pizza.* NeurIPS (arXiv:2306.17844). **[V]** Different seeds find different algorithms for the same task.
- **2023** Alam, Raff, Biderman, Oates, Holt. *Recasting self-attention with holographic reduced representations.* ICML (arXiv:2305.19534). **[V]** Hrrformer: O(T·H log H) attention via HRR binding.
- **2023** Hersche, Zeqiri, Benini, Sebastian, Rahimi. *A neuro-vector-symbolic architecture for solving Raven's progressive matrices.* Nat. Mach. Intell. 5:363–375 (arXiv:2203.04571). **[V]**
- **2023** Bietti, Cabannes, Bouchacourt, Jégou, Bottou. *Birth of a transformer: a memory viewpoint.* NeurIPS (arXiv:2306.00802). **[V]** Weights as associative memories of near-orthogonal embeddings; induction-head formation.
- **2023** Dziri et al. *Faith and fate: limits of transformers on compositionality.* NeurIPS (arXiv:2305.18654). **[V]**
- **2023** Lake & Baroni. *Human-like systematic generalization through a meta-learning neural network.* Nature 623:115–121. **[V]**
- **2023** Schaeffer, Miranda, Koyejo. *Are emergent abilities of large language models a mirage?* NeurIPS (outstanding paper). **[V]**
- **2023** Michaud, Liu, Girit, Tegmark. *The quantization model of neural scaling.* NeurIPS. **[V]**
- **2023** Soulos, Hu, McCurdy, Chen, Fernandez, Smolensky, Gao. *Differentiable tree operations promote compositional generalization.* ICML (arXiv:2306.00751). **[V]**
- **2023** Langenegger, Karunaratne, Hersche, Benini, Sebastian, Rahimi. *In-memory factorization of holographic perceptual representations.* Nat. Nanotechnol. 18 (arXiv:2211.05052). **[V]**
- **2023** Bricken et al. *Towards monosemanticity: decomposing language models with dictionary learning.* Anthropic. **[V]**; Cunningham et al. *Sparse autoencoders find highly interpretable features in language models.* arXiv:2309.08600. **[V]**
- **2023** Orvieto et al. *Resurrecting recurrent neural networks for long sequences.* ICML (arXiv:2303.06349). **[V]** Complex-diagonal linear recurrences (the LRU) — phasor dynamics.
- **2023–24** Delétang et al. *Language modeling is compression.* ICLR 2024 (arXiv:2309.10668). **[V]**
- **2023** Kleyko et al. *Efficient decoding of compositional structure in holistic representations.* arXiv:2305.16873. **[V-t]**
- **2023** Gu & Dao. *Mamba: linear-time sequence modeling with selective state spaces.* arXiv:2312.00752. **[R]**
- **2023** Poli et al. *Hyena hierarchy: towards larger convolutional language models.* ICML. **[R]** Long convolutions (HRR-style binding) as token mixing.
- **2023–24** Park, Choe, Veitch. *The linear representation hypothesis and the geometry of large language models.* ICML 2024 (arXiv:2311.03658). **[V]**; Park, Choe, Jiang, Veitch. *The geometry of categorical and hierarchical concepts in large language models.* ICLR 2025 (oral) (arXiv:2406.01506). **[V]**
- **2023–24** Feng & Steinhardt. *How do language models bind entities in context?* ICLR 2024 (arXiv:2310.17191). **[V]** Binding-ID vectors form a continuous subspace.
- **2023–24** Cabannes, Dohmatob, Bietti. *Scaling laws for associative memories.* ICLR 2024. **[V]**
- **2023–24** Webb, Frankland, Altabaa, Segert, Krishnamurthy, Campbell, Russin, Giallanza, Dulberg, O'Reilly, Lafferty, Cohen. *The relational bottleneck as an inductive bias for efficient abstraction.* Trends Cogn. Sci. 28(9):829–843 (arXiv:2309.06629). **[V]**
- **2023** Johnston & Fusi. *Abstract representations emerge naturally in neural networks trained to perform multiple tasks.* Nat. Commun. 14:1040. **[V]**
- **2024** Hänni, Mendel, Vaintrob, Chan. *Mathematical models of computation in superposition.* (ICML MI workshop) **[R]**
- **2024** Sun et al. *Learning to (learn at test time): RNNs with expressive hidden states.* arXiv:2407.04620. **[R]** TTT.
- **2024** Allen-Zhu & Li. *Physics of language models: part 3.3, knowledge capacity scaling laws.* arXiv:2404.05405. **[V]** ≈ 2 bits of knowledge per parameter.
- **2024** Huh, Cheung, Wang, Isola. *The platonic representation hypothesis.* ICML (arXiv:2405.07987). **[V]**
- **2024** Loshchilov, Hsieh, Sun, Ginsburg. *nGPT: normalized transformer with representation learning on the hypersphere.* arXiv:2410.01131. **[V]**
- **2024** Kymn et al. *Computing with residue numbers in high-dimensional representation.* Neural Comput. 37(1) (arXiv:2311.04872). **[V]**
- **2024** Kymn, Mazelet, Thomas, Kleyko, Frady, Sommer, Olshausen. *Binding in hippocampal-entorhinal circuits enables compositionality in cognitive maps.* NeurIPS 37:39128–39157. **[V]**
- **2024** Renner, Supic, Danielescu, Indiveri, Frady, Sommer, Sandamirskaya. *Visual odometry with neuromorphic resonator networks.* Nat. Mach. Intell. 6:653–663. **[V]**
- **2024** Karunaratne, Hersche, Sebastian, Rahimi. *On the role of noise in factorizers for disentangling distributed representations.* arXiv:2412.00354. **[V]** Noise extends operational capacity ≥ 50× over baseline resonators (authors' claim).
- **2024** Shumailov et al. *AI models collapse when trained on recursively generated data.* Nature. **[V]**
- **2024** Trinh, Wu, Le, He, Luong. *Solving olympiad geometry without human demonstrations.* Nature. **[V]** Neural LM guides a symbolic deduction engine; synthetic data.
- **2024** Yang, Wang, Zhang, Shen, Kim. *Parallelizing linear transformers with the delta rule over sequence length.* NeurIPS (arXiv:2406.06484). **[V]**
- **2024** Mejri, Amarnath, Chatterjee. *LARS-VSA: a vector symbolic architecture for learning with abstract rules.* arXiv:2405.14436. **[V]**
- **2024** *Generalized holographic reduced representations (GHRR).* arXiv:2405.09689. **[V]** Non-commutative binding; authors report improved LM performance when replacing attention (not independently checked).
- **2024** Hersche et al. *Probabilistic abduction for visual abstract reasoning via learning rules in vector-symbolic architectures.* arXiv:2401.16024. **[V-t]**
- **2024** Chollet et al. *ARC Prize 2024: technical report.* arXiv:2412.04604. **[V-t]**
- **2024** Templeton et al. *Scaling monosemanticity.* Anthropic. **[R]**
- **2024** He. *Mixture of a million experts.* arXiv:2407.04153. **[R]**
- **2024** Berges et al. *Memory layers at scale.* arXiv:2412.09764. **[R]**
- **2023–24** Sorrenti et al. *Wake-sleep consolidated learning.* arXiv:2401.08623. **[V]**
- **2023–24** VSA tooling/theory seen as titles: *Self-attention based semantic decomposition in VSAs* arXiv:2403.13218 **[V-t]**; *Compositional factorization of visual scenes with convolutional sparse coding and resonator networks* arXiv:2404.19126 **[V-t]**; *A Walsh–Hadamard derived linear VSA* arXiv:2410.22669 **[V-t]**; *Capacity analysis of VSAs* arXiv:2301.10352 **[V-t]**; *Factorizers for distributed sparse block codes* arXiv:2303.13957 **[V-t]**; *Integer factorization with compositional distributed representations* arXiv:2203.00920 **[V-t]**; *Analyzing the capacity of distributed vector representations to encode spatial information* arXiv:2010.00055 **[V-t]**.

## Era 8 — 2025 → 3 Oct 2026: VSA ideas inside LLMs, symbolic structure found in trained networks, ARC and world models

### 8a. VSA / HDC proper
- **2025** Shaw, Furlong, Anderson, Orchard. *Developing a foundation of vector symbolic architectures using category theory.* arXiv:2501.05368. **[V]** VSA ops as right Kan extensions of the external tensor product; reduce to element-wise operations.
- **2025** Bronzini, Nicolini, Lepri, Staiano, Passerini. *Hyperdimensional probe: decoding LLM representations via vector symbolic architectures.* arXiv:2509.25045. **[V]**
- **2025** Dhayalkar. *Attention as binding: a vector-symbolic perspective on transformer reasoning.* arXiv:2512.14709. **[V]** *(theory only)* Maps attention/residual stream to bind/unbind/superpose; proposes a "VSA-likeness" metric.
- **2025** Joffe & Eliasmith. *Vector symbolic algebras for the Abstraction and Reasoning Corpus.* arXiv:2511.08747. **[V]** ARC Prize 2025 runner-up paper: 10.8% ARC-AGI-1 train, **3.0% eval**; 94.5% Sort-of-ARC, 83.1% 1D-ARC.
- **2025** Hanley, Tomkins-Flanagan, Kelly. *Hey Pentti, we did it! A fully vector-symbolic Lisp.* arXiv:2510.17889 **[V-t]**; *…(more of) it!: a vector-symbolic Lisp with residue arithmetic.* arXiv:2511.08767 **[V]** Lisp 1.5 encoded in VSA state.
- **2025–26** Deng & Raviv. *Efficient vector symbolic architectures from histogram recovery.* arXiv:2511.01838. **[V]** Reed–Solomon + Hadamard codes; list-decoding recovery.
- **2026** Chung et al. *Geometric priors for generalizable world models via vector symbolic architecture.* arXiv:2602.21467. **[V]** Learnable FHRR encoders; transitions = element-wise complex multiplication; 87.5% zero-shot accuracy on unseen state-action pairs, +53.6% on 20-step rollouts vs MLP, 4× noise robustness (authors' claims; discrete grid worlds only).
- **2026** Snyder, Capodieci, Gorsich, Parsa. *HyperSpace: a generalized framework for spatial encoding in hyperdimensional representations.* arXiv:2604.15113 **[V]**; *SRMU: relevance-gated updates for streaming hyperdimensional memories.* arXiv:2604.15121 **[V]**.
- **2026** Poore. *A wave-geometric duality for hyperdimensional computing.* arXiv:2604.22863. **[V]**
- **2026** *qFHRR: rethinking Fourier holographic reduced representations through quantized phase and integer arithmetic.* arXiv:2604.25939. **[V]**
- **2026** Chuma, Otsuk, Sato. *How to build Marcus's algebraic mind: algebro-deterministic substrate over Galois fields.* arXiv:2605.21379 **[V]** (authors state it does not outperform LLMs); companion arXiv:2607.16573 **[V-t]**.
- **2026** Leonhart. *Sutra: tensor-op RNNs as a compilation target for vector symbolic architectures.* arXiv:2605.20919. **[V]**
- **2026** Chouinard. *On periodic distributed representations using Fourier embeddings.* arXiv:2605.10818. **[V]**
- **2026** Habashy & Eliasmith. *Geodesic flow matching for denoising high-dimensional structured representations.* arXiv:2606.00248. **[V]** Manifold-aware cleanup on the SSP torus.
- **2026** Pence, Yamada, Singh. *Recursive binding on a budget: subspace carving in order-p tensor memories.* arXiv:2606.11391. **[V]** Orthogonal subspace carving; TPR as a special case of Clifford-algebra binding.
- **2026** Velasquez Olivera, Thomas, Saad. *Disentanglement with holographic reduced representations.* arXiv:2606.09725. **[V]**
- **2026** Kumar. *Holographic memory for zero-shot compositional reasoning in knowledge graphs: a mechanistic study of where and why it fails.* arXiv:2606.24948. **[V]** **Negative result:** HRR/FHRR retrieve single hops (MRR ≈ 0.35) but compose at chance; bottleneck is retrieval capacity under superposition.
- **2026** Overmann. *Creating intelligence: a computational foundation for AGI.* arXiv:2606.31819. **[V]** *(theory only)* Set theory + HDC; no experiments.
- **2026** Dhayalkar. *Overcoming the impedance mismatch: a theoretical roadmap for fusing foundation models and knowledge graphs.* arXiv:2606.15656. **[V]** *(theory only)*
- **2026** Schlegel, Rachkovskij, Kleyko, Loutfi, Streif, Osipov. *Real-valued hyperdimensional sequence representations with Hadamard product binding and shift equivariance.* arXiv:2608.28334. **[V]**
- **2026** *Vector symbolic policy gradient.* arXiv:2608.18404. **[V]** Actions as unit hypervectors; update = advantage-weighted bundling.
- **2026** Abid & Furlong. *Learning holographic reduced representations with Clifford variational autoencoders.* arXiv:2609.28409. **[V]** Learns codes on a Clifford torus; beats Gaussian/hyperspherical VAEs on bind/unbind and bundle-capacity benchmarks (authors' claim).
- **2026** *Superposed inference for hyperdimensional computing (SupHDC).* arXiv:2609.32320. **[V]**
- **2026** *Bridging the gap between hyperdimensional computing and kernel methods via the Nyström method.* AAAI 2025; arXiv:2608.06860. **[V-t]**
- **2026** *A unifying perspective on language model representations: from filler-role structure to mechanistic interpretability.* arXiv:2608.29034. **[V]** TPR as common root of SAEs, probes, additive analogies, activation patching.
- **2026** *HoloByte: continuous hyperspherical distillation for tokenizer-free modeling.* arXiv:2603.16917. **[V]**
- **2026** Seen as titles only (**[V-t]**): *Derandomizing dense binary hypervector codebooks for quantized scalars* arXiv:2609.38471; *Learning a vector-symbolic model for socio-cultural tasks* arXiv:2608.02807; *Gram-Space: structure-preserving codebook compression for memory-efficient neuro-symbolic AI* arXiv:2608.01528; *Holographic invariant storage: design-time safety contracts via VSAs* arXiv:2603.13558; *hdlib 2.0* arXiv:2601.02509; *Hypervectors from optical disorder* arXiv:2609.24108; *Hyperdimensional computing for structured querying on tabular data embeddings* arXiv:2606.13871.
- **2025** **[W]** Arndt. *Holographic knowledge manifolds: a novel pipeline for continual learning without catastrophic forgetting in large language models.* arXiv:2509.10518 — **withdrawn** (the abstract states significant errors). Do not cite as evidence.

### 8b. Symbolic / algebraic structure found in trained networks
- **2025** Yang, Campbell, Huang, Wang, Cohen, Webb. *Emergent symbolic mechanisms support abstract reasoning in large language models.* ICML (arXiv:2502.20332). **[V]** Symbol-abstraction heads → symbolic-induction heads → retrieval heads.
- **2025** Wu, Geiger, Millière. *How do transformers learn variable binding in symbolic programs?* ICML (arXiv:2505.20896). **[V]** Three training phases; residual stream used as addressable memory.
- **2025** Smolensky, Fernandez, Zhou, Opper, Davies, Gao. *Mechanisms of symbol processing for in-context learning in transformer networks.* J. Artif. Intell. Res. (arXiv:2410.17498). **[V]** Addresses computability, **not** learnability.
- **2025** *Discovering group structures via unitary representation learning.* ICLR 2025. **[V]** Differentiable approach using matrix representations of finite groups with a unitarity regulariser, so valid group structure emerges without evaluating the axioms (authors not checked).
- **2025** Nichani, Lee, Bietti. *Understanding factual recall in transformers via associative memories.* ICLR (arXiv:2412.06538). **[V]** Associative-memory capacity scales linearly in parameters.
- **2026** McCoy, Soulos, Linzen, Smolensky. *The emergent symbolic structure of artificial neural networks.* arXiv:2608.29530. **[V]** Vector representations of many networks (incl. LLMs) approximated in closed form by symbolic structures; interventions predictably change behaviour.
- **2026** Lee, Viégas, Wattenberg. *Tensor product representation probes reveal shared structure across linear directions.* arXiv:2605.09967. **[V]** Othello-GPT only.
- **2026** Vural, Bietti, Soltanolkotabi, Wu. *Learning to recall with transformers beyond orthogonal embeddings.* arXiv:2603.15923. **[V]**
- **2026** Ravfogel, Yehudai, Bruna, Bietti. *Geometric factual recall in transformers.* arXiv:2605.12426. **[V]**
- **2026** Zhang et al. *Weights to code: extracting interpretable algorithms from the Discrete Transformer.* arXiv:2601.05770. **[V]** Superposition hinders symbolic recovery.
- **2025** Pham, Raya, Negri, Zaki, Ambrogioni, Krotov. *Memorization to generalization: emergence of diffusion models from associative memory.* ICLR (oral) (arXiv:2505.21777). **[V]**
- **titles only [V-t]:** *Factual recall in linear associative memories: sharp asymptotics* arXiv:2605.10795; *Sharp capacity thresholds in linear associative memory* arXiv:2605.05189; *Muon in associative memory learning* arXiv:2602.05725; *Visual symbolic mechanisms* arXiv:2506.15871; *Mixing mechanisms: how language models retrieve bound entities in-context* arXiv:2510.06182; *Revealing algorithmic deductive circuits for logical reasoning* arXiv:2605.27824; *One mechanism for many mental spaces* arXiv:2607.10248; *Cultural binding heads in language models* arXiv:2605.28543; *Whither symbols in the era of advanced neural networks?* arXiv:2508.05776.

### 8c. Superposition, geometry, scaling, representation convergence
- **2025** Liu, Liu, Gore. *Superposition yields robust neural scaling.* NeurIPS (oral; best-paper runner-up) (arXiv:2505.10465). **[V]** Strong superposition ⇒ loss ∝ 1/(model dimension) from geometric overlap.
- **2026** Ivanitskiy, Jasper, King, Mixon. *Towards a mathematical theory of superposition.* arXiv:2608.27540. **[V]** Random low-coherence dictionaries: support recovery to sparsity ≈ d / log n.
- **2026** Gröger, Wen, Brbić. *Revisiting the platonic representation hypothesis: an Aristotelian view.* arXiv:2602.14486. **[V]** Global spectral similarity largely vanishes after calibration; local neighbourhoods converge.
- **2026** Li et al. *Platonic representation hypothesis on world models.* arXiv:2608.23720. **[V]**
- **2026** Simon, Kunin et al. *There will be a scientific theory of deep learning.* arXiv:2604.21691. **[V]**
- **titles only [V-t]:** *From data statistics to feature geometry: how correlations shape superposition* arXiv:2603.09972; *Geometric limits of knowledge distillation: a minimum-width theorem via superposition theory* arXiv:2604.04037; *Hidden not deleted: how networks suppress entangled features* arXiv:2609.27593; *Superposition as lossy compression* arXiv:2512.13568.
- **2025** Bernstein (Thinking Machines). *Modular manifolds.* Blog, Sept 2025. **[V]** Manifold Muon on the Stiefel manifold; hypersphere constraint as the simplest case. Titles only **[V-t]**: *Training nGPT* arXiv:2608.01284; *Rethinking language model scaling under transferable hypersphere optimization* arXiv:2603.28743; *GeoNorm* arXiv:2601.22095; *Muon on the Stiefel manifold admits an exact closed-form update* arXiv:2608.06218; *HELM: hyperbolic LLMs via mixture-of-curvature experts* arXiv:2505.24722; *Hyperbolic large language models* arXiv:2509.05757.
- **2026** *Semigroup-JEPA: latent dynamics consistency for zero-shot physics generalization.* arXiv:2609.10464. **[V]** Action-conditioned autoregressive rollout enforces a semigroup structure; up to 2× lower open-loop error than DINO-WM; gain mainly from the encoder.
- **2026** Liu Hung Ming. *Probing the latent world: emergent discrete symbols and physical structure in latent representations.* arXiv:2603.20327. **[V]** V-JEPA 2 latents: compact, graded rather than categorical.

### 8d. Memory, test-time learning, new architectures
- **2025** Behrouz, Zhong, Mirrokni. *Titans: learning to memorize at test time.* arXiv:2501.00663. **[V]**; *It's all connected: a journey through test-time memorization, attentional bias, retention, and online optimization.* arXiv:2504.13173 **[V]**; *ATLAS* arXiv:2505.23735 **[V-t]**; *Nested learning: the illusion of deep learning architectures.* NeurIPS (arXiv:2512.24695) **[V]**; *Test-time regression: a unifying framework for designing sequence models with associative memory.* arXiv:2501.12352 **[V-t]**.
- **2026** DeepSeek-AI. *Conditional memory via scalable lookup: a new axis of sparsity for large language models (Engram).* arXiv:2601.07372. **[V]** O(1) hashed n-gram embedding lookup; 27B scale beats iso-parameter/iso-FLOP MoE (authors' claim).
- **2025** *The Dragon Hatchling: the missing link between the transformer and models of the brain.* arXiv:2509.26507. **[V]** Sparse positive activations; Hebbian working memory.
- **2025** Darlow, Regan, Risi, Seely, Jones. *Continuous thought machines.* arXiv:2505.05522. **[V]** Neural synchronisation as latent representation.
- **2025** Miyato, Löwe, Geiger, Welling. *Artificial Kuramoto oscillatory neurons.* ICLR (oral). **[V]**
- **2026** *Phasor agents: oscillatory graphs with three-factor plasticity and sleep-staged learning.* arXiv:2601.04362. **[V]** Wake/sleep separation reported to expand stable learning by 67%; REM-like replay +45.5 pp on mazes (authors' claims).
- **2025** Gershman, Fiete, Irie. *Key-value memory in the brain.* Neuron (arXiv:2501.02950). **[V]**
- **2026** *Geometric phase transition enables extreme hippocampal memory capacity.* arXiv:2605.17199. **[V]**
- **2026** Park. *Position: hippocampal explicit memory is the cornerstone for AGI.* ICML 2026 position track (arXiv:2606.11245). **[V]** *(position paper; no architecture)*
- **titles only [V-t]:** *Sparse Delta Memory: scaling the state of linear RNNs through sparsity* arXiv:2607.07386; *Memory for large language models* arXiv:2607.25380.

### 8e. Reasoning, general intelligence, benchmarks
- **2025** Wang et al. *Hierarchical reasoning model.* arXiv:2506.21734. **[V]** 27 M parameters; authors report 40.3% ARC-AGI-1; ARC Prize's independent check reported ≈ 32% (pass@2) on ARC-AGI-1 and ≈ 2% on ARC-AGI-2 (secondary report).
- **2025** Jolicoeur-Martineau. *Less is more: recursive reasoning with tiny networks (TRM).* arXiv:2510.04871. **[V]**; analysis arXiv:2512.11847 **[V-t]**.
- **2026** Freinschlag, Bertram, Kobler, Mayr, Klambauer. *Symbol-equivariant recurrent reasoning models.* arXiv:2603.02193. **[V]** Architectural permutation-equivariance; 2 M parameters; Sudoku 4×4 → 25×25 generalisation (authors' claim).
- **2025** Yue et al. *Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model?* NeurIPS (best-paper runner-up) (arXiv:2504.13837). **[V]**
- **2025** DeepSeek-AI. *DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning.* Nature 645 (arXiv:2501.12948). **[V]** Rule-based verifiable rewards; no neural reward model.
- **2025** Hafner, Pasukonis, Ba, Lillicrap. *Mastering diverse control tasks through world models.* Nature 640:647–653. **[V]**
- **2025** Chollet. *ARC-AGI-2: a new challenge for frontier AI reasoning systems.* arXiv:2505.11831. **[V-t]**
- **2026** Chollet, Knoop, Kamradt, Landers. *ARC Prize 2025: technical report.* arXiv:2601.10904. **[V]** 1,455 teams; best private ARC-AGI-2 score 24%; "refinement loops" are the defining theme; ≈ 7 M-parameter zero-pretraining test-time-training systems are competitive; frontier reasoning remains knowledge-bound.
- **2026** *ARC-AGI-3: a new challenge for frontier agentic intelligence.* arXiv:2603.24621. **[V-t]** Launched 2026-03-25; press reports launch-day frontier-LLM scores below 0.4% (secondary sources; confirm at arcprize.org).
- **2026** Vahdati, Aioanei, Suresh, Lehmann. *The ARC of progress towards AGI: a living survey of abstraction and reasoning.* arXiv:2603.13372. **[V]** 82 approaches; all paradigms drop 2–3× from ARC-AGI-1 to -2; test-time adaptation and refinement loops are critical.
- **2025** Silver & Sutton. *Welcome to the era of experience.* (essay) **[R]**
- **2023** Morris et al. *Levels of AGI: operationalizing progress on the path to AGI.* arXiv:2311.02462. **[R]**
