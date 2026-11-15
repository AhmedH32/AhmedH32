# Ahmed Hassan

**Computer Engineering @ AASTMT (Rank #1, GPA 3.94) | 3× ECPC National Finalist | Codeforces Expert (1618)**  
Focusing on **ML Systems**, **High-Throughput LLM Serving**, **Compound AI Architectures**, and **Algorithmic Optimization**.

---

### Core Focus & Systems Engineering

- **High-Throughput LLM Serving & Hardware-Aware Execution:** Production inference optimization using **vLLM** (`AWQ-Marlin` W4A16 weight-unpacking dispatch, `FlashAttention-2`, piecewise/full `CUDA Graph` capture, and Triton JIT warmup), PagedAttention GQA KV-cache sizing, and multi-tier **Radix Prefix Caching** (`<50ms` warm TTFT; `130+ tok/s` single-stream and `3,900+ tok/s` batched throughput).

- **Compound AI DAGs, System-1 Routing & Hybrid RAG:** Decoupling sub-50ms bidirectional System-1 decision encoders (`UltraBERT-380M` under `bfloat16` Tensor Core autocast) from System-2 LLM synthesis; two-stage retrieval combining dense bi-encoders (`BGE-base`, `ChromaDB`) with temperature-scaled Cross-Encoder reranking (`BGE-reranker`) and PEFT/QLoRA synthetic alignment.

- **GPU Memory Profiling & Production MLOps:** Multi-model single-GPU co-location across Turing (`sm_75`), Ampere (`sm_86`), and Ada (`sm_89`) architectures; bounding dynamic activation VRAM spikes via stage-decoupled `asyncio` admission semaphores; multi-profile **Docker Compose** (`8GB` consumer vs. `24GB–48GB` datacenter GPUs), serverless GPU orchestration (**Modal**), and automated **GitHub Actions** hardware SLO gates.

- **Low-Level Algorithmic Rigor:** 1,500+ competitive programming problems solved in Modern C++; specialized in graph theory, dynamic programming, segment trees with lazy propagation, tree query decompositions (HLD, Centroid), and cache-aware asymptotic optimization.

---

### Featured Systems & Research Repositories

- [**ai-gym-coach**](https://github.com/AhmedH32/ai-gym-coach) ([**Live Web App**](https://ai-gym-coach-nine.vercel.app) · [**API `/docs`**](https://ahmed-h-moustafa2003--ai-gym-coach-v2-gymcoachservice-api.us-east.modal.run/docs)) — Compound AI biomechanical RAG & workout inference engine co-locating a 380M System-1 router (`Vela-3`), 2-stage Cross-Encoder RAG (`ChromaDB`, `100%` Recall@3, `0.958` MRR), and `Qwen2.5-7B-AWQ` (`vLLM` `awq_marlin`) on a single GPU. Achieves a **23.5×–34.0× E2E latency reduction** (`3.10s` on L40S), **48.5ms warm TTFT**, and **3,902.5 tok/s** across 16 concurrent users with zero OOMs.

- [**earthquake-tsunami-prediction**](https://github.com/AhmedH32/earthquake-tsunami-prediction) — Cost-asymmetric ML pipeline resolving spherical anti-meridian distortions via 3D Cartesian manifold projections and calibrated decision thresholding ($F_2 = 0.922$, `96.6%` recall).

- [**elderly-assist-system**](https://github.com/AhmedH32/elderly-assist-system) — Multi-threaded edge perception engine decoupling real-time YOLO pose estimation from kinematic fall heuristics and offline ONNX speech synthesis with zero-disk frame persistence.

- [**ml-from-first-principles**](https://github.com/AhmedH32/ml-from-first-principles) — Implementing core ML/DL architectures and backward passes from scratch in pure NumPy, focusing on SIMD vectorization, memory layout, and matrix calculus.

---

### Connect

- **LinkedIn:** [linkedin.com/in/ahmed-hassan-a6130a254](https://www.linkedin.com/in/ahmed-hassan-a6130a254)
- **Codeforces:** [codeforces.com/profile/Ahmedhassan3233](https://codeforces.com/profile/Ahmedhassan3233)
- **Email:** [ahmed.h.moustafa2003@gmail.com](mailto:ahmed.h.moustafa2003@gmail.com)
