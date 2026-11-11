# QuALITY Benchmark Evaluation & RAPTOR Analysis

**Project**: CS561 — Hierarchical RAG with RAPTOR: Tree-Organized Retrieval for Long-Document Question Answering  
**Model Stack**: Google Gemini (`models/gemini-embedding-001` + `models/gemini-3.5-flash-lite`)  
**Dataset**: [QuALITY (NYU-MLL)](https://github.com/nyu-mll/quality.git) *(Pang et al., NAACL 2022)*

---

## 1. Dataset Details

| Attribute | Specification |
| :--- | :--- |
| **Dataset Name** | **QuALITY** (*Question Answering with Long Input Texts, Yes!*) |
| **Repository** | `https://github.com/nyu-mll/quality.git` |
| **Total Articles** | **762 long-form articles** (~4,150–5,000 words per article; ~5,000–6,000 tokens) |
| **Total Questions** | **6,737 multiple-choice questions** (4 options each) |
| **Task Formulation** | 4-way Multiple-Choice Question Answering over narrative context |

### Train / Validation / Test Splits

| Split | Article Count | Question Count | Average Words per Article |
| :--- | :---: | :---: | :---: |
| **Train** | 300 | 2,523 | 4,165.9 |
| **Validation (Dev)** | 230 | 2,086 | 4,121.8 |
| **Test** | 232 | 2,128 | 4,184.5 |
| **Total** | **762** | **6,737** | **~4,150** |

### Feature Fields

* `article_id`, `set_unique_id`, `title`, `author`, `year`, `topic`: Metadata and genre information (science fiction, essays, short stories).
* `article`: The full raw text of the document without arbitrary context truncation.
* `questions`: List of question objects with the following annotations:
  * `question`: Multi-hop reasoning question string.
  * `options`: 4 multiple-choice candidates (`[opt1, opt2, opt3, opt4]`).
  * `gold_label`: Consensus ground-truth index (1-indexed: 1, 2, 3, or 4).
  * `writer_label`: Option chosen by the author who wrote the question.
  * `difficult`: Binary indicator (`1` if speed skimmers failed, proving non-trivial reasoning).
  * `validation`: Verification results across 3 to 5 independent human readers.
  * `speed_validation`: Ratings from workers who were only allowed 45 seconds to skim.

### Annotation Methodology

1. **Question Writing**: Human annotators read complete long narratives (untimed) and formulated questions designed explicitly to resist Ctrl+F keyword matching and superficial sentence lookup.
2. **Untimed Consensus Validation**: 3 to 5 independent validators read the full story and selected options. A `gold_label` was established only when annotators reached strong consensus ($\ge 3/5$).
3. **Speed Validation Filtering**: A separate group of annotators was allowed only 45 seconds to skim the text. Questions that untimed readers solved correctly but speed-skimmers failed were flagged with `difficult = 1`, making QuALITY the gold standard for testing whether hierarchical RAG avoids "lost in the middle" failures compared to flat RAG.

---

## 2. Concrete Example: Step-by-Step RAPTOR Test

### Step 1: Input

* **Source File**: `quality_repo/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev`
* **Article Title**: *"The Girl in His Mind"* by Robert F. Young
* **Document Length**: 28,515 characters / 4,888 words (~6,000 tokens)
* **Sample Passage Excerpt**:
  > *"Deirdre was seventeen then, and the graduating class of the local high school was having its senior prom... Blake suggested she go with the young man, but she flared up with fierce indignation: 'Did he scrimp and go without in order to put me through high school and college? Did he give up his right arm for me?'..."*
* **Target Question**:
  > *"Why does Deirdre get so upset when Blake Past suggests she go to prom with the young man?"*
* **Candidates**:
  * `[1]` Because Blake is trying to guilt Deirdre into going with the young man by telling her that it'll ease her conscience.
  * `[2]` Because Deirdre has fallen in love with Blake, despite his age, and wants him to take her to the prom.
  * `[3]` Because Blake is acting like he's her father, which is a sensitive topic for Deirdre because she lost her real parents.
  * `[4]` Because the young man gave up his right arm in order to afford tickets to the prom, and this disgusts Deirdre.

---

### Step 2: Intermediate Features

```
                             [Layer 1: 5 Root / Summary Nodes]
                                   ▲                ▲
                        (GMM Clustering)       (Summarization)
                                   │                │
            [Layer 0: 27 Leaf Chunks (250 tokens each, 3072-d vectors)]
                                   ▲
                           (Text Splitting)
                                   │
              [Input Document: 4,888 words (approx. 6,000 tokens)]
```

#### A. Chunking & Tokenization (Layer 0)
* **Chunking Configuration**: Tokenizer `cl100k_base` with a maximum window size of 250 tokens.
* **Output**: **27 leaf chunks** representing the complete narrative at fine granularity.

#### B. Dense Embeddings
* **Embedding Model**: `models/gemini-embedding-001`
* **Vector Dimension**: $\mathbf{e}_i \in \mathbb{R}^{3072}$
* **Vector Preview (Leaf Node 0)**:
  ```json
  [-0.0126, -0.0035, 0.0108, -0.0591, 0.0301, ...]
  ```

#### C. Dimensionality Reduction (UMAP)
* High-dimensional embeddings (3,072 dimensions) are reduced to preserve manifold topology:
  $$\text{dim}_{\text{reduced}} = 10$$

#### D. Soft Clustering (Gaussian Mixture Models + BIC)
* **Cluster Determination**: Bayesian Information Criterion (BIC) minimizes model complexity penalty to automatically select the optimal cluster count:
  $$K_{\text{optimal}} = 5 \text{ clusters}$$
* **Soft Assignment**: Chunks belong to clusters based on Gaussian posterior probabilities ($P(\text{cluster} \mid \mathbf{e}) > 0.5$), allowing thematic overlap.

#### E. Hierarchical Summarization (Layer 1)
* Clustered chunks are synthesized into abstract parent nodes using `gemini-3.5-flash-lite`.
* **Example Summary Node 27 (linking Children {4, 14, 15, 16})**:
  > *"Nathan Blake, a 38-year-old man, reflects on his complex, guilt-driven relationship with Deirdre, a young woman he purchased and supported through school, acknowledging his suppressed romantic attachment..."*
* **Compiled Tree Structure**: **32 Total Nodes** (27 granular leaves in Layer 0 + 5 thematic root summaries in Layer 1).

#### F. Collapsed Tree Retrieval
* Query vector is compared across **all 32 nodes** simultaneously via cosine similarity.
* **Retrieved Context**: Top-5 nodes containing **4,526 characters** combining thematic summaries of Blake and Deirdre's dynamic alongside exact dialogue lines from the prom argument.

---

### Step 3: Output

* **RAPTOR Generated Answer**:
  > **Predicted Choice**: `[2]`  
  > **Model Justification**: *Deirdre gets upset because she has fallen in love with Nathan (Blake Past) and wants him to take her to the prom. When Nathan proposes that she attend with a younger suitor, she passionately defends Nathan's exclusive sacrifices ("Did he scrimp and go without in order to put me through high school and college?") and resents his attempts to deny their mutual connection by playing the role of an older guardian or father figure.*
* **Gold Ground-Truth Label**:
  > **Option 2** (*"Because Deirdre has fallen in love with Blake, despite his age, and wants him to take her to the prom."*)
* **Result**: **100% Match (Correct Prediction)**.

---

## 3. Second Validation Sample

| Attribute | Details |
| :--- | :--- |
| **Question** | *"Sabrina York is:"* |
| **Options** | `[1]` a criminal that Blake is hunting<br>`[2]` a psycheye that taught Blake all the tricks<br>`[3]` an old friend of Blake's<br>`[4]` Eldoria's alter ego |
| **Retrieved Context** | 4,975 characters from Layer 0 (narrative scenes) and Layer 1 (detective assignment summary) |
| **RAPTOR Prediction** | **Option 1**: *"The text and summary explicitly state that Sabrina York is a criminal whom private psycheye Nathan Blake is pursuing through his memory-country."* |
| **Target Gold Label** | **Option 1** (*"a criminal that Blake is hunting"*) |
| **Result** | **100% Match (Correct Prediction)** |
