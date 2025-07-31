# PatchTimeXer: Methodology

## 4. Methodology

### 4.1 Problem Formulation

Given a multivariate time series dataset $\mathcal{D} = \{(\mathbf{X}_i, \mathbf{Y}_i)\}_{i=1}^N$, where $\mathbf{X}_i \in \mathbb{R}^{L \times D}$ represents the input sequence of length $L$ with $D$ variables, and $\mathbf{Y}_i \in \mathbb{R}^{H \times D}$ represents the target sequence of length $H$. The goal of long-term time series forecasting is to learn a mapping function $f: \mathbb{R}^{L \times D} \rightarrow \mathbb{R}^{H \times D}$ that can accurately predict future values.

For each sample, we also have access to exogenous features $\mathbf{M}_i \in \mathbb{R}^{(L+H) \times F}$, where $F$ represents the number of temporal features (e.g., hour of day, day of week, month).

### 4.2 PatchTimeXer Architecture

PatchTimeXer is a novel hybrid architecture that combines the computational efficiency of patch-based processing from PatchTST with the global modeling capabilities of TimeXer. The model consists of four main components: (1) Hybrid Patch Embedding, (2) Exogenous Feature Embedding, (3) Hybrid Encoder, and (4) Prediction Head.

#### 4.2.1 Hybrid Patch Embedding

**Patch Creation and Embedding**

Given an input sequence $\mathbf{X} \in \mathbb{R}^{L \times D}$, we first apply instance normalization for each variable independently:

$$\tilde{\mathbf{X}}[:, d] = \frac{\mathbf{X}[:, d] - \mu_d}{\sigma_d}$$

where $\mu_d$ and $\sigma_d$ are the mean and standard deviation of the $d$-th variable.

The normalized sequence is then permuted to $\tilde{\mathbf{X}} \in \mathbb{R}^{D \times L}$ and divided into non-overlapping patches using the unfold operation:

$$\mathbf{P} = \text{Unfold}(\tilde{\mathbf{X}}, \text{size}=P, \text{step}=S)$$

where $P$ is the patch length and $S$ is the stride. This results in $\mathbf{P} \in \mathbb{R}^{D \times N_p \times P}$, where $N_p = \lfloor \frac{L-P}{S} \rfloor + 1$ is the number of patches.

**Value Embedding**

Each patch is linearly projected to the model dimension:

$$\mathbf{E}_{\text{patch}} = \mathbf{P} \mathbf{W}_v$$

where $\mathbf{W}_v \in \mathbb{R}^{P \times d_{\text{model}}}$ is the learnable projection matrix.

**Global Token Integration**

Following TimeXer's approach, we introduce learnable global tokens $\mathbf{G} \in \mathbb{R}^{D \times 1 \times d_{\text{model}}}$ for each variable to capture global patterns:

$$\mathbf{E}_{\text{global}} = \text{Repeat}(\mathbf{G}, B)$$

where $B$ is the batch size.

**Positional Encoding**

We apply sinusoidal positional encoding to the patch embeddings:

$$\mathbf{PE}[pos, 2i] = \sin(pos / 10000^{2i/d_{\text{model}}})$$
$$\mathbf{PE}[pos, 2i+1] = \cos(pos / 10000^{2i/d_{\text{model}}})$$

**Final Embedding**

The final patch embedding combines patches and global tokens:

$$\mathbf{E}_{\text{hybrid}} = \text{Concat}(\mathbf{E}_{\text{patch}} + \mathbf{PE}, \mathbf{E}_{\text{global}})$$
$$\mathbf{E}_{\text{final}} = \text{LayerNorm}(\text{Dropout}(\mathbf{E}_{\text{hybrid}}))$$

#### 4.2.2 Exogenous Feature Embedding

The exogenous features $\mathbf{M} \in \mathbb{R}^{L \times F}$ are processed using an inverted embedding strategy:

$$\mathbf{E}_{\text{exo}} = \text{Linear}(\mathbf{M}^T) + \mathbf{PE}_{\text{temporal}}$$

where the temporal positional encoding captures time-specific patterns in the exogenous features.

#### 4.2.3 Hybrid Encoder Architecture

The hybrid encoder consists of $N_L$ identical layers, each containing both self-attention and cross-attention mechanisms.

**Self-Attention Mechanism**

For each encoder layer $l$, the self-attention operates on the patch embeddings:

$$\mathbf{Q}^{(l)} = \mathbf{E}^{(l-1)} \mathbf{W}_Q^{(l)}$$
$$\mathbf{K}^{(l)} = \mathbf{E}^{(l-1)} \mathbf{W}_K^{(l)}$$
$$\mathbf{V}^{(l)} = \mathbf{E}^{(l-1)} \mathbf{W}_V^{(l)}$$

$$\text{SelfAttn}^{(l)} = \text{Softmax}\left(\frac{\mathbf{Q}^{(l)} (\mathbf{K}^{(l)})^T}{\sqrt{d_k}}\right) \mathbf{V}^{(l)}$$

**Global-Exogenous Cross-Attention**

The global tokens (last position in each sequence) attend to exogenous features:

$$\mathbf{G}_{\text{query}}^{(l)} = \mathbf{E}^{(l-1)}[:, -1, :] \mathbf{W}_{G}^{(l)}$$
$$\mathbf{K}_{\text{exo}}^{(l)} = \mathbf{E}_{\text{exo}} \mathbf{W}_{K_{\text{exo}}}^{(l)}$$
$$\mathbf{V}_{\text{exo}}^{(l)} = \mathbf{E}_{\text{exo}} \mathbf{W}_{V_{\text{exo}}}^{(l)}$$

$$\text{CrossAttn}^{(l)} = \text{Softmax}\left(\frac{\mathbf{G}_{\text{query}}^{(l)} (\mathbf{K}_{\text{exo}}^{(l)})^T}{\sqrt{d_k}}\right) \mathbf{V}_{\text{exo}}^{(l)}$$

**Layer Computation**

Each encoder layer performs the following computations:

$$\mathbf{E}_1^{(l)} = \text{LayerNorm}(\mathbf{E}^{(l-1)} + \text{Dropout}(\text{SelfAttn}^{(l)}))$$

$$\mathbf{G}_{\text{updated}}^{(l)} = \text{LayerNorm}(\mathbf{G}_{\text{orig}}^{(l)} + \text{Dropout}(\text{CrossAttn}^{(l)}))$$

$$\mathbf{E}_2^{(l)} = \text{Concat}(\mathbf{E}_1^{(l)}[:, :-1, :], \mathbf{G}_{\text{updated}}^{(l)})$$

$$\mathbf{E}^{(l)} = \text{LayerNorm}(\mathbf{E}_2^{(l)} + \text{Dropout}(\text{FFN}(\mathbf{E}_2^{(l)})))$$

where $\text{FFN}$ is a position-wise feed-forward network:

$$\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$

#### 4.2.4 Prediction Head

The final encoder output is reshaped and passed through a prediction head:

$$\mathbf{H}_{\text{final}} = \text{Reshape}(\mathbf{E}^{(N_L)}, (B, D, d_{\text{model}} \times (N_p + 1)))$$

$$\hat{\mathbf{Y}} = \text{Dropout}(\text{Linear}(\text{Flatten}(\mathbf{H}_{\text{final}})))$$

where the linear layer projects from $d_{\text{model}} \times (N_p + 1)$ to the prediction length $H$.

**Denormalization**

The final predictions are denormalized using the statistics computed during normalization:

$$\mathbf{Y}_{\text{pred}} = \hat{\mathbf{Y}} \cdot \sigma + \mu$$

### 4.3 Key Innovations

#### 4.3.1 Computational Efficiency

PatchTimeXer achieves computational efficiency through several mechanisms:

1. **Patch-based Processing**: Reduces sequence length from $L$ to $N_p \approx L/P$, leading to attention complexity reduction from $O(L^2)$ to $O((L/P)^2)$.

2. **Channel Independence**: Each variable is processed independently during patch embedding, enabling parallel computation.

3. **Sparse Cross-Attention**: Only global tokens participate in cross-attention with exogenous features, maintaining efficiency.

#### 4.3.2 Multi-Scale Feature Integration

The architecture integrates information at multiple scales:

1. **Local Patterns**: Captured through patch-level self-attention within each patch.
2. **Global Dependencies**: Modeled via global tokens that aggregate information across patches.
3. **Exogenous Information**: Incorporated through cross-attention between global tokens and temporal features.

#### 4.3.3 Enhanced Normalization Strategy

The model employs a sophisticated normalization approach:

1. **Instance Normalization**: Applied at input to handle non-stationarity.
2. **Layer Normalization**: Used throughout the architecture for training stability.
3. **Residual Connections**: Implemented around attention and feed-forward layers.

### 4.4 Training Procedure

#### 4.4.1 Loss Function

For forecasting tasks, we use Mean Squared Error (MSE) loss:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{BHD} \sum_{i=1}^B \sum_{t=1}^H \sum_{d=1}^D (\mathbf{Y}_{i,t,d} - \hat{\mathbf{Y}}_{i,t,d})^2$$

For more robust training, we also experiment with Mean Absolute Error (MAE):

$$\mathcal{L}_{\text{MAE}} = \frac{1}{BHD} \sum_{i=1}^B \sum_{t=1}^H \sum_{d=1}^D |\mathbf{Y}_{i,t,d} - \hat{\mathbf{Y}}_{i,t,d}|$$

#### 4.4.2 Optimization Strategy

We employ the Adam optimizer with the following configuration:

- Learning rate: $\alpha = 1 \times 10^{-4}$
- Beta coefficients: $\beta_1 = 0.9$, $\beta_2 = 0.999$
- Weight decay: $\lambda = 1 \times 10^{-5}$
- Learning rate scheduling: Cosine annealing with warm restarts

#### 4.4.3 Regularization Techniques

1. **Dropout**: Applied after attention mechanisms and in feed-forward networks with rate $p = 0.1$.
2. **Layer Normalization**: Applied before attention and feed-forward computations.
3. **Early Stopping**: Training is stopped if validation loss doesn't improve for 3 consecutive epochs.

### 4.5 Experimental Setup

#### 4.5.1 Datasets

We evaluate PatchTimeXer on multiple benchmark datasets:

1. **Weather**: 21 meteorological indicators recorded every 10 minutes for one year.
2. **ETT (Electricity Transformer Temperature)**: Four datasets (ETTh1, ETTh2, ETTm1, ETTm2) with different temporal granularities.
3. **Exchange Rate**: Exchange rates of 8 foreign countries collected daily.
4. **Traffic**: Road occupancy rates measured by sensors in California.

#### 4.5.2 Evaluation Metrics

We use two primary metrics for evaluation:

1. **Mean Squared Error (MSE)**:
   $$\text{MSE} = \frac{1}{HD} \sum_{t=1}^H \sum_{d=1}^D (\mathbf{Y}_{t,d} - \hat{\mathbf{Y}}_{t,d})^2$$

2. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{HD} \sum_{t=1}^H \sum_{d=1}^D |\mathbf{Y}_{t,d} - \hat{\mathbf{Y}}_{t,d}|$$

#### 4.5.3 Baseline Models

We compare PatchTimeXer against state-of-the-art models:

1. **PatchTST**: The original patch-based transformer model
2. **TimeXer**: The exogenous-aware time series transformer
3. **TimesNet**: A task-general time series model
4. **DLinear**: A simple linear model for forecasting
5. **FEDformer**: Frequency enhanced decomposed transformer
6. **Autoformer**: Decomposition transformers with Auto-Correlation

#### 4.5.4 Hyperparameter Configuration

The following hyperparameters are used across experiments:

- Model dimension: $d_{\text{model}} = 256$
- Number of attention heads: $h = 8$
- Number of encoder layers: $N_L = 2$
- Feed-forward dimension: $d_{ff} = 512$
- Patch length: $P = 16$
- Stride: $S = 8$
- Dropout rate: $p = 0.1$
- Batch size: $B = 16$

#### 4.5.5 Implementation Details

The model is implemented in PyTorch and trained on NVIDIA GPUs. Key implementation considerations include:

1. **Memory Optimization**: Gradient checkpointing for large models
2. **Numerical Stability**: Mixed precision training using automatic mixed precision (AMP)
3. **Reproducibility**: Fixed random seeds and deterministic operations

### 4.6 Ablation Studies

To validate the effectiveness of individual components, we conduct comprehensive ablation studies:

#### 4.6.1 Component Analysis

1. **Patch Embedding**: Compare standard patch embedding vs. hybrid patch embedding
2. **Global Tokens**: Evaluate performance with and without global tokens
3. **Cross-Attention**: Assess the impact of global-exogenous cross-attention
4. **Normalization**: Compare different normalization strategies

#### 4.6.2 Hyperparameter Sensitivity

1. **Patch Length**: Vary $P \in \{8, 12, 16, 20, 24\}$
2. **Model Dimension**: Test $d_{\text{model}} \in \{128, 256, 512\}$
3. **Number of Layers**: Evaluate $N_L \in \{1, 2, 3, 4\}$

### 4.7 Computational Complexity Analysis

#### 4.7.1 Time Complexity

The computational complexity of PatchTimeXer is analyzed as follows:

1. **Patch Embedding**: $O(LD \cdot d_{\text{model}})$
2. **Self-Attention**: $O(N_p^2 \cdot D \cdot d_{\text{model}})$
3. **Cross-Attention**: $O(D \cdot L \cdot d_{\text{model}})$
4. **Feed-Forward**: $O(N_p \cdot D \cdot d_{\text{model}} \cdot d_{ff})$

Overall complexity: $O(N_p^2 \cdot D \cdot d_{\text{model}})$ where $N_p \ll L$.

#### 4.7.2 Space Complexity

Memory requirements include:

1. **Model Parameters**: $O(d_{\text{model}}^2 \cdot N_L)$
2. **Activations**: $O(B \cdot N_p \cdot D \cdot d_{\text{model}})$
3. **Attention Matrices**: $O(B \cdot h \cdot N_p^2 \cdot D)$

### 4.8 Theoretical Analysis

#### 4.8.1 Approximation Capability

PatchTimeXer can be viewed as a universal function approximator for time series forecasting. The combination of patch-based local processing and global token-based global processing enables the model to capture both short-term fluctuations and long-term dependencies.

#### 4.8.2 Inductive Biases

The architecture incorporates several beneficial inductive biases:

1. **Locality Bias**: Patch-based processing assumes that nearby time points are more related
2. **Temporal Bias**: Positional encoding captures temporal order information
3. **Global Bias**: Global tokens enable long-range dependency modeling

### 4.9 Statistical Significance Testing

To ensure the reliability of our results, we employ statistical significance testing:

1. **Multiple Runs**: Each experiment is conducted with 5 different random seeds
2. **Confidence Intervals**: Report 95% confidence intervals for all metrics
3. **Statistical Tests**: Use paired t-tests to compare model performance
4. **Effect Size**: Report Cohen's d to measure practical significance

This methodology provides a comprehensive framework for evaluating the effectiveness of PatchTimeXer in long-term time series forecasting tasks while ensuring reproducibility and statistical rigor. 