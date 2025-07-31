# PatchTimeXer: Literature Review

## 2. Literature Review

### 2.1 Time Series Forecasting: An Overview

Time series forecasting has been a fundamental problem in machine learning and statistics for decades. Traditional approaches can be broadly categorized into statistical methods and machine learning approaches. Classical statistical methods such as ARIMA (Autoregressive Integrated Moving Average) [Box & Jenkins, 1970], Exponential Smoothing [Holt, 1957], and state space models [Kalman, 1960] have dominated the field for many years. However, these methods often struggle with complex patterns, non-linear relationships, and high-dimensional multivariate time series.

The advent of deep learning has revolutionized time series forecasting. Early neural network approaches including Multi-Layer Perceptrons (MLPs) and Recurrent Neural Networks (RNNs) [Rumelhart et al., 1986] showed promise but suffered from limitations such as vanishing gradients and sequential processing bottlenecks.

### 2.2 Recurrent Neural Networks and Long Short-Term Memory

The introduction of Long Short-Term Memory (LSTM) networks [Hochreiter & Schmidhuber, 1997] marked a significant advancement in sequence modeling. LSTMs addressed the vanishing gradient problem through gating mechanisms that allow selective information retention and forgetting. Variants such as Gated Recurrent Units (GRUs) [Cho et al., 2014] further simplified the architecture while maintaining performance.

Several studies have applied LSTM-based architectures to time series forecasting:

- **LSTNet** [Lai et al., 2018] combined convolutional and recurrent layers with attention mechanisms for multivariate time series forecasting.
- **DA-RNN** [Qin et al., 2017] introduced dual-stage attention mechanisms to handle variable selection and temporal attention in encoder-decoder frameworks.
- **DeepAR** [Salinas et al., 2020] utilized autoregressive RNNs with probabilistic forecasting capabilities for large-scale time series prediction.

Despite these advances, RNN-based models suffer from inherent sequential processing limitations, making them computationally inefficient for long sequences and difficult to parallelize effectively.

### 2.3 Convolutional Neural Networks for Time Series

Convolutional Neural Networks (CNNs) emerged as an alternative approach for time series modeling, offering parallelization advantages over RNNs. Key developments include:

- **WaveNet** [van den Oord et al., 2016] introduced dilated convolutions for modeling long-range dependencies in sequential data.
- **TCN (Temporal Convolutional Networks)** [Bai et al., 2018] demonstrated that well-designed CNNs could outperform RNNs on many sequence modeling tasks.
- **DSANet** [Huang et al., 2019] combined CNN with self-attention mechanisms for multivariate time series forecasting.

While CNNs offer computational advantages, they typically require deep architectures to capture long-range dependencies, leading to increased model complexity and potential optimization challenges.

### 2.4 The Transformer Revolution

The introduction of the Transformer architecture [Vaswani et al., 2017] fundamentally changed sequence modeling across multiple domains. The key innovation was the self-attention mechanism, which allows direct modeling of dependencies between any two positions in a sequence, regardless of their distance.

#### 2.4.1 Core Transformer Components

The original Transformer consists of:

1. **Multi-Head Self-Attention**: Enables the model to attend to different representation subspaces simultaneously.
2. **Position Encoding**: Injects positional information since attention mechanisms are permutation-invariant.
3. **Feed-Forward Networks**: Apply point-wise transformations to attention outputs.
4. **Residual Connections and Layer Normalization**: Facilitate training stability and gradient flow.

#### 2.4.2 Transformers in Time Series Forecasting

Several works have adapted Transformers for time series forecasting:

- **Transformer** [Vaswani et al., 2017] was initially applied to time series in an encoder-decoder framework.
- **LogTrans** [Li et al., 2019] introduced convolutional self-attention to reduce computational complexity.
- **Reformer** [Kitaev et al., 2020] employed locality-sensitive hashing and reversible layers to improve memory efficiency.

However, direct application of Transformers to long time series faces significant challenges due to quadratic complexity in sequence length.

### 2.5 Efficient Attention Mechanisms

To address the computational limitations of standard attention, numerous efficient attention mechanisms have been proposed:

#### 2.5.1 Sparse Attention Patterns

- **Sparse Transformer** [Child et al., 2019] introduced factorized attention patterns to reduce complexity.
- **Longformer** [Beltagy et al., 2020] combined local windowed attention with global attention for selected positions.
- **BigBird** [Zaheer et al., 2020] used a combination of global, local, and random attention patterns.

#### 2.5.2 Linear Attention

- **Linformer** [Wang et al., 2020] projected keys and values to lower dimensions.
- **Performer** [Choromanski et al., 2021] used random feature maps to approximate attention.
- **FNet** [Lee-Thorp et al., 2021] replaced attention with Fourier transforms.

#### 2.5.3 Hierarchical Approaches

- **Pyraformer** [Liu et al., 2021] introduced pyramidal attention to model multi-scale temporal patterns.
- **Informer** [Zhou et al., 2021] proposed ProbSparse attention for long sequence forecasting.

### 2.6 Patch-Based Methods in Time Series

The concept of patch-based processing, inspired by computer vision, has gained attention in time series analysis:

#### 2.6.1 Motivation and Benefits

Patch-based methods offer several advantages:

1. **Computational Efficiency**: Reduces sequence length and attention complexity.
2. **Local Pattern Capture**: Patches naturally capture local temporal patterns.
3. **Parallelization**: Independent patch processing enables efficient parallelization.

#### 2.6.2 PatchTST: The Pioneer

**PatchTST** [Nie et al., 2023] was the first to systematically apply patch-based processing to time series forecasting:

- **Patching Strategy**: Divides time series into non-overlapping patches.
- **Channel Independence**: Processes each variable separately to avoid potential negative transfer.
- **Efficient Architecture**: Uses standard Transformer encoders on patch sequences.

PatchTST demonstrated state-of-the-art performance on multiple benchmarks while maintaining computational efficiency. However, it has limitations:

1. **Local Focus**: Primarily captures local patterns within patches.
2. **Limited Global Modeling**: Lacks explicit mechanisms for global pattern capture.
3. **No Exogenous Integration**: Cannot effectively incorporate external features.

### 2.7 Exogenous Feature Integration

Incorporating exogenous features (external variables) has been recognized as crucial for improving forecasting accuracy:

#### 2.7.1 Traditional Approaches

- **Vector Autoregression (VAR)** [Sims, 1980] models multivariate relationships.
- **ARIMAX** extends ARIMA to include exogenous variables.
- **State Space Models** [Durbin & Koopman, 2012] naturally incorporate external inputs.

#### 2.7.2 Deep Learning Approaches

- **DA-RNN** [Qin et al., 2017] uses attention to select relevant exogenous features.
- **DSANet** [Huang et al., 2019] combines global and local attention for multivariate modeling.
- **TFT (Temporal Fusion Transformer)** [Lim et al., 2021] introduces variable selection networks for static, observed, and known future inputs.

#### 2.7.3 TimeXer: Advanced Exogenous Integration

**TimeXer** [Wang et al., 2024] represents a significant advancement in exogenous feature integration:

- **Dual Embedding**: Separate processing of endogenous and exogenous features.
- **Global Tokens**: Learnable tokens that aggregate global information.
- **Cross-Attention**: Enables interaction between endogenous patterns and exogenous features.

TimeXer demonstrated superior performance on datasets with rich exogenous information but requires careful handling of computational complexity for long sequences.

### 2.8 Normalization and Stationarity

Handling non-stationary time series has been a persistent challenge:

#### 2.8.1 Traditional Stationarity Methods

- **Differencing**: Removes trends through first or seasonal differencing.
- **Detrending**: Explicitly removes trend components.
- **Logarithmic Transformation**: Stabilizes variance in multiplicative models.

#### 2.8.2 Deep Learning Normalization

- **RevIN (Reversible Instance Normalization)** [Kim et al., 2022] normalizes inputs and denormalizes outputs.
- **Non-stationary Transformer** [Liu et al., 2022] learns adaptive normalization parameters.
- **DishTS** [Wang et al., 2023] proposes dish-ts normalization for better stationarity handling.

### 2.9 Current State-of-the-Art Models

Recent state-of-the-art models in long-term time series forecasting include:

#### 2.9.1 Transformer-Based Models

- **FEDformer** [Zhou et al., 2022] employs frequency domain processing with Fourier transforms.
- **ETSformer** [Woo et al., 2022] incorporates exponential smoothing into Transformer architecture.
- **Crossformer** [Zhang & Yan, 2023] uses dimension-segment-wise attention for efficiency.

#### 2.9.2 CNN-Based Models

- **TimesNet** [Wu et al., 2023] transforms 1D time series to 2D tensors for CNN processing.
- **ModernTCN** [Donà et al., 2023] revisits temporal convolutional networks with modern techniques.

#### 2.9.3 Linear Models

Surprisingly, simple linear models have shown competitive performance:

- **DLinear** [Zeng et al., 2023] decomposes time series and applies linear projections.
- **NLinear** [Zeng et al., 2023] normalizes inputs before linear transformation.

#### 2.9.4 Hybrid Approaches

- **iTransformer** [Liu et al., 2024] inverts dimensions to treat variables as tokens.
- **PatchMixer** [Chen et al., 2023] combines patching with MLP-Mixer architecture.

### 2.10 Research Gaps and Motivation

Despite significant progress, several limitations persist in current approaches:

#### 2.10.1 PatchTST Limitations

1. **Local Pattern Bias**: Primarily focuses on local patterns within patches.
2. **Global Context Loss**: Limited ability to capture long-range dependencies across patches.
3. **Exogenous Integration**: Cannot effectively incorporate external features.
4. **Feature Interaction**: Channel independence prevents cross-variable modeling.

#### 2.10.2 TimeXer Limitations

1. **Computational Complexity**: High complexity for long sequences.
2. **Patch Efficiency**: Doesn't leverage patch-based efficiency gains.
3. **Scalability**: Limited scalability to very long sequences.

#### 2.10.3 General Challenges

1. **Efficiency vs. Effectiveness Trade-off**: Most efficient models sacrifice modeling capability.
2. **Multi-scale Modeling**: Difficulty in capturing patterns at different temporal scales.
3. **Robust Normalization**: Handling non-stationary data remains challenging.
4. **Exogenous Feature Selection**: Determining relevant external features is often manual.

### 2.11 Theoretical Foundations

#### 2.11.1 Universal Approximation

Neural networks, including Transformers, are universal function approximators [Hornik et al., 1989]. For time series forecasting, this means that sufficiently large networks can theoretically approximate any measurable forecasting function. However, practical considerations such as sample complexity and computational efficiency are crucial.

#### 2.11.2 Attention as Kernel Methods

Recent work [Tsai et al., 2019] has shown connections between attention mechanisms and kernel methods. This perspective provides theoretical insights into why attention-based models are effective for capturing complex patterns in sequential data.

#### 2.11.3 Inductive Biases

The choice of architecture introduces specific inductive biases:

- **RNNs**: Sequential processing bias, suitable for temporal dependencies.
- **CNNs**: Locality bias, effective for local pattern detection.
- **Transformers**: Permutation invariance, requiring explicit positional encoding.

### 2.12 Evaluation Methodologies

#### 2.12.1 Standard Metrics

Common evaluation metrics include:

- **Mean Squared Error (MSE)**: Penalizes large errors quadratically.
- **Mean Absolute Error (MAE)**: Provides robust error measurement.
- **Mean Absolute Percentage Error (MAPE)**: Scale-independent relative error.

#### 2.12.2 Advanced Evaluation

Recent works have introduced more sophisticated evaluation:

- **Directional Accuracy**: Measures prediction direction correctness.
- **Distribution-based Metrics**: Evaluate prediction uncertainty.
- **DTW (Dynamic Time Warping)**: Measures shape similarity between sequences.

### 2.13 Benchmark Datasets

Standard benchmark datasets in time series forecasting include:

1. **ETT (Electricity Transformer Temperature)**: Four datasets with different granularities.
2. **Weather**: Meteorological data with multiple variables.
3. **Traffic**: Road occupancy rate data from California sensors.
4. **Exchange Rate**: Daily exchange rates of foreign currencies.
5. **ILI (Influenza-Like Illness)**: Weekly influenza patient ratios.

### 2.14 Summary and Research Direction

The literature review reveals that while significant progress has been made in time series forecasting, there remains a clear gap between computational efficiency and modeling capability. PatchTST achieves efficiency through patch-based processing but lacks global modeling capabilities. TimeXer provides excellent exogenous feature integration but sacrifices computational efficiency.

This motivates the development of PatchTimeXer, which aims to combine the computational efficiency of patch-based processing with the global modeling capabilities and exogenous feature integration of advanced attention mechanisms. The hybrid approach seeks to address the limitations of both parent models while maintaining their respective strengths.

The next section presents our proposed methodology for achieving this hybrid architecture, detailing the theoretical foundations and practical implementation of PatchTimeXer.

### References

[This section would contain full academic references for all cited works] 