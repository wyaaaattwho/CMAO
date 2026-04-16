# CMAO: Correct-Mode Advantage Optimization

We propose to move beyond answer-level rewards and introduce **fine-grained optimization within correct responses**.

Given a prompt $q$, sample a group:

$$
Y = \{y_1, \dots, y_G\}, \quad y_i = (c_i, a_i),
$$

where $c_i$ is the reasoning (CoT) and $a_i$ is the final answer.

We define the following concepts:

1. Answer Correctness
    
    $$
    r_i^{ans}=\begin{cases}1 \quad\text{if } a_i \text{ is correct.}\\0\quad\text{othewise.}\end{cases}
    $$
    
2. Reasoning Quality (Verifier / Preference Score)
    
    $$
    
    r_i^{\text{qual}} \in [0,1].

    In training, we use this score only to induce **pairwise preferences within correct samples**, rather than regressing directly on the raw scalar.
    $$
    
3. Reasoning Mode
    
    $$
    m_i = \phi(c_i),
    $$
    
    where $\phi$ extracts the reasoning pattern.
    

We define CMAO advantage thereafter:

$$

\hat A_i^{\text{CMAO}}
=
\lambda_{\text{ans}} \hat A_i^{\text{ans}}
+
\lambda_{\text{qual}} \hat A_i^{\text{qual}}
+
\lambda_{\text{mode}} \hat A_i^{\text{mode}}.

$$

The A**nswer Advantage** is defined in accordance with GRPO as:

$$

\hat A_i^{\text{ans}}
=
\frac{r_i^{\text{ans}} - \mu^{\text{ans}}}{\sigma^{\text{ans}} + \varepsilon}.
$$

This term defines how good is your answer itself within in a group, thus increasing the probability of these answers.

We then define the **Correct-Only Pairwise Quality Advantage** as:

$$

\text{Let }\mathcal I^+ = \{i : r_i^{\text{ans}} = 1\}
$$

For any two correct samples $i,j \in \mathcal I^+$, define a margin-based preference:

$$

\operatorname{pref}(i,j)
=
\begin{cases}
1 & \text{if } r_i^{\text{qual}} - r_j^{\text{qual}} > \delta,\\
-1 & \text{if } r_j^{\text{qual}} - r_i^{\text{qual}} > \delta,\\
0 & \text{otherwise.}
\end{cases}
$$

Then the training-time quality advantage is:

$$

\hat A_i^{\text{qual}}
=
\mathbf{1}[i \in \mathcal I^+]
\cdot
\operatorname{clip}
\left(
\frac{1}{\max(1,|\mathcal I^+|-1)}
\sum_{j \in \mathcal I^+, j \neq i}\operatorname{pref}(i,j),
-1,1
\right).
$$

With this term, we can guarantee that even when all answers are correct:

$$
\hat A^{\text{ans}} = 0, \quad \hat A^{\text{qual}} \neq 0,
$$

thus we can make sure the learning continues beyond answer saturation, while also reducing the impact of noisy scalar quality estimates.

Finally, we define **Mode-Balance Advantage.** We first consider mode frequency within correct samples:

$$

\hat p_Y(m)
=
\frac{
\sum_j \mathbf{1}[m_j = m]\mathbf{1}[r_j^{\text{ans}}=1]
}{
\sum_j \mathbf{1}[r_j^{\text{ans}}=1]
},
$$

Let 

$$

b_i =
\mathbf{1}[r_i^{\text{ans}}=1]
\cdot r_i^{\text{qual}}
\cdot \big(-\log(\hat p_Y(m_i))\big),
$$

the advantage would be:

$$

\hat A_i^{\text{mode}}
=
\frac{b_i - \mu^{\text{mode}}}{\sigma^{\text{mode}} + \varepsilon}
$$

The purpose of this advantage term is to encourage **rare but high-quality modes** and also prevent **collapse to a single reasoning strategy.**

The final objective of our CMAO would be:

$$

\mathcal L_{\text{CMAO}}(\theta)
=
\mathbb E\left[
\min\Big(
\rho_i(\theta)\hat A_i^{\text{CMAO}},
\operatorname{clip}(\rho_i(\theta),1-\epsilon,1+\epsilon)\hat A_i^{\text{CMAO}}
\Big)
\right]
-\beta \mathrm{KL}(\pi_\theta\|\pi_{\text{ref}}),
$$

where:

$$

\rho_i(\theta)=\frac{\pi_\theta(y_i\mid q)}{\pi_{\theta_{\text{ref}}}(y_i\mid q)}.
$$

CMAO  wnats to address a fundamental limitation of current LLM RL: answer-level rewards saturate too early, leading to both learning stagnation and mode collapse. By introducing correctness-conditioned quality and mode-aware advantages, CMAO enables continued reasoning improvement while preserving high-quality diversity.
