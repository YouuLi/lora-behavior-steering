# Method and mathematical conventions

## Effective updates

For every LoRA target module, the package reconstructs the scaled effective update

\[
\Delta W_m = (\alpha_{\mathrm{LoRA}}/r)B_mA_m.
\]

All module updates are treated as one structured vector. Inner products are global
Frobenius inner products summed over modules. No module-wise normalization is applied.

## Recommended shared direction

For paired low/high endpoints trained with seed `i`, define

\[
d_i = \Delta W_{i,\mathrm{high}}-\Delta W_{i,\mathrm{low}}.
\]

The default direction is the normalized mean `d_i`. This directly estimates the common
paired contrast and gives every seed equal coefficient weight.

## Why centered PCA on differences is inappropriate here

Standard PCA centers the observations. Applying it to the `d_i` produces

\[
\tilde d_i=d_i-\bar d.
\]

The shared mean contrast `bar d` is removed, so the principal components describe
seed-to-seed variation. When a PCA-style estimator is desired, this repository offers
uncentered SVD of the differences instead.

## Original pooled-endpoint PCA

The original experiment pooled all low and high endpoints, centered the pooled set,
and performed PCA through a sample Gram matrix. It then selected a component using the
known endpoint labels and pairings. Consequently, this estimator is described as
pooled-endpoint PCA with label-assisted component selection.

For each component, three properties are recorded:

1. explained-variance ratio;
2. standardized low/high group separation;
3. the fraction of paired high-minus-low projections that are positive.

The component must explain at least 1% of variance and have fully consistent paired
ordering across seeds. Among eligible components, the one with the largest absolute
standardized separation is selected. Its sign is oriented from low to high. If the
consistency filter leaves no candidates, selection falls back to all components above
the variance threshold.

## Steering

The two directions are combined additively:

\[
W' = W + c_d v_d + c_r v_r.
\]

The implementation adds updates in place inside a context manager and restores the
base parameters after each coefficient setting. This allows an entire coefficient grid
to be evaluated without reloading the model.

## Scope

Linear composition is an empirical approximation. Coefficient values are not expected
to have identical behavioral meaning across model families, extraction methods, or
normalization conventions. Report the direction norm convention, coefficient range,
decoding settings, and observed operating region with every experiment.

