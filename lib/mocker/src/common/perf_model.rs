// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Performance model for timing simulations in the mocker.
//!
//! This module provides two timing models:
//! 1. Polynomial: Hardcoded polynomial formulas (default, backward compatible)
//! 2. Interpolated: Grid-based interpolation from profiler data (loaded from NPZ files)

use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2, Array3};
use ndarray_interp::InterpolateError;
use ndarray_interp::interp1d::{Interp1DBuilder, Linear};
use ndarray_interp::interp2d::{Bilinear, Interp2DBuilder};
use std::path::Path;
use std::sync::Arc;

/// Trait to abstract over legacy and batch-aware prefill timing surfaces.
pub trait PrefillInterpolator: Send + Sync {
    fn interp(&self, batch_size: f64, new_tokens_per_request: f64, prefix: f64) -> Result<f64>;
}

/// Trait to abstract over 2D interpolation for decode timing
pub trait DecodeInterpolator: Send + Sync {
    fn interp(&self, x: f64, y: f64) -> Result<f64, InterpolateError>;
}

/// Callback trait for direct AIC SDK calls.
/// Implementors call the Python AIC SDK via PyO3 GIL.
pub trait AicCallback: Send + Sync {
    /// Predict prefill latency in ms.
    /// Parameters: (batch_size, effective_isl, prefix)
    fn predict_prefill(&self, batch_size: usize, effective_isl: usize, prefix: usize) -> f64;

    /// Predict decode (generation) latency in ms.
    /// Parameters: (batch_size, isl, osl)
    fn predict_decode(&self, batch_size: usize, isl: usize, osl: usize) -> f64;
}

/// Wrapper to implement PrefillInterpolator for the concrete Interp1D type
struct PrefillInterp1D {
    inner: ndarray_interp::interp1d::Interp1D<
        ndarray::OwnedRepr<f64>,
        ndarray::OwnedRepr<f64>,
        ndarray::Ix1,
        Linear,
    >,
}

impl PrefillInterpolator for PrefillInterp1D {
    fn interp(&self, batch_size: f64, new_tokens_per_request: f64, _prefix: f64) -> Result<f64> {
        self.inner
            .interp_scalar(batch_size * new_tokens_per_request)
            .map_err(Into::into)
    }
}

/// Three-dimensional prefill timing surface over scheduler-local batch shape.
///
/// Axes are batch size, fresh tokens per request, and KV-read/prefix tokens per
/// request. Values are milliseconds. Linear extrapolation is intentional and
/// matches the legacy 1D interpolator's behavior outside the measured domain.
struct PrefillInterp3D {
    batch_size: Array1<f64>,
    new_tokens_per_request: Array1<f64>,
    kv_read_tokens_per_request: Array1<f64>,
    values_ms: Array3<f64>,
}

impl PrefillInterpolator for PrefillInterp3D {
    fn interp(&self, batch_size: f64, new_tokens_per_request: f64, prefix: f64) -> Result<f64> {
        let (b0, b1, bt) = interpolation_bounds(&self.batch_size, batch_size)?;
        let (n0, n1, nt) =
            interpolation_bounds(&self.new_tokens_per_request, new_tokens_per_request)?;
        let (k0, k1, kt) = interpolation_bounds(&self.kv_read_tokens_per_request, prefix)?;

        let c000 = self.values_ms[[b0, n0, k0]];
        let c001 = self.values_ms[[b0, n0, k1]];
        let c010 = self.values_ms[[b0, n1, k0]];
        let c011 = self.values_ms[[b0, n1, k1]];
        let c100 = self.values_ms[[b1, n0, k0]];
        let c101 = self.values_ms[[b1, n0, k1]];
        let c110 = self.values_ms[[b1, n1, k0]];
        let c111 = self.values_ms[[b1, n1, k1]];

        let c00 = lerp(c000, c100, bt);
        let c01 = lerp(c001, c101, bt);
        let c10 = lerp(c010, c110, bt);
        let c11 = lerp(c011, c111, bt);
        let c0 = lerp(c00, c10, nt);
        let c1 = lerp(c01, c11, nt);
        Ok(lerp(c0, c1, kt))
    }
}

fn lerp(left: f64, right: f64, position: f64) -> f64 {
    left + (right - left) * position
}

fn interpolation_bounds(axis: &Array1<f64>, value: f64) -> Result<(usize, usize, f64)> {
    if !value.is_finite() {
        bail!("interpolation coordinate must be finite, got {value}");
    }
    validate_interpolation_axis(axis)?;
    let coordinates = axis
        .as_slice()
        .context("interpolation axis must be contiguous")?;

    let upper = coordinates.partition_point(|coordinate| *coordinate <= value);
    let (lower_index, upper_index) = if upper == 0 {
        (0, 1)
    } else if upper >= axis.len() {
        (axis.len() - 2, axis.len() - 1)
    } else {
        (upper - 1, upper)
    };
    let lower = axis[lower_index];
    let position = (value - lower) / (axis[upper_index] - lower);
    Ok((lower_index, upper_index, position))
}

fn validate_interpolation_axis(axis: &Array1<f64>) -> Result<()> {
    if axis.len() < 2 {
        bail!("interpolation axis must contain at least two points");
    }
    let coordinates = axis
        .as_slice()
        .context("interpolation axis must be contiguous")?;
    if coordinates.iter().any(|coordinate| !coordinate.is_finite())
        || coordinates.windows(2).any(|window| window[1] <= window[0])
    {
        bail!("interpolation axis must be finite and strictly increasing");
    }
    Ok(())
}

/// Wrapper to implement DecodeInterpolator for the concrete Interp2D type
struct DecodeInterp2D {
    inner: ndarray_interp::interp2d::Interp2D<
        ndarray::OwnedRepr<f64>,
        ndarray::OwnedRepr<f64>,
        ndarray::OwnedRepr<f64>,
        ndarray::Ix2,
        Bilinear,
    >,
}

impl DecodeInterpolator for DecodeInterp2D {
    fn interp(&self, x: f64, y: f64) -> Result<f64, InterpolateError> {
        self.inner.interp_scalar(x, y)
    }
}

/// Performance model for predicting prefill and decode timing
#[derive(Default)]
pub enum PerfModel {
    /// Default polynomial-based model using hardcoded formulas
    #[default]
    Polynomial,
    /// Interpolation-based model using profiler data
    /// Decode axes: (active_kv_tokens, context_length)
    Interpolated {
        prefill_interp: Arc<dyn PrefillInterpolator>,
        decode_interp: Arc<dyn DecodeInterpolator>,
    },
    /// AI Configurator SDK calls via Python callback.
    /// Passes the reduced prefill inputs (batch_size, effective_isl, prefix).
    Aiconfigurator { callback: Arc<dyn AicCallback> },
}

impl Clone for PerfModel {
    fn clone(&self) -> Self {
        match self {
            PerfModel::Polynomial => PerfModel::Polynomial,
            PerfModel::Interpolated {
                prefill_interp,
                decode_interp,
            } => PerfModel::Interpolated {
                prefill_interp: Arc::clone(prefill_interp),
                decode_interp: Arc::clone(decode_interp),
            },
            PerfModel::Aiconfigurator { callback } => PerfModel::Aiconfigurator {
                callback: Arc::clone(callback),
            },
        }
    }
}

impl std::fmt::Debug for PerfModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PerfModel::Polynomial => write!(f, "PerfModel::Polynomial"),
            PerfModel::Interpolated { .. } => write!(f, "PerfModel::Interpolated {{ .. }}"),
            PerfModel::Aiconfigurator { .. } => write!(f, "PerfModel::Aiconfigurator"),
        }
    }
}

impl PerfModel {
    /// Load performance model from NPZ file
    ///
    /// Expected prefill arrays use one of two backward-compatible layouts:
    /// - Legacy 1D: prefill_isl, prefill_ttft_ms
    /// - Batch-aware 3D: prefill_batch_size,
    ///   prefill_new_tokens_per_request, prefill_kv_read_tokens_per_request,
    ///   prefill_time_ms
    ///
    /// Both layouts also require:
    /// - decode_active_kv_tokens: 1D array of active KV token counts
    /// - decode_context_length: 1D array of context lengths
    /// - decode_itl: 2D array of inter-token latencies in milliseconds
    pub fn from_npz(path: &Path) -> Result<Self> {
        use ndarray_npy::NpzReader;
        use std::fs::File;

        tracing::info!("Loading performance model from NPZ file: {:?}", path);

        let file =
            File::open(path).with_context(|| format!("Failed to open NPZ file: {:?}", path))?;

        let mut npz = NpzReader::new(file)
            .with_context(|| format!("Failed to create NPZ reader for: {:?}", path))?;

        let names = npz
            .names()
            .with_context(|| "Failed to list arrays in performance NPZ")?;
        let has_batch_aware_prefill = npz_has_name(&names, "prefill_batch_size");

        let prefill_interp: Arc<dyn PrefillInterpolator> = if has_batch_aware_prefill {
            let batch_size: Array1<f64> = npz
                .by_name("prefill_batch_size")
                .with_context(|| "Failed to load prefill_batch_size from NPZ")?;
            let new_tokens_per_request: Array1<f64> = npz
                .by_name("prefill_new_tokens_per_request")
                .with_context(|| "Failed to load prefill_new_tokens_per_request from NPZ")?;
            let kv_read_tokens_per_request: Array1<f64> = npz
                .by_name("prefill_kv_read_tokens_per_request")
                .with_context(|| "Failed to load prefill_kv_read_tokens_per_request from NPZ")?;
            let values_ms: Array3<f64> = npz
                .by_name("prefill_time_ms")
                .with_context(|| "Failed to load prefill_time_ms from NPZ")?;
            let expected_shape = (
                batch_size.len(),
                new_tokens_per_request.len(),
                kv_read_tokens_per_request.len(),
            );
            if values_ms.dim() != expected_shape {
                bail!(
                    "Batch-aware prefill array dimension mismatch: values={:?}, axes={:?}",
                    values_ms.dim(),
                    expected_shape
                );
            }
            validate_interpolation_axis(&batch_size)
                .with_context(|| "Invalid prefill_batch_size axis")?;
            validate_interpolation_axis(&new_tokens_per_request)
                .with_context(|| "Invalid prefill_new_tokens_per_request axis")?;
            validate_interpolation_axis(&kv_read_tokens_per_request)
                .with_context(|| "Invalid prefill_kv_read_tokens_per_request axis")?;
            if values_ms
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                bail!("prefill_time_ms must contain only finite non-negative values");
            }
            tracing::info!(
                "Loaded batch-aware prefill model: grid={}x{}x{}",
                expected_shape.0,
                expected_shape.1,
                expected_shape.2
            );
            Arc::new(PrefillInterp3D {
                batch_size,
                new_tokens_per_request,
                kv_read_tokens_per_request,
                values_ms,
            })
        } else {
            let prefill_isl: Array1<f64> = npz
                .by_name("prefill_isl")
                .with_context(|| "Failed to load prefill_isl from NPZ")?;
            let prefill_ttft_ms: Array1<f64> = npz
                .by_name("prefill_ttft_ms")
                .with_context(|| "Failed to load prefill_ttft_ms from NPZ")?;
            if prefill_isl.len() != prefill_ttft_ms.len() {
                bail!(
                    "Prefill array length mismatch: isl={}, ttft={}",
                    prefill_isl.len(),
                    prefill_ttft_ms.len()
                );
            }
            tracing::info!("Loaded legacy prefill model: points={}", prefill_isl.len());
            let inner = Interp1DBuilder::new(prefill_ttft_ms)
                .x(prefill_isl)
                .strategy(Linear::new().extrapolate(true))
                .build()
                .with_context(|| "Failed to build prefill interpolator")?;
            Arc::new(PrefillInterp1D { inner })
        };

        // Load decode arrays
        let decode_active_kv_tokens: Array1<f64> = npz
            .by_name("decode_active_kv_tokens")
            .with_context(|| "Failed to load decode_active_kv_tokens from NPZ")?;
        let decode_context_length: Array1<f64> = npz
            .by_name("decode_context_length")
            .with_context(|| "Failed to load decode_context_length from NPZ")?;
        let decode_itl: Array2<f64> = npz
            .by_name("decode_itl")
            .with_context(|| "Failed to load decode_itl from NPZ")?;

        if decode_itl.nrows() != decode_active_kv_tokens.len()
            || decode_itl.ncols() != decode_context_length.len()
        {
            anyhow::bail!(
                "Decode array dimension mismatch: itl shape=({}, {}), active_kv={}, context={}",
                decode_itl.nrows(),
                decode_itl.ncols(),
                decode_active_kv_tokens.len(),
                decode_context_length.len()
            );
        }

        let decode_interp = Interp2DBuilder::new(decode_itl)
            .x(decode_active_kv_tokens)
            .y(decode_context_length)
            .strategy(Bilinear::new().extrapolate(true))
            .build()
            .with_context(|| "Failed to build decode interpolator")?;

        Ok(PerfModel::Interpolated {
            prefill_interp,
            decode_interp: Arc::new(DecodeInterp2D {
                inner: decode_interp,
            }),
        })
    }

    /// Create an Aiconfigurator perf model from a callback.
    pub fn from_aic_callback(callback: Arc<dyn AicCallback>) -> Self {
        PerfModel::Aiconfigurator { callback }
    }

    /// Predict prefill time in milliseconds.
    ///
    /// Callers always pass all parameters; each variant uses what it needs:
    /// - Polynomial: uses total new tokens across the batch.
    /// - Interpolated: legacy profiles use total new tokens; batch-aware
    ///   profiles use batch size, fresh tokens/request, and prefix tokens/request.
    /// - Aiconfigurator: passes (batch_size, isl - prefix, prefix) to the AIC SDK
    pub fn predict_prefill_time(&self, batch_size: usize, isl: usize, prefix: usize) -> f64 {
        let new_tokens_per_req = isl.saturating_sub(prefix);
        if batch_size == 0 || new_tokens_per_req == 0 {
            return 0.0;
        }
        let time = match self {
            PerfModel::Polynomial => {
                // Total tokens across the batch — GPU processes them in parallel
                let tokens = (batch_size * new_tokens_per_req) as f64;
                4.209989e-07 * tokens.powi(2) + 1.518344e-02 * tokens + 1.650142e+01
            }
            PerfModel::Interpolated { prefill_interp, .. } => prefill_interp
                .interp(batch_size as f64, new_tokens_per_req as f64, prefix as f64)
                .unwrap_or(0.0),
            PerfModel::Aiconfigurator { callback } => {
                callback.predict_prefill(batch_size, new_tokens_per_req, prefix)
            }
        };
        time.max(0.0)
    }

    /// Predict decode time in milliseconds.
    ///
    /// Callers always pass all parameters; each variant uses what it needs:
    /// - Polynomial: uses (active_kv_tokens, total_kv_tokens) as utilization
    /// - Interpolated: uses (active_kv_tokens, context_length)
    /// - Aiconfigurator: uses (batch_size, context_length)
    pub fn predict_decode_time(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        context_length: usize,
        total_kv_tokens: usize,
    ) -> f64 {
        if batch_size == 0 {
            return 0.0;
        }
        let time = match self {
            PerfModel::Polynomial => {
                let active_perc = if total_kv_tokens > 0 {
                    active_kv_tokens as f64 / total_kv_tokens as f64
                } else {
                    tracing::warn!("Total KV tokens is 0, using 1.0 as capacity");
                    1.0
                };
                -25.74 * active_perc.powi(2) + 54.01 * active_perc + 5.74
            }
            PerfModel::Interpolated { decode_interp, .. } => decode_interp
                .interp(active_kv_tokens as f64, context_length as f64)
                .unwrap_or(0.0),
            PerfModel::Aiconfigurator { callback } => {
                callback.predict_decode(batch_size, context_length, 2)
            }
        };
        // Token-emitting decode steps should not collapse onto the same timestamp.
        let result = time.max(1.0);
        tracing::trace!(
            "Decode time prediction: batch_size={batch_size}, active_kv_tokens={active_kv_tokens}, context_length={context_length}, time={result:.2}ms"
        );
        result
    }
}

fn npz_has_name(names: &[String], expected: &str) -> bool {
    names
        .iter()
        .any(|name| name == expected || name == &format!("{expected}.npy"))
}

#[cfg(test)]
mod tests {
    use super::{AicCallback, PerfModel, PrefillInterp3D, PrefillInterpolator};
    use ndarray::{Array1, Array3};
    use std::sync::Arc;

    struct EchoBatchCallback;

    impl AicCallback for EchoBatchCallback {
        fn predict_prefill(&self, batch_size: usize, _effective_isl: usize, _prefix: usize) -> f64 {
            batch_size as f64
        }

        fn predict_decode(&self, batch_size: usize, _isl: usize, _osl: usize) -> f64 {
            batch_size as f64
        }
    }

    #[test]
    fn fully_cached_prompt_skips_prefill() {
        assert_eq!(PerfModel::default().predict_prefill_time(1, 128, 128), 0.0);
    }

    #[test]
    fn aic_forwards_scheduler_local_batch() {
        let model = PerfModel::from_aic_callback(Arc::new(EchoBatchCallback));

        assert_eq!(model.predict_prefill_time(7, 128, 0), 7.0);
        assert_eq!(model.predict_decode_time(9, 0, 128, 0), 9.0);
    }

    #[test]
    fn batch_aware_prefill_interpolates_and_extrapolates_all_axes() {
        let batch = Array1::from_vec(vec![1.0, 4.0]);
        let fresh = Array1::from_vec(vec![64.0, 128.0]);
        let prefix = Array1::from_vec(vec![0.0, 256.0]);
        let values_ms = Array3::from_shape_fn((2, 2, 2), |(b, n, k)| {
            2.0 * batch[b] + 3.0 * fresh[n] + 5.0 * prefix[k]
        });
        let interpolator = PrefillInterp3D {
            batch_size: batch,
            new_tokens_per_request: fresh,
            kv_read_tokens_per_request: prefix,
            values_ms,
        };

        let interpolated = interpolator.interp(2.5, 96.0, 128.0).unwrap();
        assert!((interpolated - (2.0 * 2.5 + 3.0 * 96.0 + 5.0 * 128.0)).abs() < 1e-9);

        let extrapolated = interpolator.interp(7.0, 160.0, 512.0).unwrap();
        assert!((extrapolated - (2.0 * 7.0 + 3.0 * 160.0 + 5.0 * 512.0)).abs() < 1e-9);
    }
}
