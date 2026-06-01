from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


def _make_gated_delta_kernel(has_mask=False, vectorized=False):
    if not mx.metal.is_available():
        return None
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"

    # Configure g indexing based on whether gating is vectorized
    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        {g_comment}
        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_source}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * {g_access};
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }} else {{
            y[dv_idx] = static_cast<InT>(0);
          }}
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          {g_advance}
          beta_ += Hv;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = ""
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"

    return mx.fast.metal_kernel(
        name=f"gated_delta_step{suffix}",
        input_names=inputs,
        output_names=["y", "state_out"],
        source=source,
    )


def _is_vulkan_available():
    vulkan = getattr(mx, "vulkan", None)
    return vulkan is not None and vulkan.is_available()


def _make_gated_delta_vulkan_kernel(has_mask=False, vectorized=False):
    if not _is_vulkan_available():
        return None
    vulkan_kernel = getattr(mx.fast, "vulkan_kernel", None)
    if vulkan_kernel is None:
        return None

    mask_source = "(mask.data[b_idx * T + t] != uint8_t(0))" if has_mask else "true"
    if vectorized:
        g_access = "read_GtT(g.data[((b_idx * T + t) * Hv + hv_idx) * Dk + s_idx])"
    else:
        g_access = "read_GtT(g.data[(b_idx * T + t) * Hv + hv_idx])"

    source = f"""
        uint n = gl_GlobalInvocationID.z;
        uint b_idx = n / uint(Hv);
        uint hv_idx = n % uint(Hv);
        uint hk_idx = hv_idx / uint(Hv / Hk);
        uint dk_lane = gl_LocalInvocationID.x;
        uint dv_idx = gl_GlobalInvocationID.y;
        bool active_dv = dv_idx < uint(Dv);
        const uint n_per_t = uint(Dk / 32);

        uint state_base = ((b_idx * uint(Hv) + hv_idx) * uint(Dv) + dv_idx) * uint(Dk);
        float state[8];
        for (uint i = 0; i < n_per_t; ++i) {{
          uint s_idx = n_per_t * dk_lane + i;
          state[i] = active_dv ? read_StT(state_in.data[state_base + s_idx]) : 0.0;
        }}

        for (uint t = 0; t < uint(T); ++t) {{
          bool is_active = active_dv && ({mask_source});
          float kv_mem = 0.0;
          for (uint i = 0; i < n_per_t; ++i) {{
            uint s_idx = n_per_t * dk_lane + i;
            uint qk_idx = ((b_idx * uint(T) + t) * uint(Hk) + hk_idx) * uint(Dk) + s_idx;
            if (is_active) {{
              state[i] *= {g_access};
              kv_mem += state[i] * read_InT(k.data[qk_idx]);
            }}
          }}

          kv_mem = subgroupAdd(kv_mem);

          uint v_idx = ((b_idx * uint(T) + t) * uint(Hv) + hv_idx) * uint(Dv) + dv_idx;
          float delta = is_active
              ? (read_InT(v.data[v_idx]) - kv_mem) *
                    read_BtT(beta.data[(b_idx * uint(T) + t) * uint(Hv) + hv_idx])
              : 0.0;

          float out_acc = 0.0;
          for (uint i = 0; i < n_per_t; ++i) {{
            uint s_idx = n_per_t * dk_lane + i;
            uint qk_idx = ((b_idx * uint(T) + t) * uint(Hk) + hk_idx) * uint(Dk) + s_idx;
            if (is_active) {{
              state[i] += read_InT(k.data[qk_idx]) * delta;
              out_acc += state[i] * read_InT(q.data[qk_idx]);
            }}
          }}

          out_acc = subgroupAdd(out_acc);
          if (dk_lane == 0 && active_dv) {{
            y.data[v_idx] = write_InT(is_active ? out_acc : 0.0);
          }}
        }}

        if (active_dv) {{
          for (uint i = 0; i < n_per_t; ++i) {{
            uint s_idx = n_per_t * dk_lane + i;
            state_out.data[state_base + s_idx] = write_StT(state[i]);
          }}
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in"]
    if has_mask:
        inputs.append("mask")

    suffix = ""
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"

    try:
        return vulkan_kernel(
            name=f"gated_delta_step{suffix}",
            input_names=inputs,
            output_names=["y", "state_out"],
            header="#extension GL_KHR_shader_subgroup_arithmetic : require\n",
            source=source,
        )
    except RuntimeError:
        return None


_gated_delta_kernel = _make_gated_delta_kernel(has_mask=False, vectorized=False)
_gated_delta_kernel_masked = _make_gated_delta_kernel(has_mask=True, vectorized=False)
_gated_delta_kernel_vec = _make_gated_delta_kernel(has_mask=False, vectorized=True)
_gated_delta_kernel_vec_masked = _make_gated_delta_kernel(
    has_mask=True, vectorized=True
)
_gated_delta_vulkan_kernel = _make_gated_delta_vulkan_kernel(
    has_mask=False, vectorized=False
)
_gated_delta_vulkan_kernel_masked = _make_gated_delta_vulkan_kernel(
    has_mask=True, vectorized=False
)
_gated_delta_vulkan_kernel_vec = _make_gated_delta_vulkan_kernel(
    has_mask=False, vectorized=True
)
_gated_delta_vulkan_kernel_vec_masked = _make_gated_delta_vulkan_kernel(
    has_mask=True, vectorized=True
)


@mx.compile
def _gated_delta_step_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for a single recurrent step.

    Shapes:
      - q, k: [B, H, Dk]
      - v: [B, H, Dv]
      - g: [B, H] or [B, H, Dk]
      - beta: [B, H]
      - state: [B, H, Dv, Dk]
    Returns:
      - y: [B, H, Dv]
      - new_state: [B, H, Dv, Dk]
    """

    # Decay
    old_state = state
    if g.ndim == 2:
        decay = g[..., None, None]
    elif g.ndim == 3:
        decay = g[..., None, :]
    else:
        raise ValueError(f"Unsupported gating shape {g.shape}")
    state = state * decay
    kv_mem = (state * k[..., None, :]).sum(axis=-1)  # [B, H, Dv]
    delta = (v - kv_mem) * beta[..., None]  # [B, H, Dv]
    state = state + k[..., None, :] * delta[..., None]
    # Output projection along key dim with q
    y = (state * q[..., None, :]).sum(axis=-1)  # [B, H, Dv]

    if mask is not None:
        mask = mx.expand_dims(mask, axis=(1, 2, 3))
        state = mx.where(mask, state, old_state)
    return y.astype(q.dtype), state


def gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    input_type = q.dtype
    state_type = state.dtype
    if g.ndim == 4:
        kernel = _gated_delta_kernel_vec
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_kernel_vec_masked
            inputs.append(mask)
    else:
        kernel = _gated_delta_kernel
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_kernel_masked
            inputs.append(mask)

    if kernel is None:
        return gated_delta_ops(q, k, v, g, beta, state, mask)

    return kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[input_type, state_type],
    )


def gated_delta_vulkan_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    input_type = q.dtype
    state_type = state.dtype
    if Dk % 32 != 0 or Dk // 32 > 8:
        return gated_delta_ops(q, k, v, g, beta, state, mask)

    if g.ndim == 4:
        kernel = _gated_delta_vulkan_kernel_vec
        inputs = [q, k, v, g, beta, state]
        if mask is not None:
            kernel = _gated_delta_vulkan_kernel_vec_masked
            inputs.append(mask)
    else:
        kernel = _gated_delta_vulkan_kernel
        inputs = [q, k, v, g, beta, state]
        if mask is not None:
            kernel = _gated_delta_vulkan_kernel_masked
            inputs.append(mask)

    if kernel is None:
        return gated_delta_ops(q, k, v, g, beta, state, mask)

    return kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("GtT", g.dtype),
            ("BtT", beta.dtype),
            ("T", T),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[input_type, state_type],
    )


def _select_gated_delta_kernel():
    if mx.metal.is_available():
        return gated_delta_kernel
    if _is_vulkan_available():
        return gated_delta_vulkan_kernel
    return None


def gated_delta_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for prompt prefill (sequential loop).
    Supports both scalar and vectorized gating.

    Shapes:
      - q, k: [B, T, Hk, Dk]
      - v: [B, T, Hv, Dv]
      - g: [B, T, Hv] (scalar) or [B, T, Hv, Dk] (vectorized)
      - beta: [B, T, Hv]
      - state: [B, Hv, Dv, Dk]
    Returns:
      - y: [B, T, Hv, Dv]
      - state: [B, Hv, Dv, Dk]
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    ys = []
    for t in range(T):
        y, state = _gated_delta_step_ops(
            q[:, t],
            k[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            state,
            None if mask is None else mask[:, t],
        )
        ys.append(y)
    y = mx.stack(ys, axis=1)
    return y, state


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    use_kernel: bool = True,
) -> Tuple[mx.array, mx.array]:
    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    if state is None:
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if not use_kernel or mx.default_device() != mx.gpu:
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    if kernel := _select_gated_delta_kernel():
        return kernel(q, k, v, g, beta, state, mask)
    return gated_delta_ops(q, k, v, g, beta, state, mask)
