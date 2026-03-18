# filename: generate_model_ABCD_params.py

import numpy as np


# --------------------------------------------------------
# Utility
# --------------------------------------------------------

def normalize(x, axis=0, eps=1e-16):
    s = x.sum(axis=axis, keepdims=True)
    s = np.where(s < eps, 1.0, s)
    return x / s


# --------------------------------------------------------
# Dimensions
# --------------------------------------------------------

def get_dimensions(n_ticks, n_fine, n_coarse):

    Nfine = n_ticks * n_fine
    Ncoarse = n_ticks * n_coarse

    Ns = [
        Nfine,         # bar1_fine
        Nfine,         # bar2_fine
        3,             # attention
        Ncoarse,       # coarse_query (threshold)
        Ncoarse + 1    # report_choice
    ]

    # Observation modalities
    No = [
        4,                  # bar1 coarse ruler
        4,                  # bar2 coarse ruler
        4,                  # feedback
        n_ticks + 1,        # bar1 tick interval (0=NULL)
        n_ticks + 1         # bar2 tick interval
    ]

    return Ns, No, Nfine, Ncoarse



# --------------------------------------------------------
# A matrices (TRI-STATE RULER MODEL)
# --------------------------------------------------------

def build_A(n_ticks, n_fine, n_coarse, sigma_coarse=0.5, sigma_avg=0.5, dtype=np.float32):

    Ns, No, Nfine, Ncoarse = get_dimensions(n_ticks, n_fine, n_coarse)

    # --- allocate ---
    A0 = np.zeros((4, Nfine, Nfine, 3, Ncoarse, Ncoarse + 1), dtype=dtype)
    A1 = np.zeros_like(A0)
    A2 = np.zeros_like(A0)

    A3 = np.zeros((n_ticks + 1, Nfine, Nfine, 3, Ncoarse, Ncoarse + 1), dtype=dtype)
    A4 = np.zeros_like(A3)

    def gaussian_over_bins(mu, n_bins, sigma):
        xs = np.arange(n_bins)
        p = np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
        p /= p.sum()
        return p

    for f1 in range(Nfine):
        for f2 in range(Nfine):

            # Continuous mapping fine → coarse space
            mu1 = (f1 + 0.5) * n_coarse / n_fine
            mu2 = (f2 + 0.5) * n_coarse / n_fine

            p_coarse1 = gaussian_over_bins(mu1, Ncoarse, sigma_coarse)
            p_coarse2 = gaussian_over_bins(mu2, Ncoarse, sigma_coarse)

            # Average (continuous)
            mu_avg = ((f1 + f2) / 2 + 0.5) * n_coarse / n_fine
            p_avg = gaussian_over_bins(mu_avg, Ncoarse, sigma_avg)

            tick1 = f1 // n_fine
            tick2 = f2 // n_fine

            for attn in range(3):
                for q in range(Ncoarse):
                    for rep in range(Ncoarse + 1):

                        # ------------------------------
                        # A0 : bar1 coarse ruler (Gaussian)
                        # ------------------------------
                        if attn == 0:
                            below = p_coarse1[:q].sum()
                            equal = p_coarse1[q]
                            above = p_coarse1[q+1:].sum()

                            A0[1, f1, f2, attn, q, rep] = below
                            A0[2, f1, f2, attn, q, rep] = equal
                            A0[3, f1, f2, attn, q, rep] = above
                        else:
                            A0[0, f1, f2, attn, q, rep] = 1.0

                        # ------------------------------
                        # A1 : bar2 coarse ruler (Gaussian)
                        # ------------------------------
                        if attn == 1:
                            below = p_coarse2[:q].sum()
                            equal = p_coarse2[q]
                            above = p_coarse2[q+1:].sum()

                            A1[1, f1, f2, attn, q, rep] = below
                            A1[2, f1, f2, attn, q, rep] = equal
                            A1[3, f1, f2, attn, q, rep] = above
                        else:
                            A1[0, f1, f2, attn, q, rep] = 1.0

                        # ------------------------------
                        # A2 : average report (Gaussian)
                        # ------------------------------
                        if attn != 2:
                            A2[0, f1, f2, attn, q, rep] = 1.0
                        else:
                            if rep == 0:
                                A2[0, f1, f2, attn, q, rep] = 1.0
                            else:
                                report_bin = rep - 1
                                A2[3, f1, f2, attn, q, rep] = p_avg[report_bin]
                                A2[1, f1, f2, attn, q, rep] = 1.0 - p_avg[report_bin]

                        # ------------------------------
                        # A3 : bar1 tick interval (deterministic)
                        # ------------------------------
                        if attn == 0:
                            A3[tick1 + 1, f1, f2, attn, q, rep] = 1.0
                        else:
                            A3[0, f1, f2, attn, q, rep] = 1.0

                        # ------------------------------
                        # A4 : bar2 tick interval (deterministic)
                        # ------------------------------
                        if attn == 1:
                            A4[tick2 + 1, f1, f2, attn, q, rep] = 1.0
                        else:
                            A4[0, f1, f2, attn, q, rep] = 1.0

    # Normalize all modalities
    A0 = normalize(A0, axis=0)
    A1 = normalize(A1, axis=0)
    A2 = normalize(A2, axis=0)
    A3 = normalize(A3, axis=0)
    A4 = normalize(A4, axis=0)

    A = np.empty(5, dtype=object)
    A[0] = A0
    A[1] = A1
    A[2] = A2
    A[3] = A3
    A[4] = A4

    return A



# --------------------------------------------------------
# B matrices
# --------------------------------------------------------

def build_B(n_ticks, n_fine, n_coarse, dtype=np.float32):

    _, _, Nfine, Ncoarse = get_dimensions(n_ticks, n_fine, n_coarse)

    B0 = np.eye(Nfine)[:, :, None]
    B1 = np.eye(Nfine)[:, :, None]

    B2 = np.zeros((3, 3, 3))
    for u in range(3):
        B2[u, :, u] = 1.0

    B3 = np.zeros((Ncoarse, Ncoarse, Ncoarse))
    for u in range(Ncoarse):
        B3[u, :, u] = 1.0

    B4 = np.zeros((Ncoarse + 1, Ncoarse + 1, Ncoarse + 1))
    for u in range(Ncoarse + 1):
        B4[u, :, u] = 1.0

    B = np.empty(5, dtype=object)
    B[0] = B0
    B[1] = B1
    B[2] = B2
    B[3] = B3
    B[4] = B4

    return B


# --------------------------------------------------------
# C preferences
# --------------------------------------------------------

def build_C(n_ticks, n_fine, n_coarse, dtype=np.float32):

    _, No, _, _ = get_dimensions(n_ticks, n_fine, n_coarse)

    # 5 observation modalities now
    C = np.empty(5, dtype=object)

    # bar1 coarse ruler
    C[0] = np.zeros(No[0], dtype=dtype)

    # bar2 coarse ruler
    C[1] = np.zeros(No[1], dtype=dtype)

    # feedback
    C[2] = np.zeros(No[2], dtype=dtype)

    # bar1 tick interval
    C[3] = np.zeros(No[3], dtype=dtype)

    # bar2 tick interval
    C[4] = np.zeros(No[4], dtype=dtype)

    # Only care about report feedback
    # prefer VERY_CLOSE (index 3)
    # dislike NOT_CLOSE_AT_ALL (index 1)
    C[2][1] = -6.0
    C[2][3] = 6.0

    return C



# --------------------------------------------------------
# D priors
# --------------------------------------------------------

def build_D(n_ticks, n_fine, n_coarse, dtype=np.float32):

    _, _, Nfine, Ncoarse = get_dimensions(n_ticks, n_fine, n_coarse)

    D = np.empty(5, dtype=object)
    D[0] = np.ones(Nfine) / Nfine
    D[1] = np.ones(Nfine) / Nfine
    D[2] = np.array([1.0, 0.0, 0.0])
    D[3] = np.ones(Ncoarse) / Ncoarse
    D[4] = np.zeros(Ncoarse + 1)
    D[4][0] = 1.0

    return D
