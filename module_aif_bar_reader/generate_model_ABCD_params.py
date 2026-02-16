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

    No = [
        4,             # bar1 outcome (NULL, BELOW, IN, ABOVE)
        4,             # bar2 outcome
        4              # feedback
    ]

    return Ns, No, Nfine, Ncoarse


# --------------------------------------------------------
# A matrices (TRI-STATE RULER MODEL)
# --------------------------------------------------------

def build_A(n_ticks, n_fine, n_coarse, dtype=np.float32):

    Ns, No, Nfine, Ncoarse = get_dimensions(n_ticks, n_fine, n_coarse)

    A0 = np.zeros((4, Nfine, Nfine, 3, Ncoarse, Ncoarse + 1), dtype=dtype)
    A1 = np.zeros_like(A0)
    A2 = np.zeros((4, Nfine, Nfine, 3, Ncoarse, Ncoarse + 1), dtype=dtype)

    for f1 in range(Nfine):
        for f2 in range(Nfine):

            true_bin1 = (f1 * n_coarse) // n_fine
            true_bin2 = (f2 * n_coarse) // n_fine

            avg = (f1 + f2) // 2
            true_avg_bin = (avg * n_coarse) // n_fine

            for attn in range(3):
                for q in range(Ncoarse):
                    for rep in range(Ncoarse + 1):

                        # ------------------------------
                        # A0 : bar1 ruler query
                        # ------------------------------
                        if attn == 0:
                            if true_bin1 < q:
                                A0[1, f1, f2, attn, q, rep] = 1.0  # BELOW
                            elif true_bin1 == q:
                                A0[2, f1, f2, attn, q, rep] = 1.0  # IN
                            else:
                                A0[3, f1, f2, attn, q, rep] = 1.0  # ABOVE
                        else:
                            A0[0, f1, f2, attn, q, rep] = 1.0      # NULL

                        # ------------------------------
                        # A1 : bar2 ruler query
                        # ------------------------------
                        if attn == 1:
                            if true_bin2 < q:
                                A1[1, f1, f2, attn, q, rep] = 1.0
                            elif true_bin2 == q:
                                A1[2, f1, f2, attn, q, rep] = 1.0
                            else:
                                A1[3, f1, f2, attn, q, rep] = 1.0
                        else:
                            A1[0, f1, f2, attn, q, rep] = 1.0

                        # ------------------------------
                        # A2 : average report
                        # ------------------------------
                        if attn != 2:
                            A2[0, f1, f2, attn, q, rep] = 1.0
                        else:
                            if rep == 0:
                                A2[0, f1, f2, attn, q, rep] = 1.0
                            else:
                                report_bin = rep - 1
                                if report_bin == true_avg_bin:
                                    A2[3, f1, f2, attn, q, rep] = 1.0  # VERY_CLOSE
                                else:
                                    A2[1, f1, f2, attn, q, rep] = 1.0  # NOT_CLOSE

    # Normalize each likelihood modality
    A0 = normalize(A0, axis=0)
    A1 = normalize(A1, axis=0)
    A2 = normalize(A2, axis=0)

    # Proper object wrapping for pymdp
    A = np.empty(3, dtype=object)
    A[0] = A0
    A[1] = A1
    A[2] = A2

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

    C = np.empty(3, dtype=object)
    C[0] = np.zeros(No[0], dtype=dtype)
    C[1] = np.zeros(No[1], dtype=dtype)
    C[2] = np.zeros(No[2], dtype=dtype)

    # prefer correct report
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
