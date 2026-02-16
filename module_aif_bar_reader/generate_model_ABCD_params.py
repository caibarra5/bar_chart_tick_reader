# filename: generate_model_ABCD_params.py
import numpy as np


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
        Ncoarse,       # coarse_query
        Ncoarse + 1    # report_choice
    ]

    No = [
        3,             # bar1 outcome (NULL, IN, OUT)
        3,             # bar2 outcome
        4              # feedback
    ]

    return Ns, No, Nfine, Ncoarse




# --------------------------------------------------------
# A matrices
# --------------------------------------------------------

def build_A(n_ticks, n_fine, n_coarse, dtype=np.float32):

    Ns, No, Nfine, Ncoarse = get_dimensions(n_ticks, n_fine, n_coarse)

    A0 = np.zeros((3, Nfine, Nfine, 3, Ncoarse, Ncoarse + 1), dtype=dtype)
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

                        # A0
                        if attn == 0:
                            if q == true_bin1:
                                A0[1, f1, f2, attn, q, rep] = 1.0
                            else:
                                A0[2, f1, f2, attn, q, rep] = 1.0
                        else:
                            A0[0, f1, f2, attn, q, rep] = 1.0

                        # A1
                        if attn == 1:
                            if q == true_bin2:
                                A1[1, f1, f2, attn, q, rep] = 1.0
                            else:
                                A1[2, f1, f2, attn, q, rep] = 1.0
                        else:
                            A1[0, f1, f2, attn, q, rep] = 1.0

                        # A2
                        if attn != 2:
                            A2[0, f1, f2, attn, q, rep] = 1.0
                        else:
                            if rep == 0:
                                A2[0, f1, f2, attn, q, rep] = 1.0
                            else:
                                report_bin = rep - 1
                                if report_bin == true_avg_bin:
                                    A2[3, f1, f2, attn, q, rep] = 1.0
                                else:
                                    A2[1, f1, f2, attn, q, rep] = 1.0

    return np.array([A0, A1, A2], dtype=object)





# --------------------------------------------------------
# B matrices
# --------------------------------------------------------

def build_B(n_ticks, n_fine, n_coarse, dtype=np.float32):

    _, _, Nfine, Ncoarse = get_dimensions(n_ticks, n_fine, n_coarse)

    B0 = np.eye(Nfine)[:, :, None]
    B1 = np.eye(Nfine)[:, :, None]

    # attention control
    B2 = np.zeros((3, 3, 3))
    for u in range(3):
        B2[u, :, u] = 1.0

    # coarse_query control  ✅ FIXED
    B3 = np.zeros((Ncoarse, Ncoarse, Ncoarse))
    for u in range(Ncoarse):
        B3[u, :, u] = 1.0

    # report_choice control
    B4 = np.zeros((Ncoarse + 1, Ncoarse + 1, Ncoarse + 1))
    for u in range(Ncoarse + 1):
        B4[u, :, u] = 1.0

    return np.array([B0, B1, B2, B3, B4], dtype=object)




# --------------------------------------------------------
# C preferences
# --------------------------------------------------------

def build_C(n_ticks, n_fine, n_coarse, dtype=np.float32):

    _, No, _, _ = get_dimensions(n_ticks,
                                 n_fine,
                                 n_coarse)

    C = [np.zeros(n, dtype=dtype) for n in No]

    C[2][1] = -6.0
    C[2][3] = 6.0

    return np.array(C, dtype=object)


# --------------------------------------------------------
# D priors
# --------------------------------------------------------

def build_D(n_ticks, n_fine, n_coarse, dtype=np.float32):

    _, _, Nfine, Ncoarse = get_dimensions(n_ticks, n_fine, n_coarse)

    D0 = np.ones(Nfine) / Nfine        # bar1
    D1 = np.ones(Nfine) / Nfine        # bar2
    D2 = np.array([1.0, 0.0, 0.0])     # attention
    D3 = np.ones(Ncoarse) / Ncoarse    # coarse_query  ✅ FIX
    D4 = np.zeros(Ncoarse + 1)         # report_choice
    D4[0] = 1.0

    return np.array([D0, D1, D2, D3, D4], dtype=object)


