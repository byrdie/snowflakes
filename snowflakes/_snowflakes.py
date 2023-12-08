import numpy as np
import numba
import matplotlib.pyplot as plt

__all__ = [
    "snowflake",
]


def snowflake(
    a_0: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    theta: float,
    kappa: float,
    mu: float,
    rho: float,
    num_steps: int = 1000,
    num_frames: int = None,
    random: bool = False,
):
    if num_frames is None:
        num_frames = num_steps

    b_0 = np.zeros_like(a_0, dtype=float)
    c_0 = a_0.astype(float)
    d_0 = np.broadcast_to(rho, a_0.shape).copy()
    d_0[a_0] = 0

    return _snowflake(
        a_0=a_0,
        b_0=b_0,
        c_0=c_0,
        d_0=d_0,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        theta=theta,
        kappa=kappa,
        mu=mu,
        num_steps=num_steps,
        num_frames=num_frames,
    )


@numba.njit
def _snowflake(
    a_0: np.ndarray,
    b_0: np.ndarray,
    c_0: np.ndarray,
    d_0: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    theta: float,
    kappa: float,
    mu: float,
    num_steps: int,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_x, num_y = a_0.shape
    shape = (num_frames, num_x, num_y)

    num_steps_per_frame = num_steps // num_frames

    a = np.empty(shape)
    b = np.empty(shape)
    c = np.empty(shape)
    d = np.empty(shape)

    a_n = a_0
    b_n = b_0
    c_n = c_0
    d_n = d_0

    for n in range(num_steps):
        if (n % num_steps_per_frame) == 0:
            f = n // num_steps_per_frame
            a[f] = a_n
            b[f] = b_n
            c[f] = c_n
            d[f] = d_n

        a_n, b_n, c_n, d_n = _diffusion(
            a_n=a_n,
            b_n=b_n,
            c_n=c_n,
            d_n=d_n,
        )
        # print("diffusion")
        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].imshow(a_n)
        # axs[0, 1].imshow(b_n)
        # axs[1, 0].imshow(c_n)
        # axs[1, 1].imshow(d_n)

        a_n, b_n, c_n, d_n = _freezing(
            a_n=a_n,
            b_n=b_n,
            c_n=c_n,
            d_n=d_n,
            kappa=kappa,
        )
        # print("freezing")
        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].imshow(a_n)
        # axs[0, 1].imshow(b_n)
        # axs[1, 0].imshow(c_n)
        # axs[1, 1].imshow(d_n)

        a_n, b_n, c_n, d_n = _attachment(
            a_n=a_n,
            b_n=b_n,
            c_n=c_n,
            d_n=d_n,
            alpha=alpha,
            beta=beta,
            theta=theta,
        )
        # print("attachment")
        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].imshow(a_n)
        # axs[0, 1].imshow(b_n)
        # axs[1, 0].imshow(c_n)
        # axs[1, 1].imshow(d_n)

        a_n, b_n, c_n, d_n = _melting(
            a_n=a_n,
            b_n=b_n,
            c_n=c_n,
            d_n=d_n,
            mu=mu,
            gamma=gamma,
        )
        # print("melting")
        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].imshow(a_n)
        # axs[0, 1].imshow(b_n)
        # axs[1, 0].imshow(c_n)
        # axs[1, 1].imshow(d_n)

    return a, b, c, d


@numba.njit(boundscheck=True, parallel=True)
def _diffusion(
    a_n: np.ndarray,
    b_n: np.ndarray,
    c_n: np.ndarray,
    d_n: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_x, num_y = a_n.shape

    a_result = np.empty_like(a_n)
    b_result = np.empty_like(b_n)
    c_result = np.empty_like(c_n)
    d_result = np.empty_like(d_n)

    for i in numba.prange(num_x):
        for j in numba.prange(num_y):
            a_result[i, j] = a_n[i, j]
            b_result[i, j] = b_n[i, j]
            c_result[i, j] = c_n[i, j]

            if a_n[i, j]:
                d_result[i, j] = d_n[i, j]

            else:
                d_sum = 0
                norm = 0

                for m in range(-1, 2):
                    for n in range(-1, 2):
                        if m == n == -1:
                            continue
                        if m == n == 1:
                            continue

                        p = i + m
                        q = j + n

                        if not (0 <= p < num_x):
                            continue
                        if not (0 <= q < num_y):
                            continue

                        if a_n[p, q]:
                            d_npq = d_n[i, j]
                        else:
                            d_npq = d_n[p, q]

                        d_sum += d_npq
                        norm += 1

                d_result[i, j] = d_sum / norm

    return a_result, b_result, c_result, d_result


@numba.njit(boundscheck=True)
def _on_boundary(
    a_n: np.ndarray,
    i: int,
    j: int,
) -> bool:
    num_x, num_y = a_n.shape

    on_boundary = False

    for m in range(-1, 2):
        for n in range(-1, 2):
            if m == n == -1:
                continue
            if m == n == 0:
                continue
            if m == n == 1:
                continue

            p = i + m
            q = j + n

            if not (0 <= p < num_x):
                continue
            if not (0 <= q < num_y):
                continue

            if a_n[p, q]:
                on_boundary = True
                break

        if on_boundary:
            break

    return on_boundary


@numba.njit(boundscheck=True, parallel=True)
def _freezing(
    a_n: np.ndarray,
    b_n: np.ndarray,
    c_n: np.ndarray,
    d_n: np.ndarray,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_x, num_y = a_n.shape

    a_result = np.empty_like(a_n)
    b_result = np.empty_like(b_n)
    c_result = np.empty_like(c_n)
    d_result = np.empty_like(d_n)

    for i in numba.prange(num_x):
        for j in numba.prange(num_y):
            a_result[i, j] = a_n[i, j]

            if a_n[i, j]:
                b_result[i, j] = b_n[i, j]
                c_result[i, j] = c_n[i, j]
                d_result[i, j] = d_n[i, j]

            else:
                on_boundary = _on_boundary(a_n=a_n, i=i, j=j)

                if on_boundary:
                    b_result[i, j] = b_n[i, j] + (1 - kappa) * d_n[i, j]
                    c_result[i, j] = c_n[i, j] + kappa * d_n[i, j]
                    d_result[i, j] = 0
                else:
                    b_result[i, j] = b_n[i, j]
                    c_result[i, j] = c_n[i, j]
                    d_result[i, j] = d_n[i, j]

    return a_result, b_result, c_result, d_result


@numba.njit(boundscheck=True, parallel=True)
def _attachment(
    a_n: np.ndarray,
    b_n: np.ndarray,
    c_n: np.ndarray,
    d_n: np.ndarray,
    alpha: float,
    beta: float,
    theta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_x, num_y = a_n.shape

    a_result = np.empty_like(a_n)
    b_result = np.empty_like(b_n)
    c_result = np.empty_like(c_n)
    d_result = np.empty_like(d_n)

    for i in numba.prange(num_x):
        for j in numba.prange(num_y):
            if a_n[i, j]:
                a_result[i, j] = a_n[i, j]
                b_result[i, j] = b_n[i, j]
                c_result[i, j] = c_n[i, j]
                d_result[i, j] = d_n[i, j]

            else:
                num_attached_neighbors = 0
                diffusive_mass = 0

                for m in range(-1, 2):
                    for n in range(-1, 2):
                        if m == n == -1:
                            continue
                        if m == n == 0:
                            continue
                        if m == n == 1:
                            continue

                        p = i + m
                        q = j + n

                        if not (0 <= p < num_x):
                            continue
                        if not (0 <= q < num_y):
                            continue

                        if a_n[p, q]:
                            num_attached_neighbors += 1

                        diffusive_mass += d_n[p, q]

                if (num_attached_neighbors == 1) or (num_attached_neighbors == 2):
                    if b_n[i, j] >= beta:
                        a_result[i, j] = 1
                        b_result[i, j] = 0
                        c_result[i, j] = b_n[i, j] + c_n[i, j]
                        d_result[i, j] = d_n[i, j]
                    else:
                        a_result[i, j] = a_n[i, j]
                        b_result[i, j] = b_n[i, j]
                        c_result[i, j] = c_n[i, j]
                        d_result[i, j] = d_n[i, j]

                elif num_attached_neighbors == 3:
                    if b_n[i, j] >= 1:
                        a_result[i, j] = 1
                        b_result[i, j] = 0
                        c_result[i, j] = b_n[i, j] + c_n[i, j]
                        d_result[i, j] = d_n[i, j]
                    elif (diffusive_mass < theta) and (b_n[i, j] >= alpha):
                        a_result[i, j] = 1
                        b_result[i, j] = 0
                        c_result[i, j] = b_n[i, j] + c_n[i, j]
                        d_result[i, j] = d_n[i, j]
                    else:
                        a_result[i, j] = a_n[i, j]
                        b_result[i, j] = b_n[i, j]
                        c_result[i, j] = c_n[i, j]
                        d_result[i, j] = d_n[i, j]

                elif num_attached_neighbors >= 4:
                    a_result[i, j] = 1
                    b_result[i, j] = 0
                    c_result[i, j] = b_n[i, j] + c_n[i, j]
                    d_result[i, j] = d_n[i, j]

                else:
                    a_result[i, j] = a_n[i, j]
                    b_result[i, j] = b_n[i, j]
                    c_result[i, j] = c_n[i, j]
                    d_result[i, j] = d_n[i, j]

    return a_result, b_result, c_result, d_result


@numba.njit(boundscheck=True, parallel=True)
def _melting(
    a_n: np.ndarray,
    b_n: np.ndarray,
    c_n: np.ndarray,
    d_n: np.ndarray,
    mu: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_x, num_y = a_n.shape

    a_result = np.empty_like(a_n)
    b_result = np.empty_like(b_n)
    c_result = np.empty_like(c_n)
    d_result = np.empty_like(d_n)

    for i in numba.prange(num_x):
        for j in numba.prange(num_y):
            a_result[i, j] = a_n[i, j]

            if a_n[i, j]:
                b_result[i, j] = b_n[i, j]
                c_result[i, j] = c_n[i, j]
                d_result[i, j] = d_n[i, j]

            else:
                on_boundary = _on_boundary(a_n=a_n, i=i, j=j)

                if on_boundary:
                    b_result[i, j] = (1 - mu) * b_n[i, j]
                    c_result[i, j] = (1 - gamma) * c_n[i, j]
                    d_result[i, j] = d_n[i, j] + mu * b_n[i, j] + gamma * c_n[i, j]
                else:
                    b_result[i, j] = b_n[i, j]
                    c_result[i, j] = c_n[i, j]
                    d_result[i, j] = d_n[i, j]

    return a_result, b_result, c_result, d_result
