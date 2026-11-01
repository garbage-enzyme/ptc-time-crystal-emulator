"""Modulated Drude slab: harmonic-balance scattering and an independent
time-domain route (bounded pilot per the frozen declarations).

Geometry: vacuum background, slab 0 < z < d, eps_inf = 1, gamma const,
plasma modulation wp2(t) = 1 + m cos(Om t) (accepted source assumptions).
Incident unit rightgoing wave at frequency w0; channel frequencies
w_n = w0 + n Om are chosen by the caller so that no channel hits
w = 0 or +-1 exactly (signed-frequency channels, including negative
ones, are retained).

Method A (frequency mixing, exact linear algebra): per channel
S_n = E_n/(gamma - i w_n), P_n = (i/w_n) sum_q g_q S_{n-q} with
g_0 = 1, g_{+-1} = m/2, so D = M E block-couples neighbours. The
z-dynamics u = (E, E') obey u' = C u with the CONSTANT matrix
C = [[0, I], [-W^2 M, 0]], propagated by the matrix exponential; the
interface conditions (E and B = E'/(i w) continuity against vacuum
unit-normalized incident/reflected/transmitted channels) close a
(2N+1)-unknown linear solve. m = 0 reduces EXACTLY to the static slab.

Method B (independent time domain): method-of-lines DOP853 for
(E, B, S, P) on a uniform z grid with characteristic one-way boundaries
(exact in continuum 1D), prescribed incident wave at the left edge and
zero initial state; sideband amplitudes are Fourier-projected at probes
after transients decay.

Energy identity (steady periodic state): with unit incident flux,
outgoing O = sum |r_n|^2 + sum |t_n|^2 and net modulation work minus
dissipation W = int int [0.5 wp2'|S|^2 - gamma wp2 |S|^2] dz dt over
the slab and one period, computed exactly from the harmonic amplitudes;
incident + modulation work = outgoing + loss closes as O - 1 = W.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

__all__ = [
    "MODULATION_NOTE",
    "channel_frequencies",
    "mixing_matrix",
    "modulated_slab_channels",
    "modulated_energy_terms",
    "time_domain_slab",
    "project_sidebands",
]

MODULATION_NOTE = (
    "modulated slab pilot; signed-frequency sidebands; O - 1 = W_mod - "
    "Diss (no R+T=1 imposed); NOT an infinite-medium bandgap claim")


def channel_frequencies(w0, Omega, N):
    n = np.arange(-N, N + 1)
    return n, w0 + n * Omega


def mixing_matrix(w, m, gamma, eps_inf=1.0, Omega=None):
    """M[n, n'] = eps_inf delta + (i/w_n) g_{n-n'} /(gamma - i w_{n'})
    with g_0 = 1, g_{+-1} = m/2 (single-harmonic cos modulation)."""
    w = np.asarray(w, dtype=float)
    nn = w.size
    M = np.zeros((nn, nn), dtype=complex)
    for i in range(nn):
        M[i, i] = eps_inf + (1j / w[i]) / (gamma - 1j * w[i])
        for off in (-1, +1):
            j = i + off
            if 0 <= j < nn:
                g = 0.5 * m
                M[i, j] = (1j / w[i]) * g / (gamma - 1j * w[j])
    return M


def modulated_slab_channels(w0, Omega, m, d, gamma, N, eps_inf=1.0):
    """Method A: solve the harmonic-balance BVP. Returns a dict with
    r, t (complex, indexed by channel), the propagation matrix result,
    and the E'(0) solve residual norm."""
    n_idx, w = channel_frequencies(w0, Omega, N)
    nn = w.size
    M = mixing_matrix(w, m, gamma, eps_inf)
    W2 = np.diag(w ** 2)
    C = np.zeros((2 * nn, 2 * nn), dtype=complex)
    C[:nn, nn:] = np.eye(nn)
    C[nn:, :nn] = -W2 @ M
    P = expm(C * d)
    e0 = np.zeros(nn, dtype=complex)
    e0[int(np.argmin(np.abs(n_idx)))] = 1.0
    Dw = np.diag(1.0 / (1j * w))
    # u(0) = b0 + S0 v  with  E(0) = 2 e0 - Dw v,  E'(0) = v
    b0 = np.concatenate([2.0 * e0, np.zeros(nn)])
    S0 = np.vstack([-Dw, np.eye(nn)])
    R1 = P[:nn, :]
    R2 = P[nn:, :]
    A_sys = (R2 - 1j * np.diag(w) @ R1) @ S0
    b_sys = -(R2 - 1j * np.diag(w) @ R1) @ b0
    v = np.linalg.solve(A_sys, b_sys)
    resid = float(np.max(np.abs(A_sys @ v - b_sys)))
    r = e0 - Dw @ v
    u0 = b0 + S0 @ v
    ud = P @ u0
    t = ud[:nn]
    return {"n": n_idx, "w": w, "r": r, "t": t, "v": v,
            "u0": u0, "residual": resid, "M": M, "C": C}


def modulated_energy_terms(w0, Omega, m, d, gamma, N, n_panels=50,
                           order=10, eps_inf=1.0):
    """Exact harmonic evaluation of the steady-state energy terms:
    outgoing O = sum |r_n|^2 + sum |t_n|^2 (unit incident flux) and
    W = int over the slab of the period average of
    [0.5 wp2'(t)|S|^2 - gamma wp2(t)|S|^2] dz (modulation work minus
    dissipation per period). Time averages of harmonic sums use
    <|S|^2> = sum |S_n|^2, <cos(Om t)|S|^2> = (A_+1 + A_-1)/2 and
    <sin(Om t)|S|^2> = (A_+1 - A_-1)/(2i) with
    A_k = sum_n S_n conj(S_{n-k}); the z integral is composite Gauss-
    Legendre with the state propagated EXACTLY (expm) at the quadrature
    nodes — no interpolation."""
    sol = modulated_slab_channels(w0, Omega, m, d, gamma, N, eps_inf)
    n_idx, w = sol["n"], sol["w"]
    C = sol["C"]
    u0 = sol["u0"]
    xi, wgt = leggauss_local(order)
    edges = np.linspace(0.0, d, n_panels + 1)
    i_of = {int(nv): i for i, nv in enumerate(n_idx)}
    W_acc = 0.0
    Wmod_acc = 0.0
    Diss_acc = 0.0
    for p_idx in range(n_panels):
        a, b = edges[p_idx], edges[p_idx + 1]
        nodes = 0.5 * (a + b) + 0.5 * (b - a) * xi
        half = 0.5 * (b - a)
        panel_W = 0.0
        panel_wm = 0.0
        panel_di = 0.0
        for q, zz in enumerate(nodes):
            uz = expm(C * zz) @ u0
            S = uz[:len(w)] / (gamma - 1j * w)

            def A_k(k):
                # coefficient of e^{-i k Om t} in s s* (m = n - k)
                tot = 0.0 + 0.0j
                for i, nv in enumerate(n_idx):
                    j = i_of.get(int(nv) - k)
                    if j is not None:
                        tot += S[i] * np.conj(S[j])
                return tot
            A1, Am1 = A_k(1), A_k(-1)
            # REAL-signal time averages (physical field = Re s):
            # <S_real^2> = 0.5 sum |S_n|^2, <sin S_real^2> =
            # 0.5 Re[(A1 - Am1)/(2j)], <cos S_real^2> = 0.25 (A1 + Am1)
            absS2 = float(np.sum(np.abs(S) ** 2))
            wmod = 0.5 * (-m * Omega) * 0.5 * float(
                np.real((A1 - Am1) / 2j))
            diss = gamma * float(np.real(
                0.5 * absS2 + 0.25 * m * (A1 + Am1)))
            panel_wm += wgt[q] * wmod
            panel_di += wgt[q] * diss
        W_acc += half * (panel_wm - panel_di)
        Wmod_acc += half * panel_wm
        Diss_acc += half * panel_di
    O = float(np.sum(np.abs(sol["r"]) ** 2)
              + np.sum(np.abs(sol["t"]) ** 2))
    # unit incident amplitude carries flux 1/2, so the net outgoing
    # flux is (O - 1)/2 and the steady-state identity closes as
    # O - 1 = 2 W (modulation work minus dissipation, per period)
    return {"O": O, "W": W_acc, "W_mod": Wmod_acc, "Diss": Diss_acc,
            "closure": O - 1.0 - 2.0 * W_acc, "sol": sol}


def leggauss_local(order):
    return np.polynomial.legendre.leggauss(order)

def time_domain_slab(w0, Omega, m, d, gamma, z_min=-8.0, z_max=10.0,
                     h=0.025, t_end=320.0, rtol=1e-10, atol=1e-12,
                     ramp=60.0, sigma_max=8.0, sponge_start=4.0,
                     sponge_width=4.0, N=8, K_periods=None):
    """Method B: independent time-domain route (stable formulation).

    Sponge-damped method-of-lines (DOP853): smooth quadratic absorbing
    sponges in the outer vacuum regions make the semi-discretization
    dissipative; a smooth body-force strip near z_min launches the
    incident wave (the left-going part is absorbed by the sponge).
    Linearity is used EXACTLY: a reference integration WITHOUT the slab
    shares the source, sponges and numerics, so the probe differences
    isolate the scattered fields, and the n = 0 reference projection
    normalizes the incident amplitude to unity. Returns dict with r, t
    (complex sideband projections), the channel frequencies and the
    reference amplitude."""
    nz = int(round((z_max - z_min) / h)) + 1
    z = np.linspace(z_min, z_max, nz)
    inside = (z > 0.0) & (z < d)
    inside_idx = np.where(inside)[0]
    n_in = int(inside.sum())
    i_lp = int(np.argmin(np.abs(z - (-3.0))))
    i_rp = int(np.argmin(np.abs(z - 4.0)))
    dist = np.abs(z - 0.5 * d)         # distance from the slab centre
    sponge = sigma_max * np.clip(
        (dist - sponge_start) / sponge_width, 0.0, 1.0) ** 2
    g_src = np.exp(-0.5 * (z - (z_min + 1.5)) ** 2)

    def env(t):
        t = np.asarray(t, dtype=float)
        return np.where(t < ramp,
                        np.sin(0.5 * np.pi * t / ramp) ** 2, 1.0)

    def rhs_factory(with_slab):
        def rhs(t, y):
            E = y[:nz]
            B = y[nz:2 * nz]
            S = y[2 * nz:2 * nz + n_in]
            dE = -(np.roll(B, -1) - np.roll(B, 1)) / (2 * h)
            dB = -(np.roll(E, -1) - np.roll(E, 1)) / (2 * h)
            # boundaries: reflecting walls are harmless inside sponges
            dE[0] = dE[-1] = dB[0] = dB[-1] = 0.0
            dE -= sponge * E
            dB -= sponge * B
            src = env(t) * np.cos(w0 * t) * w0 * g_src
            dE = dE + src
            if with_slab:
                wp2_t = 1.0 + m * np.cos(Omega * t)
                psrc = np.zeros(nz)
                psrc[inside_idx] = wp2_t * S
                dE = dE - psrc
                dS = E[inside] - gamma * S
            else:
                dS = np.zeros(n_in)
            return np.concatenate([dE, dB, dS])
        return rhs

    n_idx, w = channel_frequencies(w0, Omega, N)
    if K_periods is None:
        # adaptive: keep the carrier's conjugate mirror (at -w0) at
        # least six Hann halfwidths from every channel
        mirror_min = float(np.min(np.abs(w + w0)))
        K_periods = max(16, 4 * int(np.ceil(6.0 * Omega / mirror_min)))
    T_period = 2 * np.pi / Omega
    t_probe0 = max(t_end - (K_periods + 0.5) * T_period, ramp + 1.0)
    # K is large enough that the Hann mainlobe (halfwidth ~ 2/KT)
    # separates every channel from the conjugate mirror of the strong
    # carrier (the mirror sits 2 w_n from channel n; K = 16 gives
    # mirror pollution at the 1e-4 relative level)
    t_eval = np.minimum(
        np.arange(t_probe0, t_end + 1e-9, T_period / 128.0), t_end)
    out = {}
    for tag, with_slab in (("ref", False), ("slab", True)):
        sol = solve_ivp(rhs_factory(with_slab), (0.0, t_end),
                        np.zeros(2 * nz + n_in), method="DOP853",
                        rtol=rtol, atol=atol, t_eval=t_eval)
        if not sol.success:
            raise RuntimeError("time-domain %s run failed: %s"
                               % (tag, sol.message))
        out[tag] = sol.y
    # by linearity the probe differences isolate the slab scattering
    refl = out["slab"][i_lp] - out["ref"][i_lp]
    tdiff_sig = out["slab"][i_rp] - out["ref"][i_rp]
    rB = project_sidebands(t_eval, refl, w, K_periods, T_period)
    ref_lp = project_sidebands(t_eval, out["ref"][i_lp], w,
                               K_periods, T_period)
    ref_rp = project_sidebands(t_eval, out["ref"][i_rp], w,
                               K_periods, T_period)
    i0 = int(np.argmin(np.abs(n_idx)))
    # normalize by the CLEAN rightgoing incident amplitude measured at
    # the RIGHT probe (the left probe sits near the source and reads
    # its backward wave and near field; vacuum is nondispersive so the
    # rightgoing amplitude is the same at both probes)
    amp = abs(ref_rp[i0])
    tB = project_sidebands(t_eval, tdiff_sig, w, K_periods, T_period) \
        + ref_rp   # transmitted = difference + incident reference
    return {"r": rB / amp, "t": tB / amp, "w": w, "n": n_idx,
            "ref_amp": amp, "t_grid": t_eval}


def project_sidebands(tt, signal, w, K_periods, T_period):
    """Project a real signal onto e^{-i w_n t} components over the last
    K periods of the window (trapezoid on the uniform probe grid)."""
    mask = tt >= tt[-1] - K_periods * T_period + 1e-9
    ts = tt[mask]
    fs = np.asarray(signal)[mask]
    # Hann window suppresses the strong-channel sidelobes that a
    # rectangular projection leaks into weak sidebands (~1e-2 level)
    W = np.hanning(ts.size)
    W_mean = float(np.mean(W))
    out = np.empty(len(w), dtype=complex)
    for i, wi in enumerate(w):
        integrand = fs * W * np.exp(1j * wi * ts)
        out[i] = np.trapezoid(integrand, ts)             / (K_periods * T_period * W_mean)
    return out
