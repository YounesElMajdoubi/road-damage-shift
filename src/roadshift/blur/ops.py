"""Flou de bougé synthétique, bruit capteur et défloutage de Wiener (numpy/scipy, sans OpenCV).

- `motion_kernel` : segment de longueur L et d'angle θ, normalisé (somme 1).
- `blur_image` : convolution par FFT avec bords en miroir, puis bruit gaussien, retour en uint8.
- `wiener_deconv` : filtre de Wiener par canal, PSF connue, constante K = rapport bruit/signal.
- `image_rng` : générateur seedé par le nom de l'image, pour que chaque image ait toujours le même flou.
"""

from __future__ import annotations

import zlib

import numpy as np
from scipy.signal import fftconvolve


def motion_kernel(length: int, angle_deg: float, oversample: int = 10) -> np.ndarray:
    """Noyau de flou de bougé linéaire. length <= 1 : identité (noyau 1x1)."""
    if length <= 1:
        return np.ones((1, 1))
    size = length if length % 2 == 1 else length + 1
    k = np.zeros((size, size))
    c = (size - 1) / 2
    t = np.linspace(-(length - 1) / 2, (length - 1) / 2, length * oversample)
    a = np.deg2rad(angle_deg)
    xs = np.clip(np.rint(c + t * np.cos(a)).astype(int), 0, size - 1)
    ys = np.clip(np.rint(c - t * np.sin(a)).astype(int), 0, size - 1)  # y vers le bas dans une image
    np.add.at(k, (ys, xs), 1.0)
    return k / k.sum()


def _conv_reflect(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    ph, pw = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(channel, ((ph, ph), (pw, pw)), mode="reflect")
    return fftconvolve(padded, kernel, mode="valid")


def blur_image(img: np.ndarray, kernel: np.ndarray, noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    """img uint8 HxWxC -> uint8 flouté puis bruité (sigma en fraction de 255)."""
    x = img.astype(np.float64)
    if kernel.shape != (1, 1):
        x = np.stack([_conv_reflect(x[..., c], kernel) for c in range(x.shape[2])], axis=-1)
    if noise_sigma > 0:
        x = x + rng.normal(0.0, noise_sigma * 255.0, x.shape)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def _psf2otf(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """FFT du noyau, centré sur l'origine (comme psf2otf de MATLAB)."""
    pad = np.zeros(shape)
    kh, kw = kernel.shape
    pad[:kh, :kw] = kernel
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.fft2(pad)


def wiener_deconv(img: np.ndarray, kernel: np.ndarray, k: float) -> np.ndarray:
    """Défloutage de Wiener à PSF connue. Bords en miroir pour limiter les artefacts de la FFT circulaire."""
    if kernel.shape == (1, 1):
        return img.copy()
    p = max(kernel.shape)
    x = np.pad(img.astype(np.float64), ((p, p), (p, p), (0, 0)), mode="reflect")
    h = _psf2otf(kernel, x.shape[:2])
    filt = np.conj(h) / (np.abs(h) ** 2 + k)
    out = np.stack([np.real(np.fft.ifft2(np.fft.fft2(x[..., c]) * filt)) for c in range(x.shape[2])], axis=-1)
    return np.clip(np.rint(out[p:-p, p:-p]), 0, 255).astype(np.uint8)


def image_rng(stem: str, seed: int) -> np.random.Generator:
    """Générateur reproductible propre à une image (indépendant de l'ordre de traitement)."""
    return np.random.default_rng([seed, zlib.crc32(stem.encode())])


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
