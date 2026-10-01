# Transmission Power Model Migration Design

**Date:** 2026-05-30
**Scope:** Replace the current Friis-based amplifier energy model in `model.py` with the link-budget-derived `eps_amp` from Tudose et al. [28] Equations (7) and (8), as specified in `docs/model_2.md`.

---

## Problem

The current `calc_tx_cost` computes the amplifier term inline from receiver sensitivity (`pth`), wavelength, and a hardcoded `t_tx = 1e-6`:

```
m_bit * layer_depth * (e_elec + (4pi/wave)^2 * pth * t_tx * d^2)
```

This conflates multiple radio parameters into an opaque expression. The spec replaces this with a physically grounded link-budget formula where `eps_amp` is derived from SNR, noise figure, bandwidth, antenna gain, PA efficiency, and data rate.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | `model.py` + `main.py` only | All algorithms use `calc_tx_cost` / `calc_node_cost` — no per-algorithm changes needed |
| P_th | Derived from link budget: `SNR * NF * N0 * BW` | Replaces the old raw `pth` parameter |
| eps_amp | Precomputed once in `__init__` | All parameters are constants for the simulation |
| calc_rx_power | Remove | Unused (no callers in the codebase) |
| WAVE | Changes from 0.1224 to 0.125 | Spec's value for 2.4 GHz: c / 2.4e9 |

---

## Changes

### 1. `NetworkModel.__init__` — new parameters

**Remove:** `pth` as a raw input parameter.

**Add** (all passed from `main.py`):
- `snr` — required SNR (linear)
- `nf_rx` — receiver noise figure (linear)
- `n0` — noise power spectral density (W/Hz)
- `bw` — channel noise bandwidth (Hz)
- `wave` — wavelength (m) — already exists, value changes to 0.125
- `gamma` — path loss exponent
- `g_ant` — antenna gain (linear)
- `eta` — TX PA efficiency
- `r_bit` — channel data rate (bps)

**Derived** (computed once in `__init__`, stored as attributes):
- `p_th = snr * nf_rx * n0 * bw`
- `eps_amp = p_th * (4*pi/wave)^gamma / (g_ant * eta * r_bit)`

**Keep unchanged:** `p_min`, `p_max`, `p_step`, `hop_max`, `e_elec`, `e_agg`, `m_pkt_s`, `m_pkt_l`, `alpha`, `beta`, `mu`.

### 2. `calc_tx_cost(d, role, layer_depth=1)` — Eq. 7

Replace:
```python
m_bit * layer_depth * (e_elec + (4*pi/wave)**2 * pth * t_tx * d**2)
```

With:
```python
m_bit * layer_depth * (e_elec + eps_amp * d ** gamma)
```

Where `m_bit = m_pkt_s` if role is `'CM'`, else `m_pkt_l`.

### 3. `calc_comm_range(power)` — d_max formula

Replace:
```python
sqrt(power * wave**2 / (pth * 16 * pi**2))
```

With:
```python
(power * g_ant * eta / (p_th * (4*pi/wave)**gamma)) ** (1/gamma)
```

### 4. `calc_rx_power(p_tx, d)` — remove

Dead code (no callers). Remove entirely.

### 5. `main.py` — parameter values

Replace:
```python
WAVE = 0.1224
PTH = 7e-10
```

With:
```python
SNR = 10              # required SNR (linear, = 10 dB)
NF_RX = 6.31          # receiver noise figure (linear, = 8 dB)
N0 = 3.98e-21         # noise PSD (W/Hz), kT at 290K
BW = 3e6              # channel bandwidth (Hz)
WAVE = 0.125          # wavelength (m), c / 2.4 GHz
GAMMA = 2.0           # path loss exponent
G_ANT = 1.0           # antenna gain (linear, = 0 dBi)
ETA = 0.30            # TX PA efficiency
R_BIT = 250e3         # data rate (bps)
```

Update `NetworkModel(...)` constructor call to pass new params instead of `pth=PTH, wave=WAVE`.

---

## Files Modified

| File | Change |
|------|--------|
| `model.py` | New radio params in `__init__`, derive `p_th`/`eps_amp`, rewrite `calc_tx_cost`, rewrite `calc_comm_range`, remove `calc_rx_power` |
| `main.py` | Replace `PTH`/`WAVE` with full link-budget parameter set, update `NetworkModel(...)` call |

---

## Verification

After implementation, smoke-test with EE-TCM (3 rounds, no plots) to confirm the simulation runs and energy accounting behaves reasonably with the new eps_amp value.

---

## Reference

Tudose, D., Gheorghe, L., & Tapus, N. (2013). Radio transceiver consumption modeling for multi-hop wireless sensor networks. UPB Scientific Bulletin, Series C, 75(1), 17-26.
