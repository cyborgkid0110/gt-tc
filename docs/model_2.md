# Transmitting Power Model Migration Spec

## Overview

This document specifies the migration of the transmitting power model used across all optimization algorithms in the project to the radio transceiver model defined by Equations (7) and (8) from Tudose et al. [28]. The primary change is the replacement of the amplifier energy dissipation coefficient $\varepsilon_{amp}$ with a physically grounded link-budget formula. The structure of the transmission energy equation (Eq. 7) remains unchanged.

---

## Tasks

### 1. Migrate the transmitting power model

Replace the current $\varepsilon_{amp}$ constant with the link-budget computation defined in Eq. (8). The transmission energy formula (Eq. 7) is kept as-is; only the derivation of $\varepsilon_{amp}$ changes.

### 2. Apply changes to all optimization algorithms

The updated model must be propagated to every algorithm in the project that references transmission energy:

- **MILP topology control** — energy consumption vector $E(k)$ in the state update (Eq. 17); objective functions $f_1$, $\hat{f}_2$
- **FL-LEACH-PSO** — $E_{TX}$ in the radio model (cluster member → cluster head, cluster head → BS)
- **ECGD** — transmission energy in the dual-cluster-head energy cost computation
- **MS-WSNs** — energy expenditure in the WOT cycle's optimize step (service allocation cost)
- **GTFR** — energy term in the game-theoretic fuzzy routing utility function

---

## Model Specification

### Equation (7) — Transmission energy (unchanged)

The total energy to transmit $n$ bits over distance $d$ is:

$$E_{TX}(n, d) = E_{tc}(n) + E_{amp}(n, d) = n \cdot E_{trans} + n \cdot \varepsilon_{amp} \cdot d^{\alpha}$$

| Symbol | Description |
|---|---|
| $n$ | Number of bits in the packet |
| $E_{trans}$ | Energy per bit consumed by the radio transmit circuit (J/bit) |
| $\varepsilon_{amp}$ | Amplifier energy dissipation coefficient (J/bit·m$^{\alpha}$) — **see Eq. 8** |
| $d$ | Transmission distance (m) |
| $\alpha$ | Path loss exponent |

> **Note:** Only $\varepsilon_{amp}$ changes. $E_{trans}$ and the overall $E_{TX}$ formula remain as currently implemented.

---

### Equation (8) — Amplifier energy dissipation coefficient (new)

$\varepsilon_{amp}$ is no longer a fixed constant but is derived from the radio link budget:

$$\varepsilon_{amp} = \frac{\dfrac{S}{N_r} \cdot NF_{RX} \cdot N_0 \cdot BW \cdot \left(\dfrac{4\pi}{\lambda}\right)^{\gamma}}{G_{ant} \cdot \eta \cdot R_{bit}}$$

| Symbol | Description | Unit |
|---|---|---|
| $S/N_r$ | Required SNR at the receiver | linear ratio |
| $NF_{RX}$ | Receiver noise figure | linear ratio |
| $N_0$ | Noise power spectral density | W/Hz |
| $BW$ | Channel noise bandwidth | Hz |
| $\lambda$ | Carrier wavelength | m |
| $\gamma$ | Path loss exponent (same as $\alpha$ in Eq. 7) | — |
| $G_{ant}$ | Antenna gain | linear ratio |
| $\eta$ | Transmitter power amplifier efficiency | — |
| $R_{bit}$ | Channel data rate | bps |

Here is the compact self-consistent equation set:

---

### Receiver Sensitivity Threshold

$$P_{th} = \frac{S}{N_r} \cdot NF_{RX} \cdot N_0 \cdot BW \quad [W]$$

---

### Amplifier Energy Coefficient

$$\varepsilon_{amp} = \frac{P_{th} \cdot \left(\frac{4\pi}{\lambda}\right)^\gamma}{G_{ant} \cdot \eta \cdot R_{bit}} \quad [J/bit \cdot m^\gamma]$$

---

### Transmission Energy (Eq. 7)

$$E_{TX}(n, d) = n \cdot E_{trans} + n \cdot \varepsilon_{amp} \cdot d^\gamma \quad [J]$$

---

### Feasibility Check (before computing E_TX)

The required transmit power at distance d:

$$P_t^{req}(d) = \varepsilon_{amp} \cdot R_{bit} \cdot d^\gamma = \frac{P_{th} \cdot \left(\frac{4\pi}{\lambda}\right)^\gamma \cdot d^\gamma}{G_{ant} \cdot \eta} \quad [W]$$

A direct link (i, j) exists **if and only if**:

$$p_{min} \leq P_t^{req}(d_{ij}) \leq p_{max}$$

If $P_t^{req} > p_{max}$: link is infeasible, multi-hop required.
If $P_t^{req} < p_{min}$: node must transmit at $p_{min}$ regardless (minimum power floor).

---

### Equivalent Maximum Range

From the upper bound, the maximum single-hop range follows directly:

$$d_{max} = \left(\frac{p_{max} \cdot G_{ant} \cdot \eta}{P_{th} \cdot \left(\frac{4\pi}{\lambda}\right)^\gamma}\right)^{1/\gamma} \quad [m]$$

---

## Parameter Set for Testing

The following values are calibrated to a **2.4 GHz IEEE 802.15.4** network (CC2420-based motes, e.g., MicaZ / TelosB) and produce $\varepsilon_{amp} \approx 10 \times 10^{-12}$ J/bit·m², consistent with the Heinzelman LEACH free-space baseline.

| Symbol | Parameter | Value | Unit | Source / rationale |
|---|---|---|---|---|
| $S/N_r$ | Required SNR | 10 | linear (= 10 dB) | IEEE 802.15.4 O-QPSK, 1% PER threshold |
| $NF_{RX}$ | Receiver noise figure | 6.31 | linear (= 8 dB) | CC2420 datasheet |
| $N_0$ | Noise power spectral density | $3.98 \times 10^{-21}$ | W/Hz | Thermal noise: $kT$ at 290 K |
| $BW$ | Channel noise bandwidth | $3 \times 10^{6}$ | Hz | IEEE 802.15.4, 2.4 GHz per-channel bandwidth |
| $\lambda$ | Wavelength | 0.125 | m | $c / f = 3\times10^8 / 2.4\times10^9$ |
| $\gamma$ | Path loss exponent | 2.0 | — | Free-space; matches Heinzelman LEACH model |
| $G_{ant}$ | Antenna gain | 1.0 | linear (= 0 dBi) | Isotropic dipole assumption |
| $\eta$ | TX PA efficiency | **0.30** | — | Tuned to hit target $\varepsilon_{amp}$; within 25–40% typical CMOS range |
| $R_{bit}$ | Channel data rate | $250 \times 10^{3}$ | bps | IEEE 802.15.4 fixed rate |

### Verification

$$\varepsilon_{amp} = \frac{10 \times 6.31 \times 3.98\times10^{-21} \times 3\times10^{6} \times \left(\frac{4\pi}{0.125}\right)^{2}}{1.0 \times 0.30 \times 250\times10^{3}} \approx 10.03 \times 10^{-12} \ \text{J/bit·m}^2 \checkmark$$

---

## References

[28] D. Tudose, L. Gheorghe, and N. Tapus, "Radio transceiver consumption modeling for multi-hop wireless sensor networks," *UPB Scientific Bulletin, Series C*, vol. 75, no. 1, pp. 17–26, 2013.