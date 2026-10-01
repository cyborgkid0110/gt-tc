# Packet Structure Model — Design Spec

**Date:** 2026-05-31
**Status:** Approved for implementation

## Problem

The maintenance-phase energy model charges transmit/receive energy proportional
to the number of bits in a packet:

```
E_TX(n, d) = n · (E_elec + eps_amp · d^gamma)
E_RX(n)    = n · E_elec
```

The bit counts `n` are currently **arbitrary magic numbers**:

| Constant | Value | Meaning |
|----------|-------|---------|
| `m_pkt_s` | 20 bits | small / cluster-member data packet |
| `m_pkt_l` | 1000 bits | large / cluster-head aggregated packet |
| `m_bit` | 8 (hardcoded twice) | sensing/processing sample width |

These values are not derived from any packet definition, so the per-round
transceiving energy is not physically grounded.

## Goal

Replace the arbitrary bit counts with a **structured packet definition** based on
the application **payload fields** carried by each packet. Use **minimal but
realistic fields**, so each packet size is derived and justified rather than
guessed.

> **Scope of the structure:** only the **application payload** is modeled. PHY/MAC
> framing overhead (preamble, SFD, PHR, MAC header, FCS) is **not** included — the
> energy model counts payload bits only.

## Decisions (from brainstorming)

1. **Granularity:** structured payload fields, minimal set (not an exhaustive field list).
2. **Aggregation:** fixed aggregated packet — a CH fuses all members into **one**
   fixed-size packet (perfect aggregation), independent of cluster size.
3. **Data-packet payload:** node ID (2 B) + sensor value (2 B) = 4 B = 32 bits.
4. **Aggregated-packet payload:** a fixed **cluster summary** digest — cluster ID +
   member count + mean + min + max = 72 bits. Larger than a single data packet
   (it carries fused information from the whole cluster) but independent of N.
5. **Sensing sample:** bump the hardcoded ADC sample from 8 → 16 bits to match the
   16-bit sensor reading.
6. **No framing overhead:** PHY/MAC/FCS bits are excluded; packet size = payload only.

### Rationale for AGG > DATA

With framing overhead removed, the payload *is* the information content. A single
reading (32 bits) cannot represent a whole cluster, so the aggregated packet must
carry a richer digest. A fixed cluster summary (count + mean/min/max) is larger
than one reading yet stays constant regardless of cluster size, which preserves
the "fixed aggregated packet" decision and keeps `m_pkt_l` a single constant.

## Packet structure (payload only, all sizes in bits)

```
DATA packet (CM reading):
  node id        16   (2 B)
  sensor value   16   (2 B)
  -----------------------------
  total          32   (4 B)

AGG packet (CH cluster summary):
  cluster id     16   (2 B)
  member count    8   (1 B)
  mean           16   (2 B)
  min            16   (2 B)
  max            16   (2 B)
  -----------------------------
  total          72   (9 B)
```

`m_pkt_s` and `m_pkt_l` remain two distinct named quantities. The aggregated
payload can later be enriched (e.g. add variance) without touching call sites.

Sensing/processing ADC sample width: **16 bits** (replaces the hardcoded `8`).

## Architecture

The packet structure mirrors the existing radio-parameter pattern: the **payload
field breakdown is declared in `main.py`** (alongside `SNR`, `N0`, …) and the
**totals are derived in `NetworkModel.__init__`** (as `eps_amp` is derived from the
link-budget fields). The `NetworkModel` interface attribute names `m_pkt_s` and
`m_pkt_l` are **kept unchanged**, so no algorithm call site is touched.

### Data flow

```
main.py: payload field constants (bits)
   │  pass as params
   ▼
NetworkModel.__init__:
   m_pkt_s = data_payload   # 32  (data packet)
   m_pkt_l = agg_payload    # 72  (aggregated packet)
   sensor_sample_bits = 16
   │  read by
   ▼
calc_tx_cost / calc_node_cost / compute_maintenance_costs   (unchanged logic)
```

## Code changes

| File | Change |
|------|--------|
| `main.py` | Remove `M_PKT_S=20`, `M_PKT_L=1000`. Add the payload field block (data payload, aggregated payload, sensor sample) as documented named constants. Pass them into `NetworkModel(...)`. |
| `model.py` `__init__` | Accept the payload-field params (with sensible defaults). Derive `self.m_pkt_s`, `self.m_pkt_l`, `self.sensor_sample_bits`. |
| `model.py` `calc_node_cost` (≈ line 153) | `m_bit = 8` → `self.sensor_sample_bits`. |
| `model.py` `compute_maintenance_costs` (≈ line 242) | `m_bit = 8` → `self.sensor_sample_bits`. |
| algorithms | **none** — they continue to read `net.m_pkt_s`, `net.m_pkt_l`, and call `calc_tx_cost` / `calc_node_cost` / `compute_maintenance_costs`. |

### Proposed constant block (main.py)

```python
# ---- Packet structure (application payload only), all in bits ----
# Data packet payload (cluster-member reading)
DATA_PAYLOAD = 32          # node id 16 + sensor value 16  (4 B)
# Aggregated packet payload (cluster-head fixed cluster summary)
AGG_PAYLOAD  = 72          # cluster id 16 + count 8 + mean 16 + min 16 + max 16  (9 B)
# Sensing / processing
SENSOR_SAMPLE_BITS = 16    # 16-bit ADC sample
```

### Proposed derivation (model.py)

```python
self.m_pkt_s = params.get('data_payload', 32)            # data packet
self.m_pkt_l = params.get('agg_payload', 72)             # aggregated packet
self.sensor_sample_bits = params.get('sensor_sample_bits', 16)
```

## Scope / non-goals

- **No** PHY/MAC/FCS framing overhead — payload bits only.
- **No** change to routing/relay semantics, the link-budget radio model, or the
  energy equations — only the **bit counts** (`n`) that feed them.
- **No** scaling of the aggregated packet with cluster size — it is a fixed digest.
- **No** fragmentation logic.
- `e_elec`, `e_agg`, `eps_amp`, and all power bounds are unchanged.

## Expected effect

- Data-packet TX/RX bits: 20 → 32 (~1.6× more).
- Aggregated-packet bits: 1000 → 72 (~14× cheaper) and now ~2.25× the data packet
  (was 50×), a far more sensible ratio for perfect aggregation.
- Sensing sample: 8 → 16 bits (sensing/processing cost doubles; it is a small
  voltage-based term).

Net result: per-round energy is dominated by the per-bit electronics + amplifier
terms over realistic payloads, with a logical DATA:AGG ratio. Absolute lifetimes
will shift across all algorithms — expected and correct; benchmarking stays fair
because every algorithm draws from the same `NetworkModel` energy layer.

## Verification

1. Confirm derived sizes: `net.m_pkt_s == 32`, `net.m_pkt_l == 72`,
   `net.sensor_sample_bits == 16`.
2. Smoke-test all algorithms (3 rounds, `MPLBACKEND=Agg`, no plot windows):
   GT2, LEACH, GTFR, DIA, MIA, TCLE, EFTCG-1, EFTCG-2, FL-LEACH-PSO, SCA-LEVY,
   EE-TCM. (FC-CRA is excluded pending its separate unresolved issue.)
3. Confirm no `NameError` / `AttributeError` and `dead` counts behave sanely.
