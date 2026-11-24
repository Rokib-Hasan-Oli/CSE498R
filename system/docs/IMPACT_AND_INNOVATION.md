# How BanglaLPR can solve problems for Bangladesh, companies and industry

This guide turns the system (V2: 91.7-92.7 % whole plate exact, 96.8-98.7 % character accuracy, 20-42 FPS on this PC)
into concrete uses and new solution ideas. Facts about Bangladesh come from the deep-research report
[`research/Bangla LPR impact in Bangladesh.md`](research/Bangla%20LPR%20impact%20in%20Bangladesh.md). Every number there
links to its source, and its caveats (conflicting sources, search-snippet prices) apply here too. Numbers about our
system come from `eval/results/` and the web app. Items marked **proposal** are ideas, not deployments.

---

## 1. The national problem in one paragraph

Bangladesh tried to identify vehicles with RFID tags from 2012, and it did not work. At highway toll booths about 90 %
of RFID stickers fail, and at Jamuna Bridge only 15 % of vehicles are detected automatically (Prothom Alo, Mar 2025).
In 2026 the state moved to **cameras**:
- **DMP:** an AI/ANPR case system went live on 7 May 2026. It covered 19 intersections with about 1,500 cases and
  38,000 summonses by 30 June. DMP wants 500 camera points but calls the cameras too costly and has asked for cheaper
  plug-and-play units.
- **Highway Police:** 1,427 AI cameras are planned on the Dhaka-Chattogram highway.
- **BRTA:** front plates on motorcycles (about 5.1 M of 6.85 M registered vehicles) are mandatory from 15 Nov 2026.

No read accuracy of the DMP system has been published. That leaves room for a **low-cost, Bangla-accurate, honest
(abstaining) plate reader**, which is what this project builds.

## 2. Problems the system can solve (and two it cannot)

| User | Problem today (reported) | What BanglaLPR adds (proposal) | Needs |
|---|---|---|---|
| **DMP / Highway Police** | 19 of a planned 500 points; cameras too costly; unreadable or painted plates; wrong-owner notices | Cheap edge unit per approach (Pi 5 + Hailo class, see §4); every plate gets a status: only *verified* plates (95 % of plates, 95.3 % exact in our test) go to automatic notices, the rest to a person | BRTA register access |
| **Toll operators (RHD, Bridge Authority, expressways)** | Padma ETC ≈ 1 % of crossings; RHD ETC ≈ 6 % of bridge traffic (Feb 2025); manual booth 300-400 vehicles/h | Camera backup when the RFID tag fails; plate ↔ account match; an independent **vehicle count to audit** contractor revenue (about 80 % of Dhaka-Bhanga toll revenue went to the operator) | ETC accounts, operator cooperation |
| **Police: fake / stolen vehicles** | ~78,000 vehicles on fake plates (2025, one source); stolen-bike networks | **Watchlist alert** (in the web app now); "same plate in two places at once"; **class-letter check**: the series letter encodes vehicle type (e.g. অ = motorcycle ≤ 100 cc, ক = car ≤ 1,000 cc), so a truck letter on a car is suspicious even without a database | Hotlists; a human confirms every hit |
| **BPC / fuel stations** | 2026 per-vehicle caps (2 L bikes, 10 L cars) enforced with **paper receipts**; ride-share bikes checked by hand | A shared plate log at pump entry stops buying at a second station; automatic ride-share check | A data-sharing mandate (privacy concern, see §6) |
| **Chattogram port, land ports** | ~8,000 vehicles/day at Chattogram; digital gate pass already used | Bind each gate pass to the truck that actually enters; catch reused passes | Port pass system integration |
| **Parking (city, malls, hospitals, campuses)** | App-only pilots (DNCC 202 spots, no ANPR) | Ticketless entry/exit, overstay fines, bKash/Nagad payment | Notice + retention policy |
| **Ride-hailing, insurers, fleets** | Ownership and registration-age rules broken; no local fleet data | A **plate-verification API**: photo in, plate + status out | API contracts |
| Plateless battery rickshaws (4.5-6 M+) | No plate | **Nothing**: a camera cannot read a plate that does not exist | Registration policy |
| Wrong owner in the register | Notices go to previous owners | **Nothing**: the error is in BRTA's data | BRTA data quality |

## 3. New solution ideas that make this research more than "another YOLO + OCR"

### 3.1 Send text, not video: an edge-first design (how fast data moves)

Our web app already shows the idea on every photo. The event message an edge camera would send is
`{"cam":…, "ts":…, "plate":"ঢাকা মেট্রো-গ ১৭-৪১১৯", "en":"Dhaka Metro-Ga 17-4119", "st":"verified", "c":0.85}`.
It measured about **170 bytes, against 48 KB for the photo (99.66 % less data)**. The research report's bandwidth
table (assumption: 20,000 reads per camera per day) shows what that means per camera per month:

| What is sent | Data / month | Average rate | 4G cost / month (Aug 2026 prepaid packs) |
|---|---|---|---|
| Text events only | ~0.3 GB | ~1 kbps | one Tk 109 pack |
| Text + 10 KB plate crop | ~6.3 GB | ~19 kbps | Tk 69-217 |
| 1080p video stream (4 Mbps) | ~1,296 GB | 4 Mbps | Tk 14,100-44,700 |

**How fast an alert travels** (proposal, with the numbers that are known):
- Reading on the device: about 25-50 ms per photo on our PC (GPU / CPU). On a Pi 5 + Hailo it is not yet measured.
- Network: the median mobile latency in Bangladesh was 25 ms (Ookla, Jan 2024). A 170-byte MQTT message adds only a
  2-byte header (an HTTP request carries 800-1,000 bytes of headers).
- So a watchlist hit can reach a control room in **well under 1 second on 4G**. The research report's 5-second
  "read to alert" target is realistic. The weak link is coverage, not speed.

**Protocol (proposal):**
- MQTT 5, QoS 1, over a persistent TLS session.
- The text event goes first; the crop follows, referenced by event ID.
- A **durable outbox on the device** (SQLite, like the app's `lpr.db`) keeps events through power cuts (2-6 h a day)
  and network drops. Every event carries its capture time, so late events arrive marked as late instead of lost.

### 3.2 Starlink and other links: where each one fits

| Link | Use it for | Facts (research report) |
|---|---|---|
| **Dual-SIM 4G** (e.g. GP + Robi) | the default for every camera | 4G reaches ~95 % of the population, but BTRC's July 2025 drive test found **41-77 % of tested highway areas without 4G**, depending on the operator; two operators cover each other's gaps |
| **Starlink** | 4G dead zones (hill tracts, char and haor roads) and **shared sites** (one dish per toll plaza or highway node serving many cameras) | Licensed 29 Apr 2025, service from 20 May 2025; 4 local gateways (data routed inside Bangladesh, which helps with data-localisation rules); Mini kit Tk 24,900; business plans from Tk 4,600/month (official business tiers seen only in search snippets: check); median upload 5.4 Mbps in a crowdsourced sample, plenty for events and too little for video; **power 25-40 W**, several times a Pi-class reader |
| **LoRaWAN** | at most a ~20-byte binary alert / heartbeat on a private gateway | No frequency plan for Bangladesh on TTN; 51-222-byte payloads; a JPEG crop does not fit |
| **Fibre** | toll plazas, control buildings | ~142,000 km backbone; plaza backhaul not documented |
| 5G | not needed | an event stream needs ~20 kbps |

**The Starlink idea in one line (proposal):** a solar-powered toll-plaza or highway hub with one Starlink Mini serves
6-10 edge plate readers over local Ethernet/Wi-Fi, so remote roads get the same 1-second alerts as Dhaka. Video never
crosses the satellite; only 170-byte events and small crops do. The research found **no documented solar or
satellite-backhauled ALPR deployment in South Asia** (needs a careful literature check before claiming a first).

### 3.3 Solar edge unit (proposal)

A Raspberry Pi 5 + Hailo NPU unit draws about 7-8 W (measured by others with a Hailo HAT). With a camera + IR (~8 W)
and a router (~4 W), about 20 W in total, it needs **~200 W of panel and ~1.8 kWh of LiFePO4** for 3 sunless days in
Bangladesh. A Jetson unit needs about twice that, and adding Starlink more than doubles the site. (Sizing assumptions
are in the research report, section "Solar".) The hardware is roughly Tk 56,000-58,500 per reader before enclosure,
IR, router and power. DMP's pilot cost about USD 7,100 per intersection.

### 3.4 "Abstain when unsure": the trust feature

A confident wrong read harms people: wrong-owner notices, scam calls pretending to be DMP fines, and wrongful hotlist
detentions abroad. BanglaLPR V2 already separates **verified** plates (two readers agree: 95 % of plates, 95.3 % right)
from **review** plates (5 %), and the web app shows which readers agreed. Turning this into a calibrated guarantee
(e.g. "auto-accepted plates ≥ 99 % right") is step 5 of `IMPROVEMENT_GUIDE.md`. It is publishable (no plate-specific
conformal paper exists), and it also satisfies the data law's right to restrict fully automatic decisions.

### 3.5 Research directions no Bangla plate paper has yet (ranked in the research report)

1. **A Bangla multi-frame track benchmark** with track voting (our video mode does the voting already).
2. **Calibrated review routing** (conformal prediction) on top of the reader vote.
3. **Grammar-constrained decoding** over the closed BRTA plate grammar.
4. A confidence-gated **small vision-language model** fallback.
5. The first dataset of the **new single-line motorcycle front plates** (from 15 Nov 2026).
6. Test-time adaptation of the readers for night; rendered synthetic plates for rare districts; plate vs vehicle-type
   consistency to catch fake plates.

A flagship paper could be: *"A Bangla multi-frame plate benchmark with calibrated abstention, measured on a
solar-capable Pi 5 + Hailo unit."*

## 4. For companies: products and business models (proposal)

| Product | Customer | Price anchor (research report) |
|---|---|---|
| Low-cost Bangla edge ANPR kit + annual maintenance contract | DMP, city corporations, Highway Police | DMP pilot ≈ USD 7,100 per intersection; DMP asks for cheaper units; maintenance is what earlier ITS projects lacked |
| Parking software with bKash / Nagad | malls, hospitals, campuses, housing | Plate Recognizer's ParkPow: USD 20 per camera per month |
| Bangla plate-reading API | ride-hailing, insurers, fleets, garages | Plate Recognizer: USD 50 per 50,000 lookups (no documented Bangla support) |
| Privacy-compliance module (retention, keyed hashing, audit log) | private operators | new duty under the 2026 data law |

The ANPR market is about USD 4.1-4.5 bn (2025-26) and grows 9-10 % a year (publisher estimates). No vendor documents
Bangla-script accuracy. Our edge is **Bangla accuracy, low-cost edge hardware, data kept in the country, and honest
abstention**.

## 5. A 3-phase pilot plan (proposal)

| Phase | Where | What is measured | Done when |
|---|---|---|---|
| 1. Campus gate (2-4 weeks) | NSU gate, one camera, the web app on a laptop | real video track accuracy (Improvement step 1), night accuracy with IR, review rate | 1,000 labelled tracks; accuracy and review rate published |
| 2. Edge unit (1-2 months) | Pi 5 + Hailo in an IP66 box at a parking entry, 4G + outbox | FPS on the board, power, uptime through power cuts, alert latency | 30 days unattended; latency log |
| 3. Partner site (3+ months) | a toll plaza or a low-traffic tunnel (Karnaphuli: 4,000-4,500 vehicles/day) | count audit vs operator numbers; watchlist demo; optional Starlink hub | a written evaluation with the operator |

## 6. Law and privacy: build it in from day one

- Plate + time + place is **personal data** under the Personal Data Protection Act 2026 (location data and identifiers
  count). Private operators need notice (signage), a lawful basis, a stated retention period and a way to handle
  requests. Police use falls under the broad Section 24 exemptions. Check the gazette text and commencement date before
  making compliance claims; sources conflict on penalties.
- Data classed confidential/restricted, and data of critical-infrastructure bodies, must stay in Bangladesh. **Edge
  processing with storage on an in-country server avoids the problem**, whatever link carries the data (Starlink's
  local gateways help).
- Suggested defaults (from EU/UK/US practice in the report):
  - Reads with no hit are deleted within **72 h** for enforcement and **30 days** for parking.
  - Hits are kept until the case closes.
  - Every access is logged.
  - Counting uses **keyed hashes** (HMAC with rotating keys). The plate space (~2 billion combinations) is small enough
    to brute-force an unkeyed hash.
  - A human confirms every hit before any notice is sent.
- The web app already has the basics: all data stays in `webapp/data/` on this PC, there is an **Apply retention**
  button (delete results older than N days), a full delete, and CSV export.
