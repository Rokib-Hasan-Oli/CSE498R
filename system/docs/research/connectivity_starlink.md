# Connectivity for edge ALPR cameras in Bangladesh: Starlink, mobile, fixed, LoRaWAN, protocols and bandwidth math

Research date: 31 Oct 2026. Every figure carries the date of the source or the data period. Notes on how I got the data: the Opensignal report pages and the official starlink.com Bangladesh pages could not be fetched (403 / JavaScript-only), so their figures come from search-engine snippets and are marked "(snippet)". Wherever Bangladesh data is missing and I used a global example, it is labelled **[GLOBAL]**. My own arithmetic is in the "Inferences" sections only.

---

## 1. Starlink in Bangladesh (launch, licence, plans, prices, hardware, measured speed/latency, business/government use) and other satellite options

### Takeaway
Starlink has a 10-year BTRC NGSO licence (29 Apr 2025) and launched commercially on 20 May 2025. Residential service costs BDT 4,200 or 6,000 a month. The business "Local Priority" tier starts at BDT 4,600 a month for 50 GB, and a kit costs about BDT 25-47k. Latency in Bangladesh is low (35 ms Ookla median, Q4 2025), but upload is modest (5.4 Mbps crowdsourced median, Jul 2026). That suits an event-only ALPR link well and live video poorly. I found no evidence that OneWeb/Eutelsat is licensed in Bangladesh. The national GEO satellite (Bangabandhu/Bangladesh Satellite-1, run by BSCL) sells VSAT, but I found no published tariff.

### Cited Findings
**Licence and launch timeline**
- 25 Mar 2025: Chief Adviser Yunus told authorities to ensure Starlink's commercial launch within 90 days. 7 Apr 2025: Starlink applied under BTRC's "Guidelines for Non-Geostationary Orbit (NGSO) Satellite Services Operators in Bangladesh". 21 Apr: BTRC agreed in principle. 28 Apr: the Chief Adviser approved. Operations started Tuesday 20 May 2025. Bangladesh was the second South Asian country with Starlink, after Sri Lanka — [TBS/UNB, 20 May 2025](https://www.tbsnews.net/bangladesh/starlink-ca-yunus-greets-all-involved-1147181)
- On 29 Apr 2025 BTRC handed over two licences: an NGSO Operator Licence (Licensing Division) and a Radio Communication Apparatus Licence (Spectrum Division; the same article also calls it the "Radio Communication Operators Licence"). Both are valid for 10 years — [TBS, 29 Apr 2025](https://www.tbsnews.net/node/1129576)
- Starlink also needed tariff approval before launch. The NGSO policy requires use of a local broadband gateway / International Internet Gateway (IIG) — (snippet) [The Daily Star, Apr 2025](https://www.thedailystar.net/business/news/starlink-gets-satellite-internet-licence-btrc-3883181); [Dhaka Tribune](https://www.dhakatribune.com/bangladesh/379992/btrc-issues-licenses-to-starlink-to-operate-in)
- Licence 29 Apr 2025 and commercial launch 20 May 2025 are confirmed by [The Daily Star, 14 Apr 2026](https://www.thedailystar.net/business/economy/news/starlink-bangladesh-blazing-fast-few-can-afford-4151116)

**Ground infrastructure and regulatory frictions (relevant to government/police use)**
- Four local gateways were inspected by BTRC: two in the Hi-Tech Park at Kaliakair, Gazipur (run by Felicity IDC, with extra connectivity from Fiber@Home), one in Rajshahi (Bondstein Technologies, linked to Kaliakair over Summit Communications fibre) and one in Jashore (FIDC). BTRC "could not verify whether the systems were fully functional" — [The Daily Star, 9 Sep 2025](https://www.thedailystar.net/business/news/starlink-establishes-four-local-gateways-3982116)
- More gateways are planned at Syedpur, Cox's Bazar, Sylhet and Comilla. Starlink had 80 Gbps of IIG bandwidth and was using 30 Gbps. The lawful-interception/monitoring tool Starlink gave the NTMC was "not working as expected". BTRC also said foreign roaming Starlink users in Bangladesh were not being routed through local ground stations as the guidelines require. A request to use IPLC/unfiltered IP to serve neighbouring countries is not covered by the guidelines — [Prothom Alo, 2 Nov 2025](https://en.prothomalo.com/bangladesh/hb1rntaehr)
- Starlink has proposed exporting bandwidth from its local infrastructure, and BTRC is reviewing the proposal — [The Daily Star, 14 Apr 2026](https://www.thedailystar.net/business/economy/news/starlink-bangladesh-blazing-fast-few-can-afford-4151116)

**Plans and monthly prices (BDT)**
- Residential BDT 6,000/month and Residential Lite BDT 4,200/month at launch. Billing starts at activation or 30 days after delivery. Standard Kit BDT 47,000 plus BDT 2,800 shipping, total BDT 49,800 — [TelecomTalk, 21 May 2025](https://telecomtalk.info/starlink-launches-in-bangladesh-shows-what-it/994596/)
- As listed on Starlink's Bangladesh site per [Ajker Patrika, 6 Jul 2026](https://www.ajkerpatrika.com/en/technology/ajpmhyhwwvpns): Residential Tk 6,000/month, Residential Lite Tk 4,200, Business "starting at Tk 4,600/month", Maritime Tk 32,000/month, setup equipment Tk 47,000 one-time.
- Business "Local Priority" tiers (snippet, from the official starlink.com/en-bd/business/service-plans page, which could not be fetched; undated): BDT 4,600 for 50 GB, 11,800 for 500 GB, 20,800 for 1 TB and 38,800 for 2 TB. Add-ons are BDT 1,800 per +50 GB and BDT 9,000 per +500 GB. Includes network priority, a public IP and a dashboard. "Global Priority" starts at BDT 32,000/month for 50 GB — [Starlink BD business plans](https://starlink.com/en-bd/business/service-plans); a Ryans 2026 listing matches the 4,600 and 20,800 figures (snippet) — [Ryans](https://www.ryans.com/brand/starlink). The 4,600 figure matches Ajker Patrika's "Business from Tk 4,600" above.
- The Bangladesh plan is about US$50/month, which the article says is up to 10x the price of local broadband. Felicity IDC sells packages with 1-4 TB for Tk 25,000-75,000 — [The Daily Star, 14 Apr 2026](https://www.thedailystar.net/business/economy/news/starlink-bangladesh-blazing-fast-few-can-afford-4151116)
- Fixed terrestrial broadband is cited at 5 Mbps for as little as BDT 500/month (snippet; undated, from a 2025 Starlink pre-launch article) — [Financial Express](https://thefinancialexpress.com.bd/trade/starlink-set-to-launch-in-bd-but-some-hurdles-remain)

**Hardware (retail, Bangladesh)**
- Star Tech (fetched 31 Oct 2026): Starlink Mini Kit BDT 24,900 (list 26,500), Standard Kit BDT 46,900 (list 49,500), Flat High Performance "upcoming". Gen 3 router BDT 15,000, mounts BDT 3,100-8,300 — [Star Tech](https://www.startech.com.bd/starlink)
- A High-Performance kit is quoted at BDT 185,000 by retailers (snippet) — [Star Tech](https://www.startech.com.bd/starlink)

**Resellers / business and government channel**
- Robi Axiata, Felicity IDC and the state-owned Bangladesh Satellite Company Ltd (BSCL) each signed US$2.5 million distribution agreements focused on Business/Priority plans — [The Daily Star, 14 Apr 2026](https://www.thedailystar.net/business/economy/news/starlink-bangladesh-blazing-fast-few-can-afford-4151116)
- Robi markets "Local Priority" and "Global Priority" for fixed and portable use. Resellers get 15% off hardware and 5% off service, plus portal/API access, and set their own prices. Corporate and government clients go through resellers, while individuals can still buy on starlink.com. Target customers include banks, hospitals, factories, data centres, aviation and maritime — [The Daily Star, 9 Sep 2025](https://www.thedailystar.net/business/news/starlink-establishes-four-local-gateways-3982116)
- B-Trac Solutions signed on 26 Jan 2026 as a sales agent for BSCL's Starlink service (snippet) — [LinkedIn](https://bd.linkedin.com/in/sajjat-mahmud-b600487)
- I found **no explicit rule barring business or government use**. The only restrictions found are the local-gateway/IIG and lawful-interception requirements above.

**Subscribers (BTRC data)**
- 1,862 customers on 30 Sep 2025, of which 1,251 residential — [Prothom Alo, 2 Nov 2025](https://en.prothomalo.com/bangladesh/hb1rntaehr)
- 3,469 customers in Jan 2026 — [The Daily Star, 14 Apr 2026](https://www.thedailystar.net/business/economy/news/starlink-bangladesh-blazing-fast-few-can-afford-4151116)
- 5,389 customers in May 2026, "most of them corporate". BSCL says all its kit customers are corporate — [Ajker Patrika, 6 Jul 2026](https://www.ajkerpatrika.com/en/technology/ajpmhyhwwvpns)

**Measured performance in Bangladesh**
- Ookla, Q4 2025: median download 88.95 Mbps and median response time 35 ms, level with New Zealand (35 ms) and Australia (36 ms). No upload figure given. Sri Lanka in the same period: 140.68 Mbps and 103 ms — [The Daily Star, 14 Apr 2026](https://www.thedailystar.net/business/economy/news/starlink-bangladesh-blazing-fast-few-can-afford-4151116)
- Crowdsourced (tnetweather), 1-31 Jul 2026, 1,684 accepted measurements: median download 27 Mbps (P10 2.3, P90 94.1 Mbps), median upload 5.4 Mbps, median minimum RTT 64 ms. Methodology is only loosely described — [tnetweather](https://tnetweather.com/starlink-coverage/bangladesh/). **This conflicts with Ookla's 88.95 Mbps.** The datasets differ in period and method.
- Train trial: BSCL and Bangladesh Railway started on 13 Mar 2026 on three intercity trains (Parjatak, Upaban, Bonolota Express), free to passengers — [Meatechwatch, 16 Mar 2026](https://meatechwatch.com/2026/03/16/starlink-internet-launched-on-bangladesh-trains-in-trial-deployment/). An unverified blog reported 8-88 Mbps down, 13-47 Mbps up and 23-47 ms latency on a moving train — [Basenor](https://www.basenor.com/blogs/news/starlink-goes-live-on-bangladesh-trains-speeds-routes-whats-next) (mainstream outlets did not confirm these numbers).
- Pre-launch demo at the Bangladesh Investment Summit (Apr 2025): about 170 Mbps average with 70+ users on one dish, routed via a Malaysian IIG — (snippet) [The Daily Star](https://www.thedailystar.net/tech-startup/news/users-test-out-starlink-internet-speed-bangladesh-investment-summit-3867141)

**Power draw (matters for solar roadside sites)**
- Starlink Mini averages 25-40 W. Peak figures vary (up to about 60-75 W at start-up or in snow-melt mode, per secondary sources), and a measured peak was about 40 W. It can run from USB-C PD 100 W. The standard dish draws about 50-75 W or 75-100 W depending on generation — [CDW](https://www.cdw.com/product/starlink-mini-kit/8388748); [Panbo](https://panbo.com/starlink-mini-is-available-but-with-limited-data-options/); [Wireless Moves test](https://blog.wirelessmoves.com/?p=13403)
- **[GLOBAL]** A US integrator puts a Starlink dish at about 50-75 W continuous and an AI camera at 5-15 W, so the dish dominates the solar budget — [Carthage Electronics](https://carthageelectronics.com/?p=3139). A Japanese Starlink+solar camera unit uses a 5,000 Wh battery and a 600 W panel and says Starlink draws 30-50 W — [Only Style / IPROS](https://pr.www.ipros.com/en/onlystyle/product/detail/2001489038)

**Other satellite options**
- OneWeb/Eutelsat: my searches found no Bangladesh licence or service announcement. Results only covered India, where Eutelsat OneWeb and Jio's Orbit Connect hold GMPCS/IN-SPACe authorisations — [Inc42](https://inc42.com/buzz/oneweb-seeks-fast-track-nod-for-satellite-broadband-services-in-south-asia/)
- Bangabandhu/Bangladesh Satellite-1 (GEO, operated by BSCL): BSCL lists VSAT-based internet/intranet for business, government and remote areas — [BSCL services](https://bscl.gov.bd/?p=19). ATMs of Dutch-Bangla Bank (4) and Eastern Bank (5) were connected via its VSAT (older report) — [Financial Express](https://thefinancialexpress.com.bd/national/bangabandhu-satellite-to-connect-atms-for-secured-banking-services-1556371734). Its bandwidth has been a "hard sell" — [Dhaka Tribune](https://www.dhakatribune.com/amp/business/232208/bangabandhu-1%E2%80%99s-bandwidth-is-proving-to-be-a-hard). I found no public VSAT tariff.
- BSCL itself now resells Starlink and has connected 12 schools in three hill-tract districts (plan: 150+) — [The Daily Star, 14 Apr 2026](https://www.thedailystar.net/business/economy/news/starlink-bangladesh-blazing-fast-few-can-afford-4151116)

### Inferences
- For an event-only ALPR site, the cheapest Starlink option on paper is Residential Lite (BDT 4,200, "standard priority") or Local Priority 50 GB (BDT 4,600, with priority and a public IP). A government roadside deployment would most likely go through a reseller (BSCL, Robi or Felicity IDC) on Local Priority. Year 1 for one site ≈ Mini kit BDT 24,900 + 12 × 4,600 = **≈ BDT 80,100**, before mounts, power and labour.
- Starlink's latency in Bangladesh (35-64 ms) is in the same range as 4G. GEO VSAT on Bangladesh Satellite-1 has a physical minimum round trip of about 480 ms (2 × 2 × 35,786 km ÷ speed of light), which is still fine for alert messages measured in seconds.
- The median upload of 5.4 Mbps (Jul 2026) cannot carry even two 1080p H.264 streams at 4 Mbps. Starlink fits edge processing with event upload, not cloud video analytics.
- The power budget is the real constraint for off-grid solar sites. A Starlink Mini at 25-40 W draws about 5-15 times more than a 4G solar ANPR camera at 2.4-5.6 W (see Q6), so Starlink sites need much larger panels and batteries. One dish shared by several cameras, for example at a toll plaza or bridge approach, spreads that cost.

### Gaps
- No official, dated Starlink BD price page could be fetched. The Local Priority tiers come from search snippets of starlink.com plus one corroborating retailer listing and Ajker Patrika's "from Tk 4,600". They need confirming on starlink.com/bd before use in a report.
- No Bangladesh Starlink **upload** figure from Ookla. The only upload figures are crowdsourced (tnetweather) and an unverified train blog.
- No public BSCL VSAT tariff and no Bangladesh Satellite-1 latency measurement. No OneWeb/Eutelsat, Amazon Kuiper or other LEO licence found for Bangladesh. Absence in search results does not prove there is none.
- No published SLA or uptime figures for Starlink Business in Bangladesh, and no published monsoon/rain-fade performance data for Bangladesh.

---

## 2. Mobile networks: 4G coverage and measured speed/latency, 5G rollout (as of 2026), IoT/M2M SIMs, NB-IoT/LTE-M

### Takeaway
4G reaches about 95% of the population on paper. Median mobile download is about 40 Mbps (Ookla, Mar 2026), median upload is about 11-12 Mbps, and latency is about 25 ms (Ookla, Jan 2024). On highways, though, BTRC's July 2025 drive test found 41-77% of tested areas had no 4G, depending on operator. 5G is live only on GP and Robi (since 1 Sep 2025, divisional cities). Banglalink and Teletalk had not launched as of Aug 2026. Grameenphone is the only Bangladeshi NB-IoT operator on GSMA's list, and I found no LTE-M. BTRC is creating a separate "06" IoT/M2M SIM number series.

### Cited Findings
**Coverage**
- 4G covers "95% of the population", but only 55% of mobile users actually use 4G. The fibre backbone is about 142,000 km (37,000 km in 2015). 23% of industrial zones are fibre-connected. Smartphone penetration is 64% urban vs 31% rural. Power outages run 2-3 h/day in urban areas and 4-6 h/day in rural areas — [The Daily Star, 20 Jan 2026](https://www.thedailystar.net/news/bangladeshs-5g-moment-promise-challenges-and-the-road-ahead-4085521). The 95% figure traces back to [GSMA, 2021](https://www.gsma.com/mobilefordevelopment/resources/achieving-mobile-enabled-digital-inclusion-in-bangladesh/).
- Wikipedia claims 4G coverage of 100% (and 3G 99.80%) by 2025 without a primary source. Treat as unverified — [Wikipedia](https://en.wikipedia.org/wiki/Internet_in_Bangladesh)
- **BTRC drive test, Jul 2025, 574 km across Tangail, Bogura and Gaibandha.** Share of tested areas with no 4G: GP 58%, Banglalink 49%, Robi 41%, Teletalk 77%. Average downlink: GP 20.99 Mbps, Banglalink 29, Robi 21.6, Teletalk 10.35 Mbps. The 4G licences (issued 19 Feb 2018) required coverage of all upazilas, **national highways and railways** by 18 Feb 2023, and no operator has fully met this. The operators' body AMTOB disputes the method — [Dhaka Tribune, 4 Oct 2025](https://www.dhakatribune.com/bangladesh/393181/btrc-finds-gaps-in-4g-coverage-operators-fail-to)
- Opensignal 4G Availability (share of time on 4G), Feb 2025 report: Banglalink 90.4%, GP 90.2%, Robi 86.6%, Teletalk 75.1% (snippet) — [Opensignal Feb 2025](https://insights.opensignal.com/reports/2025/02/bangladesh/mobile-network-experience). In the Jan 2026 report (Oct-Dec 2025 data), GP, Airtel and Robi shared the 4G Availability award, and the Download Speed Experience winner was "just shy of 10 Mbps" (snippet) — [Opensignal Jan 2026](https://insights.opensignal.com/reports/2026/01/bangladesh/mobile-network-experience)

**Measured speeds/latency**
- Ookla, Jan 2024: mobile median 23.85 Mbps down, 11.36 Mbps up, **25 ms latency**, ranked 108th. Fixed: 41.59 down, 42.60 up, **5 ms**, ranked 109th — [The Daily Star, 27 Feb 2024](https://www.thedailystar.net/business/economy/news/bangladesh-slumps-global-mobile-internet-rankings-3553121)
- Ookla, Mar 2026: mobile median download 42.45 Mbps (about 90th) and fixed 66.02 Mbps. For comparison, India mobile 126.73, Sri Lanka 57.14 and Pakistan 28.27 Mbps — [Wikipedia list citing Speedtest, Mar 2026](https://en.wikipedia.org/wiki/List_of_countries_by_Internet_connection_speeds). Another table gives 38.51 Mbps and about 12 Mbps upload for May 2026 (snippet; conflicts slightly).
- An unnamed source reported a median of 31.15 Mbps down and 12.22 Mbps up in Jan 2026 (snippet) — [Meatechwatch, 24 Feb 2026](https://meatechwatch.com/2026/02/24/bangladeshs-4g-networks-face-a-reality-check-amid-new-qos-mandates/)
- Browser-based SpeedOf.Me (not Ookla), 2026: median latency 69 ms, mobile median 11 Mbps (methodology differs) — [SpeedOf.Me](https://speedof.me/internet-speed/bangladesh)
- No 2026 Ookla mobile latency figure found (speedtest.net is blocked to the fetch tool).

**5G status (as of the latest 2026 reports)**
- Robi and GP launched 5G commercially on 1 Sep 2025 in select metropolitan areas. GP claimed all divisional cities, but some Dhaka users reported no 5G signal. Robi targeted 200+ sites by end-2025 and 1,000 by end-2026 (snippet) — [The Daily Star](https://www.thedailystar.net/business/news/bangladesh-enters-5g-era-limited-rollout-3976221)
- As of 26 Aug 2026: GP has 5G in all divisional HQs and is deploying 700 MHz. Robi claims blanket coverage in many places at >100 Mbps. **Banglalink has not launched** and may do so if spectrum prices fall at the November 2026 renewal. Neither GP nor Robi discloses its 5G site count. 5G-capable handsets were 8.29% of handsets in Jul 2026 (Robi 10.06%, GP 7.53%, BL 7.29%), totalling 14.68 million — [The Daily Star, 26 Aug 2026](https://www.thedailystar.net/business/news/5g-ready-handsets-gain-ground-users-upgrade-4256911)
- Opensignal (Jan 2026): neither Banglalink nor Teletalk had announced 5G plans. Robi won 5G Download Speed and GP won 5G Video Experience (snippet) — [Opensignal Jan 2026](https://insights.opensignal.com/reports/2026/01/bangladesh/mobile-network-experience)

**IoT / M2M SIMs, NB-IoT / LTE-M**
- GP launched a corporate M2M plan in 2017 with industrial SIMs (heat, corrosion and vibration resistant), data packs and a self-service portal — [The Daily Star](https://online91.thedailystar.net/bytes/tech-happening/grameenphone-launches-m2m-service-1508335); [Future Startup, 18 Dec 2017](https://futurestartup.com/2017/12/18/grameenphone-launches-m2m-plan-internet-of-things-in-dhaka-and-grameenphones-tenacity/)
- GSMA's Mobile IoT launch list (updated Nov 2025) shows **Grameenphone, NB-IoT** as the only Bangladesh entry. No LTE-M is listed. Globally there are 129 LTE-M and 140 NB-IoT networks — [GSMA Mobile IoT launches](https://www.gsma.com/iot/mobile-iot-commercial-launches/)
- BTRC plan (4 Dec 2025): the "06" number series is reserved for IoT/M2M (061 Banglalink, 062 Teletalk, 063 Robi, 064 GP). IoT SIMs will have no voice, corporate/device-based registration, and recycling of inactive numbers after 3 months. BTRC will ask the NBR to exempt IoT SIMs from the Tk 300 SIM tax. Robi says IoT/M2M SIM use grew 20% in a year — [The Daily Star, 4 Dec 2025](https://www.thedailystar.net/business/news/btrc-introduce-separate-sim-series-iot-m2m-devices-4050931)

### Inferences
- In cities and on mapped 4G corridors, plain LTE (not NB-IoT) is the natural backhaul. About 25 ms latency and 11-12 Mbps median upload are far more than event traffic needs (≈ 20 kbps average; see Q5).
- On rural highways, the July 2025 drive test means any single operator can be missing for 41-77% of a route. A **dual-SIM / multi-operator router** (for example GP + Robi or Banglalink), with Starlink as a fallback at sites with no 4G, is a defensible design.
- 5G adds nothing essential for event-only ALPR and is limited to divisional cities. Its value is only for sites that must stream video.
- NB-IoT (GP only) can in principle carry text events. I found no Bangladesh-specific NB-IoT throughput, latency or price, so it cannot be recommended on evidence.

### Gaps
- No 2026 Ookla mobile latency. No per-operator 2026 Opensignal 4G figures (pages blocked). No coverage map data specific to national highways or toll plazas.
- No current published IoT/M2M SIM tariffs from GP, Robi or Banglalink, and no confirmation that the Tk 300 SIM-tax exemption or the 06 series has taken effect.
- No 5G site counts. Teletalk's 5G status is only implied (not launched).

---

## 3. Fixed connectivity near highways and toll plazas; LoRaWAN in Bangladesh and payload limits

### Takeaway
Bangladesh has a large fibre backbone (≈142,000 km, Jan 2026) and state fibre owned by BTCL, BCC, PGCB and Bangladesh Railway, which a 2025 "national fibre bank" plan aims to open up. I found no public data on fibre at specific toll plazas or highway camera sites. The Dhaka-Chattogram highway's 1,427-AI-camera plan does not say how the cameras are connected. LoRaWAN has no settled frequency plan for Bangladesh on TTN and only a nominal community presence. Its 51-222-byte payload limit can carry a compact binary plate event but not JSON plus a JPEG crop.

### Cited Findings
**Fibre / fixed broadband**
- Fibre backbone ≈142,000 km (vs 37,000 km in 2015) — [The Daily Star, 20 Jan 2026](https://www.thedailystar.net/news/bangladeshs-5g-moment-promise-challenges-and-the-road-ahead-4085521)
- BTRC: 135.94 million internet subscribers at end-June 2026, of which 120.84 million mobile and 15.10 million broadband (ISP+PSTN) (snippet) — [Financial Express](https://thefinancialexpress.com.bd/trade/internet-subscriber-growth-outpaces-mobile-connections-in-h1)
- A 2025 national fibre-bank plan would pool BTCL, BCC, PGCB and Railway fibre to allow 1 Gbps+ in every union. This is a proposal (snippet) — [Meatechwatch, 9 Jul 2025](https://meatechwatch.com/2025/07/09/bangladesh-to-establish-national-fibre-optic-bank-to-unlock-unused-state-owned-infrastructure/). BTCL had brought 1,200 unions onto fibre by 2017 (snippet) — [The Daily Star](https://www.thedailystar.net/business/two-firms-take-internet-2600-unions-tk-378cr-1448203)
- Only about 7.6% of Bangladeshi networks peer locally (BDIX is the largest IXP). SEA-ME-WE-6 is expected in 2026 — [APNIC blog, 3 Nov 2025](https://blog.apnic.net/2025/11/03/bangladeshs-internet-transformation-from-satellite-shadows-to-digital-highways/)
- Dhaka-Chattogram highway: 1,427 AI cameras planned at 490 points over about 250 km (Signboard to Chittagong City Gate). An AI "auto-fine" system launched on 18 Aug 2026 and sends SMS fine notices, with 449 cases and Tk 15.51 lakh in fines in its first 24 h. **Camera connectivity, control-room location and vendor are not stated** — [TBS/BSS, 11 Sep 2026](https://www.tbsnews.net/node/1539861)
- An unstated-date report says the whole route is under CCTV, with 16 points covered by AI cameras and central monitoring by Highway Police (snippet) — [Financial Express](https://thefinancialexpress.com.bd/national/ai-traffic-control-system-to-launch-on-dhaka-ctg-highway)

**LoRaWAN**
- TTN's by-country frequency table lists Bangladesh with **no frequency plan and no regulatory document** — [TTN frequencies by country](https://www.thethingsnetwork.org/wiki/LoRaWAN/Frequencies/By-Country). A TTN forum thread (latest activity Nov 2025) is still asking which plan Bangladesh should use (EU868 vs IN865 vs 866-868 MHz) — [TTN forum](https://www.thethingsnetwork.org/forum/t/ideal-frequency-plan-for-bangladesh/64505)
- A TTN Dhaka community page exists with only an introductory post — [TTN Dhaka](https://www.thethingsnetwork.org/community/dhaka/posts). BTRC issued a directive permitting IoT in nine sectors (smart building, industrial automation, smart grid, water, etc.; date not shown) — [The Daily Star](https://www.thedailystar.net/country/bangladesh-bd-telecommunication-regulatory-commission-btrc-allows-internet-of-things-iot-solutions-technology-1567198)
- Payload limits (EU863-870 regional parameters, the closest table found; IN865 not confirmed): DR0-2 = 51 bytes, DR3 = 115, DR4-7 = 222. The LoRaWAN header adds ≥13 bytes, and some libraries (LMiC) cap at 51 bytes for all data rates — [TTN forum: fair-use explained](https://www.thethingsnetwork.org/forum/t/fair-use-policy-explained/1300); [TTN EU868 parameters](https://www.thethingsnetwork.org/docs/lorawan/regional-parameters/eu868/)
- TTN fair use: an average of **30 s of uplink airtime per device per 24 h** and 10 downlinks/day. These limits do not apply on a private network — [TTN fair-use thread](https://www.thethingsnetwork.org/forum/t/the-things-network-fair-use-policy/47689)

### Inferences
- A plate event can fit LoRaWAN only if packed in binary. Bengali plate text in UTF-8 costs 3 bytes per character (roughly 50-60 bytes for "ঢাকা মেট্রো-গ ১২-৩৪৫৬"), which already exceeds the 51-byte limit at low data rates. Encoded as district index (1 B) + class letter (1 B) + 6 digits (3 B BCD) + timestamp (4 B) + lat/lon (8 B) + camera ID/confidence (2-3 B), it is about **19-20 bytes**. That fits even DR0. A 200-500 B JSON fits only DR4+ (222 B) or not at all. A 5-20 KB JPEG crop would need roughly 25-400 fragments and is impractical under the 30 s/day TTN airtime limit.
- LoRaWAN is therefore at most a **text-only alert or heartbeat fallback** on a private gateway. Its unclear Bangladesh frequency status makes it a regulatory risk.
- Toll plazas and expressway control buildings are likely to have fibre or ISP drops (the auto-fine system is centrally monitored). Pole-mounted mid-highway cameras will more likely rely on 4G, with Starlink where 4G is missing. This is an inference; no source confirms it.

### Gaps
- No public data on fibre presence at toll plazas (Padma Bridge, Dhaka Elevated Expressway, Dhaka-Mawa-Bhanga Expressway, Meghna/Gumti bridges) or along Roads & Highways Department corridors. No RHD/Highway Police tender documents found.
- No BTRC LoRa/LPWAN band allocation found. No IN865 payload table confirmed for Bangladesh. No commercial LoRaWAN operator found in Bangladesh.

---

## 4. Protocols for small event messages: MQTT vs HTTP vs WebSocket vs gRPC, store-and-forward, message sizes

### Takeaway
For roughly 0.5-20 KB events over cellular or satellite, MQTT over a persistent TLS connection has the smallest per-message overhead: a 2-byte fixed header, versus typically 800-1,000 bytes of HTTP request+response headers. It also has built-in QoS 1 delivery and broker-side session queues. Store-and-forward still needs a **durable on-device outbox**, because broker persistent sessions are bounded and, in EMQX's case, in-memory.

### Cited Findings
- MQTT minimum message overhead is 2 bytes, against an 8-byte minimum for HTTP header data, and HTTP repeats its overhead on every request — [HiveMQ](https://www.hivemq.com/article/mqtt-vs-http-protocols-in-iot-iiot)
- A typical HTTP request+response "can easily total 800-1,000 bytes of headers" even for a few-byte payload — [Kaa IoT](https://www.kaaiot.com/iot-knowledge-base/mqtt-vs-http-for-iot-device-communication-performance-comparison-and-best-practices)
- Worked example: publishing "HelloWorld" takes 24 bytes as an MQTT packet and 91 bytes as a minimal HTTP POST — [EMQX](https://www.emqx.com/en/blog/mqtt-vs-http)
- The 2-byte figure is the fixed header only. Topic, packet ID, QoS fields and MQTT 5 properties add more. Radio wake-up plus TCP/TLS setup often costs more energy and time than the payload itself — [DEV Community](https://dev.to/ranaweerasupun/mqtt-vs-http-for-iot-what-i-learned-using-both-in-production-1n5)
- WebSocket frames carry a 2-14-byte header, and client-to-server frames must be masked with a 4-byte key — [RFC 6455 §5.2](https://datatracker.ietf.org/doc/html/rfc6455#section-5.2) (protocol spec; not fetched in this session)
- gRPC runs on HTTP/2, which has a 9-byte frame header — [RFC 9113 §4.1](https://datatracker.ietf.org/doc/html/rfc9113#section-4.1). Each gRPC message adds a 5-byte length prefix — [gRPC over HTTP/2](https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md) (specs; not fetched in this session)
- EMQX persistent sessions (MQTT 3.1.1 Clean Session=0, or MQTT 5 Clean Start + Session Expiry) keep QoS 1/2 messages through outages, but EMQX v5's persistent session "is based on memory" — [EMQX docs](https://docs.emqx.com/en/emqx/v5.0/mqtt/mqtt-session-and-message-expiry.html). The EMQX Offline Messages plugin persists QoS 1/2 messages to MySQL/Redis and replays them — [EMQX offline messages](https://docs.emqx.com/en/emqx/latest/extensions/plugin-catalog/6.3/emqx-offline-messages.html). A user reported losing the oldest messages beyond the default 1,000-message session queue during a long outage — [EMQX forum](https://www.emqx.io/forum/t/how-to-detect-queue-overruns-for-persistent-sessions/179)
- **[GLOBAL product]** The Milesight 4G solar ANPR camera does ANPR at the edge, stores up to 20,000 allow/deny-list entries and up to 1 TB on microSD, and sends events by HTTP POST, FTP or SMTP. MQTT is listed only as a wake-up method — [Milesight](https://www.milesight.com/security/product/solar-powered-anpr-road-traffic-camera)
- Bangladesh resilience context: the July-August 2024 nationwide shutdown was Bangladesh's longest — [OONI, 2025](https://ooni.org/post/2025-bangladesh-report/). On 24 Dec 2025 the Council of Advisers approved the Bangladesh Telecommunication (Amendment) Ordinance 2025, which prohibits internet/telecom shutdowns (Section 97) and moves to dissolve the NTMC — [Meatechwatch, 25 Dec 2025](https://meatechwatch.com/2025/12/25/bangladesh-approves-telecommunication-ordinance-2025-guarantees-no-internet-shutdowns/)

### Inferences
- Per-event overhead comparison for a 400-byte JSON event on an already-open TLS connection: MQTT adds about 2 + topic (~20-40 B) + packet ID ≈ 25-45 B (about 10%). HTTP/1.1 adds about 800-1,000 B (2-2.5x the payload) unless kept alive and header-minimised. WebSocket adds 6-14 B per frame after a one-time HTTP upgrade. gRPC adds 14 B of framing plus HPACK-compressed headers. For 10-20 KB crops the differences become negligible (under 10%), so payload choices (crop size, JPEG quality) matter more than protocol.
- Recommended pattern: MQTT 5, QoS 1, persistent session, TLS. Send the text event (plate, time, GPS, confidence) first on a high-priority topic and the crop as a second message (or HTTP upload referenced by event ID). Keep a local SQLite/flash outbox on the edge device, because broker queues are bounded and do not cover the period before the device reconnects. This survives 4G drop-outs on rural highways, power cuts (2-6 h/day) and any future shutdown despite the 2025 ban.
- HTTP POST is acceptable when a vendor camera only supports it (as with Milesight). It costs more bytes per event and more round-trips if connections are not reused.

### Gaps
- No measured Bangladesh-network latency comparison of MQTT, HTTP, WebSocket and gRPC. No TLS handshake byte cost measured on Bangladeshi networks.
- WebSocket and gRPC overhead figures come from specs I did not fetch this session. They are standard values, but the report writer may want to cite the specs directly.

---

## 5. Bandwidth and cost math: streaming N cameras of 1080p video vs events only; data cost per GB in Bangladesh (2026); latency budget for a stolen-car alert

### Takeaway
One 1080p H.264 stream at a typical 2-8 Mbps is about 650-2,600 GB a month. Edge events (≈10 KB each with crop, 20,000 a day) are about 6.3 GB a month, roughly 200 times less, and text-only events are about 0.3 GB. At Bangladeshi prepaid prices (Tk 10.9-48 per GB, Aug 2026), events cost about Tk 70-300 a month per camera on 4G. Streaming would cost Tk 14,000-45,000 a month per camera and also exceeds typical uplink capacity. Network latency (25-65 ms) is a tiny part of an "alert within 5-10 s" budget.

### Cited Findings
- 1080p H.264 bitrates: 4-6 Mbps, with 6-8 Mbps at 30 fps (VXG). 1.5-2.5 Mbps at 15-30 fps (another guide). 3 Mbps recommended (AWS Rekognition, floor 1.5 Mbps). H.265 saves about 30-50% — [VXG](https://help.videoexpertsgroup.com/kb/recommended-bitrate-settings); [AWS Rekognition](https://docs.aws.amazon.com/rekognition/latest/dg/recommendations-camera-stored-streaming-video.html); [Fortinet sizing](https://docs.fortinet.com/document/fortirecorder/7.0.2/administration-guide/402417/sizing-guidelines) (1080p at 10 fps high quality = 2.4 Mbps)
- Retail mobile data prices (4 Aug 2026): Banglalink 1 GB/3 days Tk 17, 2 GB/7 days Tk 32, 5 GB/15 days Tk 65, 10 GB/30 days Tk 109. Grameenphone 1 GB/7 days Tk 48, 2 GB/7 days Tk 69 — [Daily Observer, 4 Aug 2026](https://observerbd.com/news/586298)
- BTRC rule (from 15 Oct 2025): no data packs under 7 days. 3-day and 15-day packs were discontinued, and packs are standardised around 7-day, 30-day and unlimited (snippet) — [bdnews24](https://bdnews24.com/technology/v36oy4p4sy); [Daily Observer, 4 Aug 2026](https://observerbd.com/news/586298). (The Observer still lists some Banglalink 3-day and 15-day packs, so the rule's application is inconsistent across sources.)
- Operators' own cost per GB was about Tk 4 on average (GP Tk 2.26, Robi Tk 5.28) against retail of Tk 25-35/GB. IIG bandwidth costs Tk 130-150 per Mbps. There is a Tk 300 tax per SIM and 20% supplementary duty — [Bonik Barta, 6 Sep 2024](https://en.bonikbarta.com/exclusive/Y1mr7cn0S586gKMz/)
- Cable.co.uk 2023: Bangladesh US$0.23 per GB, India 0.16, Pakistan 0.12, Sri Lanka 0.09 (snippet; no 2026 figure found) — [Dhaka Tribune](https://www.dhakatribune.com/business/297107/bangladesh-has-cheap-but-slow-mobile-data)
- Starlink BD prices and measured latency/upload: see Q1 (BDT 4,200-6,000 residential, Local Priority from 4,600/50 GB up to 38,800/2 TB; 35 ms Ookla Q4 2025; 5.4 Mbps upload median Jul 2026).
- Mobile latency 25 ms median (Ookla, Jan 2024); upload 11.36 Mbps (Jan 2024) to about 12 Mbps (2026) — see Q2.
- Alert latency benchmark **[GLOBAL]**: proponents of the UK National ANPR Data Centre say automated hotlist checks from mobile ANPR devices "typically" return results within about 5 seconds (a claim, not a NASPLE requirement) — [Queensland Parliament submission](https://documents.parliament.qld.gov.au/com/TC-C2D2/IANPR-93E1/submissions/00000026.pdf). Flock Safety only claims "instant" alerts against NCIC/hotlists, with no number — [Flock LPR](https://www.flocksafety.com/products/license-plate-readers)
- Bangladesh scale reference: 1,427 AI cameras are planned on the Dhaka-Chattogram highway — [TBS/BSS, 11 Sep 2026](https://www.tbsnews.net/node/1539861)

### Inferences (my arithmetic; assumptions stated)
**Per-event sizes** (from the brief): JSON 200-500 B; JPEG crop 5-20 KB. Per event: text-only ≈ 0.5 KB; text + 10 KB crop ≈ 10.5 KB; text + 20 KB crop ≈ 20.5 KB. Decimal units throughout (1 GB = 10^9 B).

**Assumed load:** 20,000 plate reads/day per camera (≈ 830/h average; an assumption, since no Bangladeshi per-lane traffic counts were researched here), 30-day month.

| Mode (1 camera) | Per day | Per month | Average bitrate |
|---|---|---|---|
| Events, text only (0.5 KB) | 10 MB | 0.3 GB | ≈ 0.9 kbps |
| Events + 10 KB crop | 210 MB | 6.3 GB | ≈ 19 kbps |
| Events + 20 KB crop | 410 MB | 12.3 GB | ≈ 38 kbps |
| 1080p H.264 @ 2 Mbps | 21.6 GB | 648 GB | 2 Mbps |
| 1080p H.264 @ 4 Mbps | 43.2 GB | 1,296 GB | 4 Mbps |
| 1080p H.264 @ 8 Mbps | 86.4 GB | 2,592 GB | 8 Mbps |

- Ratio: a 4 Mbps stream is about **206x** events with 10 KB crops (1,296 / 6.3) and about **4,300x** text-only events.
- N = 10 cameras: streaming at 4 Mbps needs a 40 Mbps sustained uplink and about 13 TB/month. Events with crops need about 63 GB/month at about 0.2 Mbps average.
- National scale (1,427 Dhaka-Chattogram cameras, if they were ALPR): streaming at 4 Mbps ≈ 5.7 Gbps sustained and ≈ 1.85 PB/month. Events with crops ≈ 9 TB/month.

**Monthly data cost per camera at Aug 2026 consumer prepaid prices** (Tk 10.9/GB Banglalink 30-day pack up to Tk 34.5/GB GP 7-day pack; IoT tariffs unknown):
- Text-only (0.3 GB): one minimum monthly pack (for example Banglalink 10 GB/30 days, Tk 109).
- Events + 10 KB crop (6.3 GB): Tk 69-217/month.
- Events + 20 KB crop (12.3 GB): Tk 134-424/month.
- Streaming at 4 Mbps (1,296 GB): Tk 14,100-44,700/month (in practice also blocked by fair-use limits and cell uplink capacity: a 4 Mbps sustained upload is about a third of the national median upload of 11-12 Mbps, on a shared cell).
- On Starlink: events fit Local Priority 50 GB (Tk 4,600) or Residential Lite (Tk 4,200). Streaming one 4 Mbps camera needs about 1.3 TB, which means the 2 TB tier at Tk 38,800. A 5.4 Mbps median upload fits only about one stream per dish.
- **Conclusion:** where 4G exists, edge processing over 4G costs roughly Tk 100-400 a month per camera in data. Starlink costs about Tk 4,200-4,600 a month per *site*, so it is worth it only where 4G is absent, or when one dish serves many cameras (a toll plaza or bridge). Cloud video streaming is not economical on either path.

**Latency budget: "stolen-car alert reaches police within X s"** (network figures from cited sources; the rest are design assumptions):
1. Capture + detection + OCR on the edge device: use the project's own measured inference time (not researched here).
2. Publish the text event over an already-open MQTT/TLS connection: about 1 RTT. 4G ≈ 25 ms median (Ookla Jan 2024). Starlink ≈ 35-64 ms (Ookla Q4 2025 / tnetweather Jul 2026). GEO VSAT ≥ ≈ 480 ms (physics). If the connection must be re-established, add about 2-3 RTTs for TCP+TLS, which is still under 0.3 s on 4G or LEO.
3. Crop upload (20 KB) at 1-5 Mbps upload: 0.03-0.16 s. Send it after the text event so it never delays the alert.
4. Server hotlist match: milliseconds for an in-memory set lookup.
5. Push to officer app / SMS / control-room screen: unmeasured for Bangladesh. The **dominant and unknown term**.
- The network contributes well under 1 s on 4G or Starlink, even with reconnects. A **≤ 5 s read-to-alert target**, matching the UK proponents' figure, is plausible network-wise. The real risks are coverage holes (41-77% no-4G on tested rural routes) and alert delivery, not bandwidth. With store-and-forward, an event recorded in a coverage hole arrives late rather than being lost. The system should report "event age" so stale hits are flagged.

### Gaps
- No published IoT/M2M SIM tariffs per GB in Bangladesh. The cost math uses consumer prepaid packs, which may not be legal or practical for fleet devices.
- No 2026 Cable.co.uk Bangladesh figure. No per-lane traffic volumes for Bangladeshi highways in this research. No measured SMS or push-notification latency in Bangladesh. No Bangladesh Police / Highway Police alert-time standard found.

---

## 6. Real deployments of satellite-backhauled or solar-powered ALPR/traffic cameras (worldwide / South Asia)

### Takeaway
Solar + LTE ALPR is mature and widespread in the US (Flock Safety, 6,000+ communities by Jul 2026). Several vendors sell 4G solar ANPR cameras that do recognition on the camera (Milesight, Hikvision). Starlink + solar camera units exist commercially. I found **no documented satellite-backhauled ALPR deployment, and no solar ALPR deployment in South Asia or Bangladesh**. Bangladesh's own highway AI cameras (Dhaka-Chattogram, Aug 2026) do not disclose their backhaul.

### Cited Findings
- **[GLOBAL] Flock Safety (US):** most devices run on solar panels with rechargeable batteries and send scan data to a central server over LTE. They are used in 4,000+ cities across 42 states (2024) and 6,000+ communities in all 50 states with 20+ billion vehicle scans a month (Flock claim, Jul 2026). Reads are matched against NCIC and state/local hotlists, and officers are alerted "instantly". Wrongful hotlist hits have led to detentions — [Wikipedia: Flock Safety](https://en.wikipedia.org/wiki/Flock_Safety); [Flock LPR page](https://www.flocksafety.com/products/license-plate-readers)
- **[GLOBAL product] Milesight 4G Solar-powered ANPR road-traffic camera:** 100 W PV panel; 555 Wh or 962 Wh battery giving 5 or 9 days backup in performance mode; camera draw 2.43-5.61 W (0.03 W standby); ANPR up to 120 km/h and 50 m over 2 lanes; on-camera allow/deny list of 20,000 entries; HTTP POST/FTP event upload; -40 to 60 °C — [Milesight](https://www.milesight.com/security/product/solar-powered-anpr-road-traffic-camera)
- **[GLOBAL product]** Hikvision sells a solar ANPR 4 MP 4G kit with an 80 W panel for "remote roads where fixed network infrastructure is absent" — [Security Wholesalers AU](https://www.securitywholesalers.com.au/product/hikvision-ds-2xs6a46g1-p-izs-c36s80-solar-anpr-4mp-bullet-4g-kit-80w-panel/)
- **[GLOBAL] Starlink + solar camera units:** a Japanese "Eco Power Camera - Starlink" uses a 5,000 Wh battery and a 600 W panel — [IPROS](https://pr.www.ipros.com/en/onlystyle/product/detail/2001489038). Senceive sells a solar cabinet that powers a Starlink Mini for remote monitoring (not LPR) — [Senceive](https://www.senceive.com/news/senceive-launches-starlink-compatibility-geowan-and-camerahub). Surveillance-trailer vendors offer Starlink "for truly remote locations" — [2M Technology](https://www.2mtechnology.net/solar-surveillance-trailers/). Orvid markets a dish-free Starlink direct-to-cell solar camera, which is pre-sale and unverified — [Product Hunt](https://www.producthunt.com/products/orvid-first-dish-free-starlink-camera)
- **[GLOBAL guidance]** With on-device LPR, only plate data and a still cross the link, so bandwidth is minimal. Size solar for the worst-season daylight — [Carthage Electronics](https://carthageelectronics.com/?p=3139). One vendor warns that adding a Starlink dish can push a site to about 500 W of solar plus a large battery — [Hubble](https://hubble.com/community/comparisons/starlink-as-iot-gateway-backhaul-power-budget-and-cost-reality-for-remote-deployments/)
- **[GLOBAL, older]** A 2019 IPVM discussion found few LPR cameras that were both solar and cellular, noting LPR often needs a "fairly significant solar power plant" — [IPVM](https://ipvm.com/discussions/lpr-question-b88ea100-88d1-4b8f-bcc4-4342542e4478)
- **Bangladesh:** the Dhaka-Chattogram highway AI auto-fine system (launched 18 Aug 2026; SMS fine notices; 449 cases in 24 h) is the closest local deployment, but its backhaul and power are not described — [TBS/BSS, 11 Sep 2026](https://www.tbsnews.net/node/1539861). Starlink's main use in Bangladesh is corporate sites, BSCL hill-tract schools and the train Wi-Fi trial, not cameras — [The Daily Star, 14 Apr 2026](https://www.thedailystar.net/business/economy/news/starlink-bangladesh-blazing-fast-few-can-afford-4151116); [Meatechwatch, 16 Mar 2026](https://meatechwatch.com/2026/03/16/starlink-internet-launched-on-bangladesh-trains-in-trial-deployment/)
- **[GLOBAL, adjacent]** Africa Mobile Networks uses Starlink backhaul for solar-powered rural base stations in Nigeria (snippet) — [TelecomTalk](https://telecomtalk.info/?p=967436)

### Inferences
- The proven global pattern is **edge ANPR + small solar + 4G + text/still events**, which is exactly the edge-processing design. The camera itself needs only about 2-6 W (Milesight), so a 100 W panel plus about 0.5-1 kWh of battery covers several days. That matters given Bangladesh's 2-6 h of daily outages and monsoon cloud.
- Adding Starlink multiplies the site's power need (25-40 W Mini continuous) and its capex (BDT 25-47k kit plus about BDT 4,600/month). It suits shared sites or 4G dead zones (hill tracts, char/riverine areas, haor roads), not every pole.
- A Bangladesh project would be among the first documented South Asian satellite-backhauled ALPR deployments, as far as this search could find. That is a possible novelty point, but it needs a more careful literature check.

### Gaps
- No independent case study with measured uptime or alert latency for solar or satellite ALPR anywhere. Vendor pages give specs, not field results.
- No South Asian (India, Pakistan, Sri Lanka, Nepal) deployment record found for solar or satellite ALPR. No Bangladesh Highway Police / RHD technical documentation on camera backhaul.
