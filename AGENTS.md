# Project instructions

- This repository is for IDS706 Week 11. Keep code and README in clear, simple English.
- Link Part 1 and Part 2 results to their run files. Add Bonus evidence as that work is completed.
- Use only the Zenodo record 19194192 and the World Bank China indicators named in README. Save source URLs and hashes for every run.
- Filter the investee to Country = China and Industry = Technology & Software. Use Type = investment for supervised learning. Count the current source; never hard-code preliminary counts.
- A funding amount is the whole round, not one investor's contribution. Zero means undisclosed. Retain source rows and investor fields. Quarantine unresolved candidate-event conflicts with reasons.
- A company, year, and round grouping is only a candidate event. Do not invent missing amounts or claim verified transaction identity.
- For supervised modeling, exclude amount and its derivatives, company names, and IDs from features. Freeze company-grouped train, validation, and test splits; fit preprocessing on train only; use validation for decisions and test once for final reporting.
- Preserve each run under its own run_id. Record source hashes, cleaning counts, quarantine reasons, merge checks, database counts, task times, and later model and agent evidence.
- Match screenshots and metrics to the corresponding run artifacts.
- Make a short commit and push after each completed Part or major feature using the configured Git identity. Do not deploy or submit without separate authorization. Keep local private material under `private/`, which is ignored by Git.
- Keep README claims tied to existing run files.
