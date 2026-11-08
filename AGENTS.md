# Project instructions

- This repository is for IDS706 Week 11. Keep code and README in clear, simple English; explain progress to the owner in Chinese.
- Keep completed Part 1 and Part 2 evidence tied to their real runs. Record Bonus as pending until its code and real runs exist.
- Use only the Zenodo record 19194192 and the World Bank China indicators named in README. Save source URLs and hashes for every run.
- Filter the investee to Country = China and Industry = Technology & Software. Use Type = investment for supervised learning. Count the current source; never hard-code preliminary counts.
- A funding amount is the whole round, not one investor's contribution. Zero means undisclosed. Retain source rows and investor fields. Quarantine unresolved candidate-event conflicts with reasons.
- A company, year, and round grouping is only a candidate event. Do not invent missing amounts or claim verified transaction identity.
- In later model work, exclude amount and its derivatives, company names, and IDs from features. Freeze company-grouped train, validation, and test splits; fit preprocessing on train only; use validation for decisions and test once for final reporting.
- Preserve each run under its own run_id. Record source hashes, cleaning counts, quarantine reasons, merge checks, database counts, task times, and later model and agent evidence.
- Screenshots and metrics must come from actual execution. Mark unavailable captures pending. Do not add generated or simulated screenshots.
- The owner authorized a short commit and push after each completed Part or major feature. Use the owner's configured Git identity. Do not deploy or submit without separate authorization. Keep local private material under `private/`, which is ignored by Git.
- Keep README claims tied to files and real runs that exist in the repository.
