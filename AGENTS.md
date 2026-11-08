# Project instructions

- This repository is for IDS706 Week 11. Keep code and README in clear, simple English.
- Link Part 1 and Part 2 results to their run files. Add Bonus evidence as that work is completed.
- Use only the Zenodo record 19194192 and the World Bank China indicators named in README. Save source URLs and hashes for every run.
- Filter the investee to Country = China and Industry = Technology & Software. Use Type = investment for supervised learning. Count the current source; never hard-code preliminary counts.
- A funding amount is the whole round, not one investor's contribution. Zero means undisclosed. Retain source rows and investor fields. Quarantine unresolved candidate-event conflicts with reasons.
- A company, year, and round grouping is only a candidate event. Do not invent missing amounts or claim verified transaction identity.
- For supervised modeling, exclude amount and its derivatives, company names, and IDs from features. Freeze company-grouped train, validation, and test splits; fit preprocessing on train only; use validation for decisions and test once for final reporting.
- Preserve each run under its own run_id. Record source hashes, cleaning counts, quarantine reasons, merge checks, database counts, task times, and later model and agent evidence.
- In the Bonus, validate Planner output before building the execution graph. Compare the fixed Airflow, simple agent, and adaptive agent workflows on the same source snapshot, frozen split, and model-fit budget. Keep test metrics out of routing decisions.
- Send the Planner only training label counts and validation size, never total or test label counts. Different validation feedback must change the next model settings or stop the workflow. Evaluate those branches with controlled inputs.
- Choose the latest completed Part 1 run by actual completion time and the Part 2 page run by actual start time, not by run ID prefix.
- Match screenshots and metrics to the corresponding run artifacts.
- Make a short commit and push after each completed Part or major feature using the configured Git identity. Do not deploy or submit without separate authorization. Keep local private material under `private/`, which is ignored by Git.
- Read API keys from the environment or ignored `private/` files. Do not write secrets to code, run artifacts, screenshots, or commits.
- Keep README claims tied to existing run files.
