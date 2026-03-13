# NFL Player & Team Analytics

[License](https://img.shields.io/badge/license-MIT-green)

---

## Quick Access

| Component | Action |
| --- | --- |
| **Player Visualizations Notebook** | [Open](visulalizations.ipynb) |
| **QB Support Notebook** | [Open](qbs.ipynb) |
| **Weekly Rankings Script** | [Open](rankings.py) |
| **Charts Gallery** | [View](figures/) |

---

## Overview

This project provides interactive NFL analytics at both the player and team level:

- Player and team visualizations: Explosiveness, consistency, cornerback coverage, redzone efficiency
- QB support: Performance vs support, team-color lines, logos/headshots
- Weekly rankings: Automated output of top-performing teams

---

## Project Structure

```
├── visualizations.ipynb
├── qbs.ipynb
├── rankings.py
├── csv/                # CSV outputs from quarterback support notebook
├── data/               # JSON from NFL API
├─- figures/            # Final interactive html and png figures
│   ├─ images/          
│   └─ html/ 
├── README.md
├── requirements.txt
```

---

## Setup

```
git clone <repo-url>
cd <project-folder>
python-m venv venv
source venv/bin/activate# macOS/Linux
venv\Scripts\activate# Windows
pip install-r requirements.txt
```

---

## Usage

### Player Visualizations Notebook

<details>
<summary>Click to Expand</summary>

- Metrics & charts:
    - Receiver Explosiveness & Consistency
    - Cornerback Target Impact
    - Redzone efficiency
    - Other player-level metrics (EPA/play, success rate, turnover-worthy plays)
- Interactive hover data: logos & headshots

**Example Charts:**

[QB Efficiency](figures/images/passing/qb_efficiency.png)

</details>

### QB Support Notebook

<details>
<summary>Click to Expand</summary>

- Visualizes QB support vs performance
- Team-color lines & logos/headshots
- Custom metrics like QB fault sack rate

**Final Output:**

[QB Support](figures/images/qbs/qbs_support_vs_performance.png)

</details>

### Weekly Rankings Script

<details>
<summary>Click to Expand</summary>

- Automatically generates weekly team rankings based on 5 statsistical categories

```
python rankings.py
```

**Example Output:**

[Offensive Table](figures/images/2025_week_18_metrics/offense_rankings_week_18.jpeg)

[Weekly Plot](figures/images/2025_week_18_metrics/team_ratings_scatter_week_18.jpeg)

</details>

---

## Notes / Limitations

- API access required; raw data not included
- Missing logos/headshots may break some visuals
- Some metrics are **custom-calculated**, not official NFL stats

---

## Contributing

- Fork the repo, add visualizations or metrics, submit PR
- Do **not** include raw API data

---

## License

MIT License