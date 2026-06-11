# Customer Segmentation with K-Means

An unsupervised machine learning project that segments mall customers into natural groups using K-Means clustering — no labels, pure pattern discovery.
Includes both a classic 2-feature approach and an advanced multi-feature approach visualized with PCA.

---

## Features

- Exploratory analysis of income, spending, and age patterns
- Elbow method to choose the number of clusters K
- K-Means clustering on income vs spending (2 features)
- Multi-feature clustering (Age + Income + Spending) with PCA visualization
- Silhouette score as a second method for evaluating K
- Business interpretation of each customer segment

---

## Project Structure

```
customer_segmentation/
├── EDA.ipynb           # Full clustering pipeline
├── Mall_Customers.csv  # Dataset (200 customers)
└── README.md
```

---

## Requirements

- Python 3.10 or higher
- pandas, numpy, matplotlib, seaborn, scikit-learn

Install:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## How to Run

```
jupyter notebook EDA.ipynb
```

---

## Approach 1 — Income vs Spending (2 Features)

### Steps
1. Scale features with `StandardScaler` so income (0–140) doesn't dominate spending (1–100)
2. Run the elbow method for K = 1 to 10
3. Fit K-Means with K=5
4. Visualize clusters with centroids
5. Interpret each segment

### The Five Segments

| Segment | Income | Spending | Strategy |
|---|---|---|---|
| **Target** | High | High | Most valuable — focus marketing here |
| **Careful** | High | Low | Untapped potential — targeted promotions |
| **Careless** | Low | High | Impulse buyers — respond to sales |
| **Standard** | Average | Average | Largest, most stable group |
| **Sensible** | Low | Low | Budget conscious — discount-driven |

---

## Approach 2 — Multi-Feature Clustering + PCA

### Steps
1. Add Age as a third feature (Age + Income + Spending)
2. Drop Gender — PCA assumes continuous variables, so binary categories distort the variance-based projection
3. Use both elbow method and silhouette score to evaluate K
4. Cluster with K=6 for business interpretability
5. Project to 2D with PCA for visualization
6. Interpret clusters using original feature averages

### Why PCA + K-Means

```
Cluster on raw features  → keeps interpretability ("high income" still means something)
Visualize in PCA space   → lets us plot 3D clusters in 2D
Interpret with raw means → business insight stays intact
```

This is the standard industry approach for clustering high-dimensional data.

---

## Key Concepts

**K-Means algorithm:**
```
1. Pick K random center points
2. Assign each customer to the nearest center
3. Move each center to the average of its members
4. Repeat until centers stop moving
```

**Elbow method:** Plot inertia (total distance to centers) for increasing K. The "elbow" — where the curve bends and flattens — is the optimal K.

**Silhouette score:** Measures how well-separated clusters are. Ranges from -1 (bad) to +1 (well separated). A second opinion on choosing K.

**Why scaling matters:**
```
Income:   ranges 0-140
Spending: ranges 1-100
Without scaling, income dominates the distance calculation.
StandardScaler puts all features on equal footing.
```

---

## Why Unsupervised Learning Matters

```
Supervised (Titanic, Housing):
  X → model → y    you KNOW the right answer to learn from

Unsupervised (this project):
  X → model → ???  there is NO right answer — you discover structure
```

This is the mindset shift for real-world data where nobody hands you labels. K-Means found the customer groups on its own — the analyst's job is translating those groups into business strategy.

---

## What This Connects To

| Concept | Source |
|---|---|
| PCA dimensionality reduction | PCA Visualizer project |
| StandardScaler in pipelines | ML Pipeline project |
| Unsupervised mindset | Extends PCA's label-free thinking |
| Distance-based grouping | K-Means core mechanism |
