# 🍽️ Zomato Restaurant Data — Exploratory Data Analysis (EDA)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-11557C)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)

A hands-on EDA project analyzing Zomato's restaurant dataset to uncover patterns across locations, restaurant types, cuisines, online ordering behavior, and customer ratings. The goal is to extract actionable insights using Python's data science stack.

---

## 📁 Project Structure

```
zomato-eda/
│
├── Zomato_EDA.ipynb       # Main Jupyter Notebook
├── zomato.csv             # Dataset (source: Kaggle)
└── README.md
```

---

## 📦 Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('dark_background')
```

| Library | Purpose |
|--------|---------|
| `pandas` | Data loading, cleaning, transformation |
| `numpy` | Numerical operations, NaN handling |
| `matplotlib` | Base plotting framework |
| `seaborn` | Statistical visualizations (countplot, boxplot) |

---

## 🔄 Phase 1: Data Cleaning & Preprocessing

Raw Zomato data has several quality issues — duplicates, malformed ratings, and irrelevant columns. This phase handles all of that systematically.

### 1.1 Load Dataset

```python
df = pd.read_csv('zomato.csv', on_bad_lines='skip', engine="python")
df.info()
df.shape
```

### 1.2 Remove Duplicates

```python
df.duplicated().sum()
df.drop_duplicates(inplace=True)
```

### 1.3 Fix the `rate` Column

The ratings column has mixed formats like `"4.1/5"`, `"NEW"`, `"-"`. We parse them into clean floats and fill missing values with the mean.

```python
def handle_rate(value):
    if value == 'NEW' or value == '-':
        return np.nan
    else:
        value = str(value).split('/')
        return float(value[0])

df['rate'] = df['rate'].apply(handle_rate)
df['rate'] = df['rate'].fillna(df['rate'].mean())
```

### 1.4 Drop Irrelevant Columns & Rename

```python
df = df.drop(['url', 'address', 'phone', 'menu_item', 'dish_liked', 'reviews_list'], axis=1)
df = df.dropna()

# Rename for clarity
df = df.rename(columns={
    'approx_cost(for two people)': 'cost2people',
    'listed_in(type)': 'type'
})
```

### 1.5 Consolidate Low-Frequency Categories

To reduce noise, infrequent restaurant types, cuisines, and locations are all clubbed into `'others'`.

```python
# Restaurant types with < 1000 entries → 'others'
rest_type_lessthan1000 = df['rest_type'].value_counts()[df['rest_type'].value_counts() < 1000]

def handle_rest_type(value):
    return 'others' if value in rest_type_lessthan1000 else value

df['rest_type'] = df['rest_type'].apply(handle_rest_type)
```

```python
# Cuisines with < 200 entries → 'others'
cuisine_less_than_200 = df['cuisines'].value_counts()[df['cuisines'].value_counts() < 200]

def handle_cuisines(value):
    return 'others' if value in cuisine_less_than_200 else value

df['cuisines'] = df['cuisines'].apply(handle_cuisines)
```

```python
# Locations with < 300 entries → 'others'
location_less_than_300 = df['location'].value_counts()[df['location'].value_counts() < 300]

def handle_location(value):
    return 'others' if value in location_less_than_300 else value

df['location'] = df['location'].apply(handle_location)
```

---

## 📊 Phase 2: Exploratory Visualizations

### 2.1 Restaurant Density by Location

Identifies which areas of Bengaluru have the highest restaurant concentration — useful for competitive analysis or expansion planning.

```python
plt.figure(figsize=(10, 10))
sns.countplot(df['location'], palette='plasma')
plt.xticks(rotation=90)
plt.title('Restaurant Count by Location')
plt.show()
```

### 2.2 Online Ordering & Table Booking Distribution

```python
# Online Order availability
plt.figure(figsize=(6, 6))
sns.countplot(df['online_order'], palette='plasma')
plt.title('Online Order Availability')
plt.show()

# Table Booking availability
plt.figure(figsize=(6, 6))
sns.countplot(df['book_table'], palette='plasma')
plt.title('Table Booking Availability')
plt.show()
```

---

## 📈 Phase 3: Feature Impact on Ratings

### 3.1 Online Ordering vs Ratings

```python
plt.figure(figsize=(6, 6))
sns.boxplot(x='online_order', y='rate', data=df, palette='plasma')
plt.title('Online Order vs Rating')
plt.show()
```

> 💡 **Insight:** Restaurants with online ordering tend to have slightly different rating distributions — reflecting the convenience factor valued by customers.

### 3.2 Table Booking vs Ratings

```python
plt.figure(figsize=(6, 6))
sns.boxplot(x='book_table', y='rate', data=df, palette='plasma')
plt.title('Table Booking vs Rating')
plt.show()
```

> 💡 **Insight:** Restaurants that offer table booking generally show higher median ratings, indicating a correlation between pre-booking services and customer satisfaction.

### 3.3 Restaurant Type vs Ratings

```python
plt.figure(figsize=(6, 6))
sns.boxplot(x='type', y='rate', data=df, palette='plasma')
plt.xticks(rotation=90)
plt.title('Restaurant Type vs Rating')
plt.show()
```

---

## 🗺️ Phase 4: Location-Wise Restaurant Type Analysis

Groups restaurants by location and type using a pivot table to reveal which dining formats are dominant in each area.

```python
df1 = df.groupby(['location', 'type'])['name'].count()
df1.to_csv('location_type.csv')

df2 = pd.read_csv('location_type.csv')
df2 = pd.pivot_table(df2, index='location', columns='type', fill_value=0, aggfunc=np.sum)
df2
```

> 💡 **Insight:** Certain locations are heavily skewed toward delivery/takeaway formats while others have more dine-in options — reflecting the demographics and lifestyle of those areas.

---

## 🍛 Phase 5: Cuisine Analysis

### 5.1 Top Cuisines by Votes

Votes act as a proxy for popularity. Cuisines are ranked by total votes across all restaurants.

```python
df3 = df[['cuisines', 'votes']].drop_duplicates()
df4 = df3.groupby(['cuisines'])['votes'].sum()
df4 = df4.to_frame().sort_values('votes', ascending=False)

# Remove top entry if it skews the chart (often 'others')
df4 = df4.iloc[1:, :]
df4.head(10)
```

> 💡 **Insight:** North Indian and Chinese cuisines consistently rank among the most voted categories, pointing to broad popularity across Bengaluru's restaurant scene.

---

## 🔑 Key Takeaways

| Area | Finding |
|------|---------|
| **Location** | BTM, Koramangala, and HSR Layout have the highest restaurant density |
| **Online Ordering** | Majority of restaurants offer online ordering; slight rating variance observed |
| **Table Booking** | Restaurants with booking options tend to have higher ratings |
| **Restaurant Type** | Delivery and casual dining dominate; fine dining maintains consistently high ratings |
| **Cuisine** | North Indian, Chinese, and South Indian are the most popular cuisines by votes |

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/zomato-eda.git
   cd zomato-eda
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn jupyter
   ```

3. Launch the notebook:
   ```bash
   jupyter notebook Zomato_EDA.ipynb
   ```

> **Dataset:** Download `zomato.csv` from [Kaggle - Zomato Bangalore Restaurants](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants) and place it in the project root.

---

## 🛠️ Skills Demonstrated

- **Data Cleaning** — handling malformed values, duplicates, missing data
- **Feature Engineering** — category consolidation using custom functions + `.apply()`
- **Pandas** — groupby, pivot tables, value_counts, filtering
- **Visualization** — countplot, boxplot with seaborn; matplotlib customization
- **Exploratory Analysis** — deriving business insights from raw restaurant data

---

## 📬 Connect

Feel free to fork, star ⭐, or raise an issue if you have suggestions!
