
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ──────────────────────────────────────────
# 0.  Setup
# ──────────────────────────────────────────
np.random.seed(42)
os.makedirs("outputs", exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 130


# ──────────────────────────────────────────
# 1.  Generate Realistic Sample Dataset
# ──────────────────────────────────────────
def generate_data(n=500):
    """Create a synthetic e-commerce orders dataset."""
    categories   = ["Electronics", "Clothing", "Books", "Home & Kitchen", "Sports"]
    regions      = ["North", "South", "East", "West"]
    payment_methods = ["Credit Card", "UPI", "Net Banking", "Cash on Delivery"]

    dates = pd.date_range(start="2023-01-01", end="2023-12-31", periods=n)

    category_col = np.random.choice(categories, n)
    price_map = {
        "Electronics":   (500, 50000),
        "Clothing":      (300,  5000),
        "Books":         (100,  2000),
        "Home & Kitchen":(200, 15000),
        "Sports":        (400, 20000),
    }

    prices    = np.array([np.random.randint(*price_map[c]) for c in category_col])
    quantities = np.random.randint(1, 6, n)
    discounts  = np.random.choice([0, 5, 10, 15, 20], n)
    revenue    = prices * quantities * (1 - discounts / 100)

    df = pd.DataFrame({
        "order_id":      range(1001, 1001 + n),
        "date":          dates,
        "category":      category_col,
        "region":        np.random.choice(regions, n),
        "payment_method":np.random.choice(payment_methods, n),
        "price":         prices,
        "quantity":      quantities,
        "discount_pct":  discounts,
        "revenue":       revenue.round(2),
        "customer_rating": np.random.randint(1, 6, n),
    })

    # Inject a few nulls to practise cleaning
    df.loc[np.random.choice(df.index, 15), "customer_rating"] = np.nan
    return df


# ──────────────────────────────────────────
# 2.  Load & Clean
# ──────────────────────────────────────────
print("=" * 55)
print("  E-COMMERCE SALES DATA ANALYSIS")
print("=" * 55)

df = generate_data()
print(f"\n[INFO] Dataset shape : {df.shape}")
print(f"[INFO] Null values   :\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# Fill missing ratings with median
df["customer_rating"] = df["customer_rating"].fillna(df["customer_rating"].median())
df["month"] = df["date"].dt.month_name()
df["month_num"] = df["date"].dt.month

print("\n[INFO] Missing values after cleaning :", df.isnull().sum().sum())


# ──────────────────────────────────────────
# 3.  Exploratory Data Analysis
# ──────────────────────────────────────────
print("\n── Summary Statistics ──")
print(df[["price", "quantity", "revenue", "customer_rating"]].describe().round(2))

# Key KPIs
total_revenue  = df["revenue"].sum()
total_orders   = len(df)
avg_order_val  = df["revenue"].mean()
top_category   = df.groupby("category")["revenue"].sum().idxmax()

print(f"\n── Key KPIs ──")
print(f"  Total Revenue    : ₹{total_revenue:,.0f}")
print(f"  Total Orders     : {total_orders}")
print(f"  Avg Order Value  : ₹{avg_order_val:,.0f}")
print(f"  Top Category     : {top_category}")


# ──────────────────────────────────────────
# 4.  Visualisations
# ──────────────────────────────────────────

# ---------- 4a. Revenue by Category ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("E-Commerce Sales Dashboard — 2023", fontsize=16, fontweight="bold", y=1.01)

cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
axes[0, 0].bar(cat_rev.index, cat_rev.values / 1e6, color=sns.color_palette("muted", len(cat_rev)))
axes[0, 0].set_title("Revenue by Category (₹ Millions)")
axes[0, 0].set_xlabel("Category")
axes[0, 0].set_ylabel("Revenue (₹M)")
axes[0, 0].tick_params(axis="x", rotation=20)

# ---------- 4b. Monthly Revenue Trend ----------
monthly = df.groupby("month_num")["revenue"].sum().reset_index()
axes[0, 1].plot(monthly["month_num"], monthly["revenue"] / 1e6,
                marker="o", color="#2196F3", linewidth=2.5)
axes[0, 1].fill_between(monthly["month_num"], monthly["revenue"] / 1e6, alpha=0.15, color="#2196F3")
axes[0, 1].set_title("Monthly Revenue Trend")
axes[0, 1].set_xlabel("Month")
axes[0, 1].set_ylabel("Revenue (₹M)")
axes[0, 1].set_xticks(range(1, 13))

# ---------- 4c. Region-wise Orders ----------
region_orders = df["region"].value_counts()
axes[1, 0].pie(region_orders, labels=region_orders.index, autopct="%1.1f%%",
               colors=sns.color_palette("pastel"), startangle=140)
axes[1, 0].set_title("Orders by Region")

# ---------- 4d. Customer Rating Distribution ----------
sns.histplot(df["customer_rating"], bins=5, ax=axes[1, 1],
             color="#4CAF50", edgecolor="white")
axes[1, 1].set_title("Customer Rating Distribution")
axes[1, 1].set_xlabel("Rating (1–5)")
axes[1, 1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("outputs/dashboard.png", bbox_inches="tight")
plt.close()
print("\n[SAVED] outputs/dashboard.png")


# ---------- 4e. Heatmap: Category × Region Revenue ----------
pivot = df.pivot_table(values="revenue", index="category",
                       columns="region", aggfunc="sum").round(0)
plt.figure(figsize=(8, 4))
sns.heatmap(pivot / 1e3, annot=True, fmt=".0f", cmap="YlGnBu",
            linewidths=0.5, cbar_kws={"label": "Revenue (₹K)"})
plt.title("Revenue Heatmap — Category × Region (₹ Thousands)")
plt.tight_layout()
plt.savefig("outputs/heatmap.png", bbox_inches="tight")
plt.close()
print("[SAVED] outputs/heatmap.png")


# ---------- 4f. Payment Method Preference ----------
pay_counts = df["payment_method"].value_counts()
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
plt.figure(figsize=(7, 4))
plt.barh(pay_counts.index, pay_counts.values, color=colors)
plt.title("Payment Method Preference")
plt.xlabel("Number of Orders")
plt.tight_layout()
plt.savefig("outputs/payment_methods.png", bbox_inches="tight")
plt.close()
print("[SAVED] outputs/payment_methods.png")


# ──────────────────────────────────────────
# 5.  Insight Report (printed)
# ──────────────────────────────────────────
print("\n" + "=" * 55)
print("  BUSINESS INSIGHTS")
print("=" * 55)

# Top month
best_month_idx = monthly.loc[monthly["revenue"].idxmax(), "month_num"]
best_month_name = pd.Timestamp(2023, int(best_month_idx), 1).strftime("%B")
print(f"\n• Best sales month      : {best_month_name}")

# Avg discount per category
disc_cat = df.groupby("category")["discount_pct"].mean().round(1)
print(f"\n• Avg discount by category:\n{disc_cat.to_string()}")

# High-value orders (revenue > 75th percentile)
threshold = df["revenue"].quantile(0.75)
high_val  = df[df["revenue"] > threshold]
print(f"\n• High-value orders (>{threshold:.0f} ₹) : {len(high_val)} ({len(high_val)/len(df)*100:.1f}%)")

# Correlation hint
corr = df[["price", "quantity", "discount_pct", "revenue", "customer_rating"]].corr()
print(f"\n• Price ↔ Revenue correlation : {corr.loc['price','revenue']:.2f}")

# Export clean data
df.to_csv("outputs/clean_orders.csv", index=False)
print("\n[SAVED] outputs/clean_orders.csv")
print("\n✅  Analysis complete! Check the 'outputs/' folder.\n")