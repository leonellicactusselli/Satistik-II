import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import math
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

SCRIPT_DIR = Path(__file__).resolve().parent
csv_path = SCRIPT_DIR / "world_happiness_report.csv"
happiness_df = pd.read_csv(csv_path)

#data frame for the year 2015
df_2015 = happiness_df[happiness_df["year"] == 2015].copy()
# drop unnamed helper columns often created during CSV export
df_2015 = df_2015.loc[:, ~df_2015.columns.str.contains("^Unnamed")]
# remove non-essential columns for current visualizations
df_2015 = df_2015.drop(columns=["Happiness Rank", "Standard Error", "year"], errors="ignore")
print("df_2015 preview:")
print(df_2015.head())

# overview of columns and their dtypes
column_info = pd.DataFrame({"column": df_2015.columns, "dtype": df_2015.dtypes.astype(str)})
print("Spalten und Datentypen in df_2015:")
print(column_info)


# Check for missing data in 2015 subset
missing_2015 = df_2015.isna().sum()
print("Missing values per column in df_2015:")
#print(missing_2015)


radar_features = [
	"Freedom",
	"Family",
	"Health (Life Expectancy)",
	"Economy (GDP per Capita)"
]
region_means = df_2015.groupby("Region")[radar_features].mean().dropna()
if not region_means.empty:
	angles = np.linspace(0, 2 * math.pi, len(radar_features), endpoint=False)
	angles = np.concatenate((angles, [angles[0]]))
	plt.figure(figsize=(8, 8))
	ax = plt.subplot(111, polar=True)
	for region, row in region_means.iterrows():
		values = row.tolist()
		values += values[:1]
		ax.plot(angles, values, label=region)
		ax.fill(angles, values, alpha=0.1)
	ax.set_xticks(angles[:-1])
	ax.set_xticklabels(radar_features)
	ax.set_title("Regional Comparison of Happiness Drivers (2015)")
	ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
	plt.tight_layout()
	plt.show()



# Numeric columns for scatterplots
numeric_cols_2015 = df_2015.select_dtypes(include=["number"])

# Scatterplots of Happiness Score against each numeric driver
feature_columns = [col for col in numeric_cols_2015.columns if col != "Happiness Score"]
n_cols = 3
n_rows = math.ceil(len(feature_columns) / n_cols) or 1
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
axes = axes.flatten()
for ax, feature in zip(axes, feature_columns):
	sns.scatterplot(data=df_2015, x=feature, y="Happiness Score", ax=ax)
	ax.set_title(f"Happiness Score vs {feature}")

# hide unused axes if any
for ax in axes[len(feature_columns):]:
	ax.axis("off")

plt.tight_layout()
plt.show()

# Choropleth map for worldwide 2015 happiness scores
fig = px.choropleth(
	df_2015,
	locations="Country",
	locationmode="country names",
	color="Happiness Score",
	hover_name="Country",
	color_continuous_scale="YlGnBu",
	title="World Happiness Scores (2015)"
)
fig.show()


# Modeling data for OLS comparisons
model_columns = [
	"Happiness Score",
	"Economy (GDP per Capita)",
	"Family",
	"Freedom",
	"Trust (Government Corruption)",
	"Generosity"
]
model_df = df_2015[model_columns].dropna().copy()

# Multiple linear regression with train/test split for baseline reference
X_pred = model_df.drop(columns=["Happiness Score"])
y = model_df["Happiness Score"].copy()
X_train, X_test, y_train, y_test = train_test_split(
	X_pred,
	y,
	test_size=0.3,
	random_state=42
)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
r2_score_test = r2_score(y_test, y_pred)
print(f"R^2 score on test data (Linear Regression): {r2_score_test:.3f}")

# Statsmodels OLS: Model A (GDP focus)
X_model_a = model_df[[
	"Economy (GDP per Capita)",
	"Freedom",
	"Trust (Government Corruption)",
	"Generosity"
]]
X_model_a = sm.add_constant(X_model_a)
ols_model_a = sm.OLS(y, X_model_a).fit()
print("\nModell A (Wirtschafts-Fokus) Summary:")
print(ols_model_a.summary())

# Statsmodels OLS: Model B (Social focus)
X_model_b = model_df[[
	"Family",
	"Freedom",
	"Trust (Government Corruption)",
	"Generosity"
]]
X_model_b = sm.add_constant(X_model_b)
ols_model_b = sm.OLS(y, X_model_b).fit()
print("\nModell B (Sozialer Fokus) Summary:")
print(ols_model_b.summary())

# Side-by-side coefficient comparison for the two OLS models
coef_comparison = pd.DataFrame({
	"Model_A": ols_model_a.params,
	"Model_B": ols_model_b.params
})
print("\nKoeffizientenvergleich beider Modelle:")
print(coef_comparison)


# ==========================================
# VISUALISIERUNGEN (FÜR DIE .PY DATEI)
# ==========================================
print("\n--- Starte Visualisierung ---")
print("BITTE BEACHTEN: Du musst jedes Fenster schließen, damit das nächste erscheint!")

# 1. Histogramm: Beweis der Normalverteilung (Antwort auf Lehrer-Feedback)
plt.figure(figsize=(10, 6))
sns.histplot(df_2015["Happiness Score"], kde=True, color="skyblue",bins=50)
plt.title("Verteilung der Zielvariable (Happiness Score)")
plt.xlabel("Happiness Score")
plt.ylabel("Anzahl Länder")
# feinere Unterteilung der Happiness-Skala für bessere Lesbarkeit
score_min = df_2015["Happiness Score"].min()
score_max = df_2015["Happiness Score"].max()
tick_step = 0.2
plt.xticks(np.arange(math.floor(score_min), math.ceil(score_max) + tick_step, tick_step))
print("Zeige Grafik 1/3: Histogramm... (Fenster schließen zum Fortfahren)")
plt.savefig(SCRIPT_DIR / "grafik_1_verteilung.png")
plt.show()

# 2. Heatmap: Beweis der Multikollinearität (Warum wir 2 Modelle brauchen)
corr_cols = [
	"Happiness Score",
	"Economy (GDP per Capita)",
	"Family",
	"Health (Life Expectancy)",
	"Freedom",
	"Trust (Government Corruption)"
]
plt.figure(figsize=(10, 8))
sns.heatmap(df_2015[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korrelations-Matrix (Das Problem: GDP vs Family)")
print("Zeige Grafik 2/3: Heatmap... (Fenster schließen zum Fortfahren)")
plt.savefig(SCRIPT_DIR / "grafik_2_korrelation.png")
plt.show()

# 3. Residuen-Plot: Validierung des Gewinner-Modells (hier Modell A)
residuals = ols_model_a.resid
fitted = ols_model_a.fittedvalues
plt.figure(figsize=(10, 6))
plt.scatter(fitted, residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuen vs. Fitted Values (Modell A)")
plt.xlabel("Vorhergesagte Werte")
plt.ylabel("Residuen (Fehler)")
print("Zeige Grafik 3/3: Residuen-Plot... (Fenster schließen zum Beenden)")
plt.savefig(SCRIPT_DIR / "grafik_3_residuen.png")
plt.show()

print("--- Analyse beendet ---")