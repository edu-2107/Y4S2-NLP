import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV and fix decimal commas
df = pd.read_csv('models.csv', dtype=str)
df = df.drop(columns=['Responsible'])  # Ignore 'Responsible'

# Replace ',' with '.' and convert to numeric where possible
for col in df.columns:
    if col != 'Model':
        df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Plotting function
def plot_grouped_bars(prefix, title, ylabel):
    cols = [c for c in df.columns if c.startswith(prefix)]
    melted = df.melt(id_vars='Model', value_vars=cols, var_name='Class', value_name='Value')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x='Class', y='Value', hue='Model')
    plt.title(f'{title} per Class')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot Precision, Recall, F1-Score
plot_grouped_bars('Precision 0', 'Precision 1', 'Precision 2')
plot_grouped_bars('Recall 0', 'Recall 1', 'Recall 2')
plot_grouped_bars('F1-Score 0', 'F1-Score 1', 'F1 Score 2')

# Plot Support
plot_grouped_bars('Support', 'Support', 'Number of Samples')

# Plot Accuracy
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='Model', y='Accuracy')
plt.title('Accuracy by Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
