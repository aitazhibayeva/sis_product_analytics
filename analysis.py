import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

student_id = input("Enter your student ID: ").strip()
filename = f"sis_dataset_{student_id}.csv"
df = pd.read_csv(filename)
df['install_date'] = pd.to_datetime(df['install_date'])

print("MOBILE GAME PRODUCT ANALYSIS")
print("\nSTEP 1: PRODUCT OVERVIEW")

total_users = len(df)
paying_users = (df['revenue'] > 0).sum()
pct_paying = (paying_users / total_users) * 100

days_in_period = (df['install_date'].max() - df['install_date'].min()).days + 1
dau = total_users / days_in_period
mau = total_users
stickiness = dau / mau

avg_session_length = df['session_length'].mean()

print(f"Total Users: {total_users}")
print(f"DAU (estimated): {dau:.1f}")
print(f"MAU: {mau}")
print(f"Stickiness (DAU/MAU): {stickiness:.2%}")
print(f"Average Session Length: {avg_session_length:.2f} minutes")
print(f"Paying Users: {paying_users} ({pct_paying:.1f}%)")

print("\n\nSTEP 2: FUNNEL ANALYSIS")

funnel_steps = {
    'Install': total_users,
    'Tutorial Complete': df['tutorial_complete'].sum(),
    'Retained D7': df['retention_d7'].sum(),
    'Purchase': paying_users
}

print("Funnel Steps:")
prev_value = total_users
for i, (step, count) in enumerate(funnel_steps.items()):
    conversion = (count / prev_value) * 100 if prev_value > 0 else 0
    print(f"{i+1}. {step}: {count} users ({conversion:.1f}% conversion)")
    prev_value = count

plt.figure(figsize=(10, 6))
steps = list(funnel_steps.keys())
values = list(funnel_steps.values())
plt.bar(steps, values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
plt.title('User Funnel Analysis', fontsize=16, fontweight='bold')
plt.ylabel('Number of Users')
plt.xlabel('Funnel Step')
for i, v in enumerate(values):
    plt.text(i, v + 10, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f'funnel_analysis_{student_id}.png', dpi=300, bbox_inches='tight')
print(f"\nFunnel chart saved: funnel_analysis_{student_id}.png")

conversions = []
for i in range(len(values)-1):
    conv = (values[i+1] / values[i]) * 100
    conversions.append(conv)
    
worst_step_idx = conversions.index(min(conversions))
print(f"\nBiggest Drop-off: {steps[worst_step_idx]} → {steps[worst_step_idx+1]}")
print(f"Only {min(conversions):.1f}% conversion")
print(f"Recommendation: Simplify tutorial or add incentives for completion")

print("\n\nSTEP 3: RETENTION & COHORT ANALYSIS")

df['week'] = df['install_date'].dt.isocalendar().week
df['cohort'] = 'Week ' + df['week'].astype(str)

cohort_analysis = df.groupby('cohort').agg({
    'user_id': 'count',
    'tutorial_complete': 'mean',
    'retention_d7': 'mean',
    'revenue': 'mean'
}).round(3)

cohort_analysis.columns = ['Users', 'D1_Retention', 'D7_Retention', 'Avg_Revenue']
print("\nCohort Performance:")
print(cohort_analysis)

cohort_pivot = df.pivot_table(
    index='cohort', 
    values='retention_d7', 
    aggfunc='mean'
)

plt.figure(figsize=(10, 6))
sns.heatmap(cohort_pivot.values.reshape(-1, 1), 
            annot=True, 
            fmt='.2%', 
            cmap='RdYlGn',
            yticklabels=cohort_pivot.index,
            xticklabels=['D7 Retention'],
            cbar_kws={'label': 'Retention Rate'})
plt.title('Weekly Cohort Retention Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'retention_heatmap_{student_id}.png', dpi=300, bbox_inches='tight')
print(f"\nRetention heatmap saved: retention_heatmap_{student_id}.png")

avg_d7_retention = df['retention_d7'].mean()
print(f"\nOverall D7 Retention: {avg_d7_retention:.1%}")
print(f"Industry benchmark for casual games: 15-25%")

print("\n\nSTEP 4: MONETIZATION ANALYSIS")

total_revenue = df['revenue'].sum()
arpu = total_revenue / total_users
arppu = total_revenue / paying_users if paying_users > 0 else 0
conversion_rate = paying_users / total_users

print(f"Total Revenue: ${total_revenue:.2f}")
print(f"ARPU (Average Revenue Per User): ${arpu:.2f}")
print(f"ARPPU (Average Revenue Per Paying User): ${arppu:.2f}")
print(f"Conversion Rate: {conversion_rate:.2%}")

avg_lifespan_days = 60
ltv = arpu * (avg_lifespan_days / 30)
print(f"\nEstimated LTV (60-day): ${ltv:.2f}")

revenue_threshold = df['revenue'].quantile(0.95)
whales = df[df['revenue'] >= revenue_threshold]
whale_revenue = whales['revenue'].sum()
whale_share = (whale_revenue / total_revenue) * 100 if total_revenue > 0 else 0

print(f"\nWhale Analysis (Top 5%):")
print(f"Number of Whales: {len(whales)}")
print(f"Whale Revenue: ${whale_revenue:.2f}")
print(f"Share of Total Revenue: {whale_share:.1f}%")
print(f"Average Whale Spending: ${whales['revenue'].mean():.2f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df[df['revenue'] > 0]['revenue'].hist(bins=30, color='#2ecc71', edgecolor='black')
plt.title('Revenue Distribution (Paying Users)', fontweight='bold')
plt.xlabel('Revenue ($)')
plt.ylabel('Number of Users')

plt.subplot(1, 2, 2)
revenue_segments = ['Non-Payers', 'Small Spenders', 'Whales']
revenue_values = [
    (df['revenue'] == 0).sum(),
    ((df['revenue'] > 0) & (df['revenue'] < revenue_threshold)).sum(),
    len(whales)
]
plt.pie(revenue_values, labels=revenue_segments, autopct='%1.1f%%',
        colors=['#e74c3c', '#f39c12', '#2ecc71'])
plt.title('User Segmentation by Revenue', fontweight='bold')

plt.tight_layout()
plt.savefig(f'monetization_analysis_{student_id}.png', dpi=300, bbox_inches='tight')
print(f"\nMonetization charts saved: monetization_analysis_{student_id}.png")

print("\n\nSTEP 5: A/B TESTING ANALYSIS")

group_a = df[df['group'] == 'A']
group_b = df[df['group'] == 'B']

metrics = {
    'Tutorial Completion': ('tutorial_complete', 'mean'),
    'D7 Retention': ('retention_d7', 'mean'),
    'Average Revenue': ('revenue', 'mean')
}

print("Comparing Group A (Control) vs Group B (New Version):\n")

results = []
for metric_name, (column, func) in metrics.items():
    a_value = getattr(group_a[column], func)()
    b_value = getattr(group_b[column], func)()
    
    if func == 'mean' and column in ['tutorial_complete', 'retention_d7']:
        n_a = len(group_a)
        n_b = len(group_b)
        p_a = a_value
        p_b = b_value
        
        p_pooled = (p_a * n_a + p_b * n_b) / (n_a + n_b)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
        z_score = (p_a - p_b) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        significance = "Significant" if p_value < 0.05 else "Not Significant"
        
        print(f"{metric_name}:")
        print(f"  Group A: {p_a:.2%}")
        print(f"  Group B: {p_b:.2%}")
        print(f"  Difference: {(p_b - p_a):.2%}")
        print(f"  Z-score: {z_score:.3f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Result: {significance}\n")
        
        results.append({
            'metric': metric_name,
            'group_a': p_a,
            'group_b': p_b,
            'significant': p_value < 0.05
        })
    else:
        print(f"{metric_name}:")
        print(f"Group A: ${a_value:.2f}")
        print(f"Group B: ${b_value:.2f}")
        print(f"Difference: ${b_value - a_value:.2f}\n")

significant_improvements = sum(1 for r in results if r['significant'] and r['group_b'] > r['group_a'])
print(f"DECISION:")
if significant_improvements >= 2:
    print("ROLL OUT Group B version")
    print("Multiple significant improvements detected")
else:
    print("NEEDS MORE DATA or A/A testing")
    print("No clear winner or insufficient significance")

print("\n\nSTEP 6: GROWTH & UA ANALYSIS")

assumed_cac = np.random.uniform(2, 5)
total_ua_spend = assumed_cac * total_users

print(f"Simulated UA Metrics:")
print(f"CAC (Cost Per Acquisition): ${assumed_cac:.2f}")
print(f"Total UA Spend: ${total_ua_spend:.2f}")
print(f"LTV: ${ltv:.2f}")
print(f"LTV:CAC Ratio: {ltv/assumed_cac:.2f}x")

if ltv / assumed_cac > 3:
    print(f"\nMarketing is SUSTAINABLE (LTV > 3x CAC)")
elif ltv / assumed_cac > 1:
    print(f"\nMarketing is MARGINALLY PROFITABLE (1x < LTV:CAC < 3x)")
else:
    print(f"\nMarketing is NOT SUSTAINABLE (LTV < CAC)")

roi = ((total_revenue - total_ua_spend) / total_ua_spend) * 100
print(f"ROI: {roi:.1f}%")

print("\n\nSTEP 7: INSIGHTS & RECOMMENDATIONS")

print("\nENGAGEMENT:")
if stickiness > 0.2:
    print("Good stickiness - users return frequently")
else:
    print("Low stickiness - improve daily rewards/content")

print(f"Session length ({avg_session_length:.1f}min) suggests {'casual' if avg_session_length < 15 else 'mid-core'} gameplay")

print("\nRETENTION:")
if avg_d7_retention > 0.25:
    print("Strong D7 retention above industry average")
else:
    print("D7 retention needs improvement")
print(f"   Focus on: {steps[worst_step_idx]} → {steps[worst_step_idx+1]} transition")

print("\nMONETIZATION:")
if whale_share > 50:
    print(f"Whale-dependent ({whale_share:.0f}% from top 5%)")
    print("Risk: Not sustainable long-term")
else:
    print(f"Balanced monetization ({whale_share:.0f}% from top 5%)")
print(f"ARPPU of ${arppu:.2f} suggests {'healthy' if arppu > 5 else 'low'} monetization")

print("\nA/B TEST:")
if significant_improvements >= 2:
    print("New version shows clear improvements")
    print("Recommendation: Full rollout")
else:
    print("Inconclusive results")
    print("Recommendation: Continue testing or revert")

print("\nGROWTH:")
if ltv / assumed_cac > 3:
    print("Strong unit economics - scale up UA")
elif ltv / assumed_cac > 1:
    print("Improve retention/monetization before scaling")
else:
    print("Fix product before increasing UA spend")

print("Analysis Complete! Check the generated images.")

with open(f'analysis_summary_{student_id}.txt', 'w') as f:
    f.write(f"PRODUCT ANALYSIS SUMMARY\n")
    f.write(f"Total Users: {total_users}\n")
    f.write(f"DAU: {dau:.1f}\n")
    f.write(f"Stickiness: {stickiness:.2%}\n")
    f.write(f"D7 Retention: {avg_d7_retention:.1%}\n")
    f.write(f"ARPU: ${arpu:.2f}\n")
    f.write(f"LTV: ${ltv:.2f}\n")
    f.write(f"Conversion Rate: {conversion_rate:.2%}\n")
    f.write(f"LTV:CAC Ratio: {ltv/assumed_cac:.2f}x\n")

print(f"\nSummary saved: analysis_summary_{student_id}.txt")
