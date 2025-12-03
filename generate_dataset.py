import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()
np.random.seed()

student_id = input("Enter your student ID: ").strip()
n_users = random.randint(800, 1200)

data = []
start_date = pd.Timestamp("2025-01-01")

for i in range(n_users):
    user_id = f"{student_id}_{i+1}"
    install_date = start_date + pd.to_timedelta(np.random.randint(0, 60), unit="D")
    sessions = np.random.poisson(3)
    session_length = abs(np.random.normal(20, 8))
    actions = np.random.randint(5, 50)
    revenue = round(np.random.exponential(1.5), 2) if random.random() < 0.25 else 0
    tutorial_complete = np.random.choice([0, 1], p=[0.3, 0.7])
    retention_d7 = np.random.choice([0, 1], p=[0.6, 0.4])
    group = random.choice(["A", "B"])
    country = random.choice(["US", "UK", "BR", "IN", "JP"])
    device = random.choice(["iOS", "Android"])
    
    data.append([
        user_id, install_date, sessions, session_length,
        actions, revenue, tutorial_complete, retention_d7,
        group, country, device
    ])

df = pd.DataFrame(data, columns=[
    "user_id", "install_date", "sessions", "session_length",
    "actions", "revenue", "tutorial_complete", "retention_d7",
    "group", "country", "device"
])

filename = f"sis_dataset_{student_id}.csv"
df.to_csv(filename, index=False)
print(f"Dataset generated: {filename} ({len(df)} users)")
