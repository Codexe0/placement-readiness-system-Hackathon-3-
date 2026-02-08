import pandas as pd
import numpy as np
import random

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

n = 500
rows = []

for i in range(n):
    student_id = i + 1
    cgpa = round(random.uniform(5.0, 9.8), 2)
    aptitude = random.randint(30, 100)
    coding = random.randint(1, 10)
    communication = random.randint(1, 10)
    mock_interview = random.randint(20, 100)
    internships = random.randint(0, 3)

    # Readiness rule (synthetic logic)
    if cgpa >= 7.0 and aptitude >= 60 and coding >= 6 and communication >= 6 and mock_interview >= 60:
        readiness = "Ready"
    else:
        readiness = "Not Ready"

    rows.append([student_id, cgpa, aptitude, coding, communication, mock_interview, internships, readiness])

df = pd.DataFrame(rows, columns=[
    "student_id","cgpa","aptitude_score","coding_skill",
    "communication_skill","mock_interview_score","internships","readiness"
])

df.to_csv("data/readiness_data.csv", index=False)
print("Readiness dataset generated!")
