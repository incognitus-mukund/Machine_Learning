import numpy as np
import pandas as pd

data = {
    "Name": ["Sumit", "Gaurav", "Bhanu", "Aditya", "Shivam"],
    "Roll_No": [47, 14, 17, 11, 43],
    "Marks": [95, 86, 89, 100, 77],
    "Attendance": [34, 86, 27, 99, 59]
}

df = pd.DataFrame(data)


def assign_grade(Marks):
    if Marks >= 90:
        return "A"
    elif Marks >= 80 and Marks <= 89:
        return "B"
    elif Marks >= 70:
        return "C"
    elif Marks >= 60:
        return "D"
    else:
        return "Fail"


df["Grade"] = df["Marks"].apply(assign_grade)

print(df)
