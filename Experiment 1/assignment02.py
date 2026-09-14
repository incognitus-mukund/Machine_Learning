import numpy as np
import pandas as pd

data = {
    "Name": ["Sumit", "Gaurav", "Bhanu", "Aditya", "Shivam"],
    "Roll_No": [47, 14, 17, 11, 43],
    "Marks": [95, 86, 89, 100, 77],
    "Attendance": [34, 86, 27, 99, 59]
}

df = pd.DataFrame(data)

x = df[df["Marks"] > 80]

print(x)
