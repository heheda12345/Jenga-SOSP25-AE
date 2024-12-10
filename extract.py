import re
from collections import defaultdict

# File name containing the log
file_name = "test"

# Keywords to look for and process
keywords = [
    "Can allocate seq time:",
    "Can allocate (cal k) time:",
    "Allocate sequence time:",
    "Can append slots TOT:",
    "Get top k time:",
    "Append token ids:",
    "Self attention forward time:",
    "Flash attention metadata build time:",
]

# Dictionary to store sums and counts for each keyword
data = defaultdict(lambda: {"sum": 0.0, "count": 0})

# Process the file
with open(file_name, "r") as file:
    for line in file:
        for keyword in keywords:
            if line.startswith(keyword):
                # Extract the number after the keyword, including scientific notation
                match = re.search(rf"{re.escape(keyword)}\s*([\d.eE+-]+)", line)
                if match:
                    value = float(match.group(1))
                    data[keyword]["sum"] += value
                    data[keyword]["count"] += 1
                    # print(f"Found {keyword.strip(':')}: {value}")

# Calculate and print the averages
for keyword in keywords:
    total_sum = data[keyword]["sum"]
    count = data[keyword]["count"]
    if count > 0:
        average = total_sum / count
        print(f"Average {keyword.strip(':')}: {average:.6f}")
        print(f"Total {keyword.strip(':')}: {total_sum:.6f}, count: {count}")
        print('-' * 50)
    else:
        print(f"No data found for {keyword.strip(':')}")
