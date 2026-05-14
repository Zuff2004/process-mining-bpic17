# João Zuffo
# Process Mining Assignment - BPIC17 Analysis
# BPIC-17 Simple Event Log Analysis

# IMPORTANT: Since for the assignment we are supposed to address many different topics and requirements, I decided to first write a plan script based on each step
# The notes will later be exported to the final report

# =====================================================================================
# AUFGABE 3.2 - SIMPLE EVENT LOG ANALYSIS
# =====================================================================================

import pm4py
import pandas as pd
import sys


# =========================================
# 1. LOAD EVENT LOG
# =========================================

# Change the path to the file from "bpic17-process-mining/data/BPI Challenge 2017.xes" to "BPI Challenge 2017.xes.gz" 

try:
    log = pm4py.read_xes("../data/BPI Challenge 2017.xes")
    print("Log Uploaded!")
except:
    print('Unable to download the log file')
    sys.exit()

print('=========== FIRST ROWS ==========')
print(log.head())

print('========== COLUMNS ==========')
print(log.columns)


# =========================================
# 2. BASIC STATISTICS
# =========================================

print('=========== BASIC STATISTICS ==========')

# 2.1: number of events
num_events = len(log) # each row is one event
print(f'Number of events: {num_events}')

# 2.2: number of cases
num_cases = log["case:concept:name"].nunique() # cases are identified through this attribute "case:concept:name"
print(f'Number of cases: {num_cases}')

# 2.3: number of activities
num_activities = log["concept:name"].nunique() # activities are stored in "concept:name" --> activity = type of action 
print(f'Number of activities: {num_activities}')


# =========================================
# 3. PROCESS VARIANTS
# =========================================

# IMPORTANT: A process variant is a unique sequence of activities

# 3.1: group activities by case
variants = log.groupby("case:concept:name")["concept:name"].apply(list)

# 3.2: count distinct variants
num_variants = variants.apply(tuple).nunique()
print(f'Number of process variants: {num_variants}')


# =========================================
# 4. CASE LENGTH
# =========================================

# IMPORTANT: Case = numbers of events in one case --> I could also count num_events / num_cases = avg

# 4.1: calculate case length for each case
case_lengths = log.groupby("case:concept:name").size()

# 4.2: calculate mean case length
avg_case_length = case_lengths.mean()
print(f'Average case length: {avg_case_length:.2f}')

# 4.3: calculate standard deviation case length
std_case_length = case_lengths.std()
print(f'Standard deviation case length: {std_case_length:.2f}')


# =========================================
# 5. CASE DURATION
# =========================================

# IMPORTANT: Duration = time between first and last event of a case
# REMINDER: In days, minutes and seconds

# 5.1: make sure timestamp column in datetime
log["time:timestamp"] = pd.to_datetime((log["time:timestamp"]))
print(log["time:timestamp"].head())

# 5.2: get first and last timestamp per case
case_times = log.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"])
print(case_times.head())

# 5.3: calculate duration per case
case_times["duration"] = case_times["max"] - case_times["min"]
print(case_times["duration"].head())

# 5.4: calculate mean and standard deviation
avg_duration = case_times["duration"].mean()
std_duration = case_times["duration"].std()

print(f'Average duration: {avg_duration}')
print(f'Standard deviation of duration: {std_duration}')

# 5.5: converting to the required units
avg_days = avg_duration.total_seconds() / (60*60*24)
avg_minutes = avg_duration.total_seconds() / 60
avg_seconds = avg_duration.total_seconds()

std_days = std_duration.total_seconds() / (60*60*24)
std_minutes = std_duration.total_seconds() / 60
std_seconds = std_duration.total_seconds()

print(f'Average days: {avg_days}')
print(f'Average minutes: {avg_minutes}')
print(f'Average seconds: {avg_seconds}')
print(f'Std days: {std_days}')
print(f'Std minutes: {std_minutes}')
print(f'Std seconds: {std_seconds}')


# =========================================
# 6. ATTRIBUTES
# =========================================

# ASSIGNMENT: number of distinct case attribute labels; number of distinct event attribute labels; number of categorical event attributes
# IMPORTANT

# 6.1: identify case attributes
case_attributes = [col for col in log.columns if col.startswith("case:")]

print('Case attributes:')
print(case_attributes)
print(f'Number of case attribute labels: {len(case_attributes)}')

# 6.2: identify event attributes
event_attributes = [col for col in log.columns if not col.startswith("case:")]

print('Event attributes:')
print(event_attributes)
print(f'Number of event attribute labels: {len(event_attributes)}')

# 6.3: count categorical event attributes
categorical_event_attributes = [col for col in event_attributes if log[col].dtype == "object" or str(log[col].dtype) == "category"]

print('Categorical event attributes:')
print(categorical_event_attributes)
print(f'Number of categorical event attributes: {len(categorical_event_attributes)}')


# =========================================
# 7. ADDITIONAL STATISTICS
# =========================================

# IMPORTANT: I chose 3 different metrics since I believe they will be useful for further analysis in the next sections

# 7.1: most frequent activity
most_frequent_activity = log["concept:name"].value_counts().idxmax()
most_frequent_activity_frequence = log["concept:name"].value_counts().max()

print(f'Most frequent activity: {most_frequent_activity}')
print(f'Number of occurrences: {most_frequent_activity_frequence}')

# 7.2: number of resources
num_resources = log["org:resource"].nunique()

print(f'Number of resources: {num_resources}')

# 7.3: most frequent resource
most_frequent_resource = log["org:resource"].value_counts().idxmax()
most_frequent_resource_count = log["org:resource"].value_counts().max()

print(f'Most frequent resource: {most_frequent_resource}')
print(f'Number of events by this resource: {most_frequent_resource_count}')

# 7.4: create a statistics table with everything here
statistics = {
    "Number of events" : num_events,
    "Number of cases" : num_cases,
    "Number of activities" : num_activities,
    "Number of process variants" : num_variants,

    "Average case length" : round(avg_case_length, 2),
    "Standard deviation case length" : round(std_case_length, 2),

    "Average case duration (days)" : str(avg_days),
    "Average case duration (minutes)" : str(avg_minutes),
    "Average case duration (seconds)" : str(avg_seconds),

    "Standard deviation case duration" : str(std_duration),

    "Standard deviation case duration (days)" : str(std_days),
    "Standard deviation case duration (minutes)" : str(std_minutes),
    "Standard deviation case duration (seconds)" : str(std_seconds),

    "Number of case attribute labels" : len(case_attributes),
    "Number of event attribute labels" : len(event_attributes),
    "Number of categorical event attributes" : len(categorical_event_attributes),

    "Most frequent activity" : most_frequent_activity,
    "Occurrences of most frequent activity" : most_frequent_activity_frequence,

    "Number of resources" : num_resources,
    "Most frequent resource" : most_frequent_resource,
    "Events by most frequent resource" : most_frequent_resource_count,
}

statistics_table = pd.DataFrame(
    list(statistics.items()),
    columns = ["Metric", "Value"]
)

# 7.5: print the table
print(statistics_table)



