# João Zuffo
# Process Mining Assignment - BPIC17 Analysis

# IMPORTANT: Since for the assignment we are supposed to address many different topics and requirements, I decided to first write a plan script based on each step
# The notes will later be exported to the final report


# =====================================================================================
# AUFGABE 3.4 - ADVANCED ANALYSIS   
# =====================================================================================

# GOAL: Predicting whether a case will have a long duration using machine learning and feature importance
# RQ: Can we predict early whether a loan application case will become a long-running case using machine learning?
# HY: Cases with more rework, more resources involved, and certain activity patterns are more likely to have longer durations

# Planned pipeline:
    # 1) Load the BPCI17 log
    # 2) Aggregate events into case-level features
    # 3) Compute case duration
    # 4) Define target: (1) = long case, (0) = short case
    # 5) Train/test split
    # 6) Train ML model
    # 7) Analyse feature importance

import pm4py 
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# =========================================
# 1. LOAD EVENT LOG & PREPARE THE SETUP
# =========================================

# 1.1: Loading the data log
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

# 1.2: Creating the folders
try:
    os.makedirs("../output/advanced", exist_ok=True)
    os.makedirs("../output/advanced/plots", exist_ok=True)
    os.makedirs("../output/advanced/tables", exist_ok=True)
    print("Folders created!")
except Exception as e:
    print("Unable to create the folders")
    print(e)
    sys.exit()


# =========================================
# 2. TIMESTAMP PREPARATION
# =========================================

# 2.2: Preparing the time stamps (long / short)
try:
    log["time:timestamp"] = pd.to_datetime(log["time:timestamp"])
    print("Time modified!")
except Exception as e:
    print("Unable to modify the time")
    print(e)
    sys.exit()


# =========================================
# 3. CREATE CASE-LEVEL FEATURES
# =========================================

# 3.1: Creating case-level table
try:
    case_features = log.groupby("case:concept:name").agg(
        case_start=("time:timestamp", 'min'),
        case_end=("time:timestamp", "max"),
        case_length=("concept:name", "count"),
        num_unique_activities=("concept:name", "nunique"),
        num_unique_resources=("org:resource", "nunique")
    ).reset_index()
    print("Case features created!")
except Exception as e:
    print("Unable to create case features")
    print(e)
    sys.exit()


# 3.2: Computing the case duration --> IN DAYS!! - explain the reason later on the report
try:
    case_features["case_duration_days"] = (case_features["case_end"] - case_features["case_start"]).dt.total_seconds() / (60 * 60 * 24)
    print("Case duration computed!")
except Exception as e:
    print("Unable to compute case duration")
    print(e)
    sys.exit()

print(case_features.head())


# =========================================
# 4. REWORK FEATURES
# =========================================
# IMPORTANT: Some activities could indicate rework -- should be considered differently as it could lead to loops i.e.

# 4.1: Rework features (repeated activities may indicate rework!!)
try:
    # 4.1.1: counting how often each acitivity appears in each case
    activity_counts = (log.groupby(["case:concept:name", "concept:name"]).size().reset_index(name="activity_count"))

    # 4.1.2: counting the number of repeated activities execution per case (if the activity appears 2 times there has been only 1 repetition!!)
    activity_counts["rework_count"] = activity_counts["activity_count"] - 1
    activity_counts["rework_count"] = activity_counts["rework_count"].clip(lower=0) # making sure there are no negatives here!!

    # 4.1.3: aggregating rework info at case level
    rework_features = activity_counts.groupby("case:concept:name").agg(total_rework = ("rework_count", "sum"),
                                                                       max_activity_repetition=("activity_count", "max"),
                                                                       num_repeated_activities=("rework_count", lambda x: (x > 0).sum())).reset_index()

    # 4.1.4: Merge with main case feature table
    case_features = case_features.merge(rework_features, on="case:concept:name", how="left")

    print("Reworks computed")
except Exception as e:
    print("Unable to compute reworks")
    print(e)
    sys.exit()

print(case_features[[
    "case:concept:name",
    "case_length",
    "total_rework",
    "max_activity_repetition",
    "num_repeated_activities"
]].head())


# =========================================
# 5. EARLY-CASE FEATURES (PREPARING THE DATA FOR THE ML)
# =========================================
# IMPORTANT: To make predictions more realistic, I will only consider the first 5 events of each case to conduct the prediction - avoiding 
# WHY 5? I tested with other and 5 seamed to the the most realistic - From the 3.2, we know the average length of the cases and 5 events represent around 15%!!
# CITATIONS: Take the info you got regarding this approach and add them to the report as sources!!! 

EARLY_EVENTS = 5 # I will reconsider this number to get the most realistic scenario regarding everything (considering the context, etc) !! BY ANALYSING THE FLOW OF THE PREVIOUS ANALYSIS

# 5.1: Creating the early log 
try:
    # Sorting log by case & timestamp
    log_sorted = log.sort_values(["case:concept:name", "time:timestamp"])

    # Event indexes
    log_sorted["event_index"] = log_sorted.groupby("case:concept:name").cumcount() + 1

    # KEEPING ONLY THE FIRST EARLY_EVENTS OF EACH CASE!!
    early_log = log_sorted[log_sorted["event_index"] <= EARLY_EVENTS].copy()

    print(f"Early log with {EARLY_EVENTS} created!")
except Exception as e:
    print("Unable to create the early log")
    print(e)
    sys.exit()

# 5.2: Creating the early features
try:
    early_features = early_log.groupby("case:concept:name").agg(
        early_case_length = ("concept:name", "count"),
        early_unique_activities = ("concept:name", "nunique"),
        early_unique_resources = ("org:resource", "nunique"),
        early_start = ("time:timestamp", "min"),
        early_end = ("time:timestamp", "max")
    ).reset_index()

    # since I am considering duration in days for the entire cases, by taking only a small sample of events from the cases, I will take the duration here in hours
    early_features["early_duration_hours"] = (early_features["early_end"] - early_features["early_start"]).dt.total_seconds() / (60 * 60)

    # dropping start and end from early_features since they are no longer needed for the research - duration was calculated already
    early_features = early_features.drop(columns=["early_start", "early_end"])

    case_features = case_features.merge(
        early_features,
        on="case:concept:name",
        how="left"
    )

    print("Early featrues computed!")

except Exception as e:
    print("Unable to compute early features!")
    print(e)
    sys.exit()

# testing results
print(case_features[[
    "case:concept:name",
    "early_case_length",
    "early_unique_activities",
    "early_unique_resources",
    "early_duration_hours"
]].head())


# =========================================
# 6. EARLY ACTIVITY PATTERNS
# =========================================
# GOAL: Understand how activities impact in the duration of cases 
# example: for cases where activity X is present early on, they are more likely to have longer durations (long-running)

# 6.1: Computing activities and checking how often they appear in the early events
try:
    # 6.1.1: Activity indicator for early events - creating dummies (0 and 1) to see if activity X appeared in case Y (>0 - True, 0 - False)
    early_activity_dummies = pd.crosstab(early_log["case:concept:name"], early_log["concept:name"])

    # 6.1.2: Transforming into binary indicators (better to work with):
    early_activity_dummies = (early_activity_dummies > 0).astype(int)

    # 6.1.3: Renaming columns for better understanding
    early_activity_dummies.columns = ["early_activity_" + str(col).replace(" ", "_").replace(":", "_") for col in early_activity_dummies.columns]

    case_features = case_features.merge(early_activity_dummies, on="case:concept:name", how="left")

    print("Early activity patterns computed!")

except Exception as e:
    print("Unable to compute early activity patterns!")
    print(e)
    sys.exit()


# =========================================
# 7. DEFINING LONG-RUNNING CASES AND TARGET VARIABLE
# =========================================
# IMPORTANT: I am considering long-running cases those cases that are above the 75% (in comparison to all others!) - 25% longer cases in other words - sounds reasnable when also further considering the first quartile the short-running, the two middle Qs as average duration

try:
    duration_threshold = case_features["case_duration_days"].quantile(0.75)

    case_features["long_case"] = (case_features["case_duration_days"] > duration_threshold).astype(int) # Transformint True/False in binary 

    print("Target variables created!")
    print(case_features["long_case"].value_counts())

except Exception as e:
    print("Unable to calculate target variables!")
    print(e)
    sys.exit()


# =========================================
# 8. FINAL DATASET PREPARATION FOR MACHINE LEARNING
# =========================================
# IMPORTANT: DISCUSS ABOUT THE DATA LEAKAGE YOU HAD (SINCE YOU ALSO CONSIDERED FULL-CASE FEATURES AND THEN YOU REMOVED THEM!!) - you can compare your results in both scenarios!!
try:
    # columns that wont be used as model features (wont be taken into account for the prediction)
    columns_to_drop = [
        "case:concept:name",
        "case_start",
        "case_end",
        "case_duration_days",
        "long_case",

        # Full-case features removed to avoid data leakage
        "case_length",
        "num_unique_activities",
        "num_unique_resources",
        "total_rework",
        "max_activity_repetition",
        "num_repeated_activities"
    ]

    X = case_features.drop(columns=columns_to_drop) # preparing to train
    Y = case_features["long_case"] # results

    # replacing empty values with 0
    X = X.fillna(0)
    
    print("Data for ML prepared")

except Exception as e:
    print("Unable to prepare data for ML")
    print(e)
    sys.exit()


# =========================================
# 9. TRAIN / TEST SPLIT
# =========================================
# IMPORTANT: explain why I used this ML practice!! + why the chosen split

try:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=123, stratify=Y)

    print("Train/Test completed!")
except Exception as e:
    print("Unable to test/split")
    print(e)
    sys.exit()


# =========================================
# 10. TRAIN RANDOM FOREST MODEL!!!
# =========================================
# IMPORTANT: explain the connection to the part 9
# IMPORTANT 2.0: explain why the values 

try:
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=123,
        class_weight="balanced",
        n_jobs=-1 # use everything
    )

    rf_model.fit(X_train, Y_train)

    print("Random forest model Trained!")

except Exception as e:
    print("Unable to train Random forest model!")
    print(e)
    sys.exit()


# =========================================
# 11. MODEL EVALUATION
# =========================================
# Understanding my resulting model
# IMPORTANT: Explain all metrics here!!

# 11.1: Calculating the accuracy and f1 score
try:
    Y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")

    print(classification_report(Y_test, Y_pred))

    print(confusion_matrix(Y_test, Y_pred))

except Exception as e:
    print("Unable to calculate the accuracy and the F1 score")
    print(e)
    sys.exit()

# 11.2: Saving classification report
try:
    report = classification_report(Y_test, Y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("../output/advanced/tables/classification_report.csv")

     # Save confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual short", "Actual long"],
        columns=["Predicted short", "Predicted long"]
    )
    cm_df.to_csv("../output/advanced/tables/confusion_matrix.csv")

    print("\nEvaluation tables exported!")

except Exception as e:
    print("Unable to evaluate model")
    print(e)
    sys.exit()


# =========================================
# 12. UNDERSTANDING THE IMPORTANCE OF EACH FEATURE TO THE ML MODEL
# =========================================

try: 
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": rf_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print(feature_importance.head(15))

    # Plot top 15 features
    top_features = feature_importance.head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Top 15 Feature Importances - Random Forest")
    plt.tight_layout()
    plt.savefig("../output/advanced/plots/random_forest_feature_importance.png", dpi=300)
    plt.close()

    print("Random Forest feature importance exported!")

except Exception as e:
    print("Unable to export Random Forest feature importance")
    print(e)
    sys.exit()