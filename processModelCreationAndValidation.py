# João Zuffo
# BPIC-17 Process Model Creation and Validation

# IMPORTANT: Since for the assignment we are supposed to address many different topics and requirements, I decided to first write a plan script based on each step
# The notes will later be exported to the final report
# FOR THE REPORT: Due to the computational complexity of alignment-based conformance checking on large-scale event logs such as BPIC-17, token-based replay was used during intermediate experimentation, while alignment-based fitness was reserved for the final model evaluation.

# Therefore:
# - Alpha Miner is kept mainly as a baseline/reference model
# - Inductive Miner is used as the main discovery algorithm because it produces sound and more structured workflow models
# - Token-based replay is used during intermediate evaluation for scalability and iterative refinement
# - Alignment-based replay is reserved only for the final refined model evaluation
#
# This trade-off between scalability, precision, fitness, and simplicity will later be discussed
# in detail in the final report.


# =====================================================================================
# AUFGABE 3.3 - PROCESS MODEL CREATION AND VALIDATION
# =====================================================================================

import pm4py
import pandas as pd
import sys
import os
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator


# =========================================
# 1. LOAD EVENT LOG
# =========================================

# IMPORTANT: This is the same update as in basicAnalysis 

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
# 2. CREATE OUTPUT FOLDERS
# =========================================

# IMPORTANT: Save the discovered process models - to be included in the appendix of the report, save quality metrics in tables
# Two different files: (1) output/models --> for Petri net / BPMN model images & output/metrics --> for csv and latex metric tables 

try:
    os.makedirs("../output/models", exist_ok=True)
    os.makedirs("../output/metrics", exist_ok=True)
    print("Output folders created")
except Exception as e:
    print("Unable to create output folders")
    print(e)
    sys.exit()


# =========================================
# 3. DISCOVER INITIAL PROCESS MODELS
# =========================================

# IMPORTANT: Apply at least two different process discovery algorithms and add the resulting process to the appendix
    # 1) Alpha Miner: Footprint matrix & baseline model, however, may produce unsound nets on complex real-world logs
    # 2) Inductive Miner: Block-structured discovery algorithm; more robust for complex logs because it produces sound and structured workflow models

# NAMING:
    # im and fm = initial and final marking
    # net = structure (places, transitions, arcs)

# 3.1: Alpha Miner
try: 
    net_alpha, im_alpha, fm_alpha = pm4py.discover_petri_net_alpha(log)
    print("Alpha Miner Discovered!")
except:
    print("Unable to discover model with Alpha Miner")
    sys.exit()

# 3.2: Inductive Miner
try: 
    net_inductive, im_inductive, fm_inductive = pm4py.discover_petri_net_inductive(log)
    print("Inductive Miner Discovered!")
except:
    print("Unable to discover model with Inductive Miner")
    sys.exit()


# =========================================
# 4. EXPORT INITIAL PROCESS MODELS
# =========================================

# IMPORTANT: Both Petri Nets will be exported to include them in the appendix!!

try:
    pm4py.save_vis_petri_net(net_alpha, im_alpha, fm_alpha, "../output/models/alpha_miner.png")
    pm4py.save_vis_petri_net(net_inductive, im_inductive, fm_inductive, "../output/models/inductive_miner.png")
    print("Initial process models exported!!")

except Exception as e:
    print("Unable to export initial models")
    print(e)
    sys.exit()


# =========================================
# 4. MY METRICS
# =========================================

# IMPORTANT: The metrics chosen were:
    # 1) Model Size --> Helps to better understand the simplicity, fitting, understanding and complexity of the model (A smaller model is structurally simpler and easier to understand)
    # 2) Arc/Connection/Arrow Density --> Mostly for relation intermodel: a lower arc density means the model is less densely connected and therefore may be simpler!!

# 4.1: Model Size = # places + # transitions + # arcs/arrows
def calculate_model_size(petri_net):
    return len(petri_net.places) + len(petri_net.transitions) + len(petri_net.arcs)

# 4.2: Arc Density = (# arcs) / (# nodes) with # nodes = # places + # transitions
def calculate_arc_density(petri_net):
    number_of_nodes = len(petri_net.places) + len(petri_net.transitions)
    if number_of_nodes == 0: return 0
    else: return len(petri_net.arcs) / number_of_nodes


# =========================================
# 5. THE 4 BASIC QUALITY METRICS 
# =========================================

# 5.1: Fitness - Using Alignment-Based --> To be later debated in the report why not the Token-Based replay approach
try:
    fitness_alpha = replay_fitness_evaluator.apply(log, net_alpha, im_alpha, fm_alpha, variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED) # IMPORTANT!!!!
    print("Fitness for Alpha Miner done!")
except:
    print("Unable to compute fitness for Alpha Miner")
    sys.exit()
try:
    fitness_inductive = replay_fitness_evaluator.apply(log, net_inductive, im_inductive, fm_inductive, variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
    print("Fitness for Inductive Miner done!")
except:
    print("Unable to compute fitness for Inductive Miner")
    sys.exit()

# 5.2: Precision - using algorithm given in the lecture
try:
    precision_alpha = pm4py.precision_token_based_replay(log, net_alpha, im_alpha, fm_alpha)
    print("Precision for Alpha Miner done")
except:
    print("Unable to compute precision for Alpha Miner")
    sys.exit()
try:
    precision_inductive = pm4py.precision_token_based_replay(log, net_inductive, im_inductive, fm_inductive)
    print("Precision for Inductive Miner done")
except:
    print("Unable to compute precision for Inductive Miner")
    sys.exit()

# 5.3: Generalization - algorithm from pm4py
try:
    generalization_alpha = pm4py.generalization_tbr(log, net_alpha, im_alpha, fm_alpha)
    print("Generalization for Alpha Miner computed!")
except:
    print("Unable to compute generalization for Alpha Mienr")
    sys.exit()
try:
    generalization_inductive = pm4py.generalization_tbr(log, net_inductive, im_inductive, fm_inductive)
    print("Generalization for Inductive Miner computed!")
except:
    print("Unable to compute generalization for Inductive Mienr")
    sys.exit()  

# 5.4: Simplicity - algorithm from pm4py
# arc_degree --> 
try:
    simplicity_alpha = pm4py.simplicity_petri_net(net_alpha, im_alpha, fm_alpha, variant="arc_degree")
    print("Simplicity for Alpha Miner computed!")
except Exception as e:
    print("Unable to compute simplicity for Alpha Miner")
    print(e)
    sys.exit()
try:
    simplicity_inductive = pm4py.simplicity_petri_net(net_inductive, im_inductive, fm_inductive, variant="arc_degree")
    print("Simplicity for Inductive Miner computed")
except Exception as e:
    print("Unable to compute simplicity for Inductive Miner")
    print(e)
    sys.exit()

# 5.4: My own metrics using part 4: "My metrics"
# 5.4.1: Model size
try:
    model_size_alpha = calculate_model_size(net_alpha)
    print("Model Size for Alpha Miner computed!")
except Exception as e:
    print("Unable to calculate Model Size for Alpha Miner")
    print(e)
    sys.exit()
try:
    model_size_inductive = calculate_model_size(net_inductive)
    print("Model Size for Inductive Miner computed!")
except Exception as e:
    print("Unable to calculate Model Size for Inductive Miner")
    print(e)
    sys.exit()
# 5.4.2: Arc Density
try:
    arc_density_alpha = calculate_arc_density(net_alpha)
    print("Arc Density for Alpha Miner computed")
except Exception as e:
    print("Unable to compute Arc Density for Alpha Miner")
    print(e)
    sys.exit()
try:
    arc_density_inductive = calculate_arc_density(net_inductive)
    print("Arc Density for Inductive Miner computed")
except Exception as e:
    print("Unable to compute Arc Density for Inductive Miner")
    print(e)
    sys.exit()


# =========================================
# 6. FILTER LOG TO APROX. 15% OF THE CASES
# =========================================

# Main Idea: Instead of modeling every rare variant, I thought of keeping the most frequent variants up to the point that it reaches around 80% of all cases
# Create the model through that would enable a controlled fitness + avoiding including rare cases leading to more precise and generic models (trade-off)

# GOAL: Keep the most frequent variants until they cover app 15% of the cases --> REDUCING NOISE

# 6.1: Variant sequence per case + frequency
try:
    variants_per_case = log.groupby("case:concept:name")["concept:name"].apply(tuple).reset_index()
    variants_per_case.columns = ["caseID", "variant"]
    variant_counts = variants_per_case["variant"].value_counts().reset_index()
    variant_counts.columns = ["variant", "frequency"]
    print("Variant Sequence per case and Frequency counted!")
except Exception as e:
    print("Unable to calculate Variant Sequence per case and Frequency!")
    print(e)
    sys.exit()

# 6.2: Computing cumultative % of cases covered
try:
    total_cases = variant_counts["frequency"].sum()
    variant_counts["cumultative_cases"] = variant_counts["frequency"].cumsum()
    variant_counts["cumultative_percentage"] = variant_counts["cumultative_cases"] / total_cases
    print("Percentages computed!")
except Exception as e:
    print("Unable to compute the percentage")
    print(e)
    sys.exit()

# 6.3: Keep variants for around 15% of all cases 
try:
    selected_variants = variant_counts[variant_counts["cumultative_percentage"] <= 0.15]["variant"]

    # Safety --> include first variant above 15% if necessary!! (avoid being significantly less than 80%)
    if len(selected_variants) < len(variant_counts):
        selected_variants = variant_counts.iloc[:len(selected_variants) + 1]["variant"]
    
    print("Computed all cases up to around 0.15 of the frequency of the log")

except Exception as e:
    print("Unable to compute 0.15 of all cases")
    print(e)
    sys.exit()

# 6.5: Select cases beloging to the variants above + filter original log
try:
    selected_cases = variants_per_case[variants_per_case["variant"].isin(selected_variants)]["caseID"]
    filtered_log = log[log["case:concept:name"].isin(selected_cases)]
    print("Log filtered for around 15 percent of all the cases")
except Exception as e:
    print("Unable to filter the log")
    print(e)
    sys.exit()


# =========================================
# 7. DISCOVER FINAL MODEL
# =========================================
# IMPORTANT: Using the above filtered log
# The final model is discovered using Inductive Miner on the filtered log.

try:
    net_final, im_final, fm_final = pm4py.discover_petri_net_inductive(filtered_log)
    print("Final process discovered!")
except Exception as e:
    print("Unable to discover the final process")
    print(e)
    sys.exit()

# SAVING THE FINAL PROCESS
try:
    pm4py.save_vis_petri_net(net_final, im_final, fm_final, "../output/models/final_model_filtered_15.png")
    print("Final model exported")
except Exception as e:
    print("Unable to export final model")
    print(e)
    sys.exit()


# =========================================
# 8. APPLYING THE METRICS TO THE FINAL MODEL
# =========================================

# 8.1: Fitness
try:
    fitness_final = replay_fitness_evaluator.apply(filtered_log, net_final, im_final, fm_final, variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
    print("Fitness for final done")
except Exception as e:
    print("Unable to compute fitness for final")
    print(e)
    sys.exit()

# 8.2: Precision - using algorithm given in the lecture
try:
    precision_final = pm4py.precision_token_based_replay(filtered_log, net_final, im_final, fm_final)
    print("Precision for Final done")
except Exception as e:
    print("Unable to compute precision for Final")
    print(e)
    sys.exit()

# 8.3: Generalization - algorithm from pm4py
try:
    generalization_final = pm4py.generalization_tbr(filtered_log, net_final, im_final, fm_final)
    print("Generalization for Final computed!")
except Exception as e:
    print("Unable to compute generalization for Final")
    print(e)
    sys.exit()

# 8.4: Simplicity - algorithm from pm4py
try:
    simplicity_final = pm4py.simplicity_petri_net(net_final, im_final, fm_final, variant="arc_degree")
    print("Simplicity for Final computed!")
except Exception as e:
    print("Unable to compute simplicity for Final")
    print(e)
    sys.exit()

# 8.4: My own metrics using part 4: "My metrics"
# 8.4.1: Model size
try:
    model_size_final = calculate_model_size(net_final)
    print("Model Size for Final computed!")
except Exception as e:
    print("Unable to calculate Model Size for Final")
    print(e)
    sys.exit()

# 8.4.2: Arc Density
try:
    arc_density_final = calculate_arc_density(net_final)
    print("Arc Density for Final computed")
except Exception as e:
    print("Unable to compute Arc Density for Final")
    print(e)
    sys.exit()


# =========================================
# 9. COMPARING THE FINAL MODEL TO THE 2 OTHERS
# =========================================

# After testing, I created the following function to prevent confusion and crashing of the code

def get_fitness_value(fitness_result):
    if "averageFitness" in fitness_result:
        return fitness_result["averageFitness"]
    elif "average_trace_fitness" in fitness_result:
        return fitness_result["average_trace_fitness"]
    elif "log_fitness" in fitness_result:
        return fitness_result["log_fitness"]
    else:
        print("Unknown fitness keys:", fitness_result.keys())
        return None
    
try:
    comparison_df = pd.DataFrame({

        "Model": [
            "Alpha Miner",
            "Inductive Miner",
            "Final Filtered Model"
        ],

        "Fitness": [
            get_fitness_value(fitness_alpha),
            get_fitness_value(fitness_inductive),
            get_fitness_value(fitness_final)
        ],

        "Precision": [
            precision_alpha,
            precision_inductive,
            precision_final
        ],

        "Generalization": [
            generalization_alpha,
            generalization_inductive,
            generalization_final
        ],

        "Simplicity": [
            simplicity_alpha,
            simplicity_inductive,
            simplicity_final
        ],

        "Model Size": [
            model_size_alpha,
            model_size_inductive,
            model_size_final
        ],

        "Arc Density": [
            arc_density_alpha,
            arc_density_inductive,
            arc_density_final
        ]
    })

    print("Comparison table created!")
    print(comparison_df)

except Exception as e:
    print("Unable to create comparison table")
    print(e)
    sys.exit()


# Exporting metric tables
try:
    comparison_df.to_csv(
        "../output/metrics/model_comparison.csv",
        index=False
    )
    print("Metric tables exported!")
    
except Exception as e:
    print("Unable to export metric tables")
    print(e)
    sys.exit()


# =========================================
# 10. BPMN EXPORT
# =========================================

try:
    bpmn_model = pm4py.convert_to_bpmn(net_final, im_final, fm_final)
    pm4py.save_vis_bpmn(bpmn_model, "../output/models/final_bpmn.png")
    print('BPMN successfully exported!')
except Exception as e:
    print('Unable to export BPMN')
    print(e)
    sys.exit()