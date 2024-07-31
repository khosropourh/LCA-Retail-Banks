import pandas as pd
import stepmix
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r"C:\Users\khosropour\Desktop\research\Bank_data.csv")

# Convert data to binary format (required for LCA)
binary_data = pd.get_dummies(data).values

# Perform LCA on all data
smm_all = stepmix.StepMix(n_components=3)
smm_all_fit = smm_all.fit(binary_data)

print("LCA Results (All Data):")
print("Number of classes:", smm_all.n_components)
class_probs = smm_all.predict_proba(binary_data)
print("Class probabilities:", class_probs)

# Perform LCA on each category separately
structure_vars = ['Chief Digital Officer', 'Digital Unit', 'Steering Committee', 'Innovation Center', 'Innovation Labs', 'Venture Capital (VC)']
structure_data = data[structure_vars]
binary_structure_data = pd.get_dummies(structure_data).values

smm_structure = stepmix.StepMix(n_components=3)
smm_structure_fit = smm_structure.fit(binary_structure_data)

print("LCA Results (Structure):")
print("Number of classes:", smm_structure.n_components)
class_probs = smm_structure.predict_proba(binary_structure_data)
print("Class probabilities:", class_probs)

process_vars = ['Evaluation and Improvement', 'Digital Product', 'Changing Business Model', 'Pioneer', 'Following Others', 'Central Investment', 'Local Investment', 'Co-Investments with Partners']
process_data = data[process_vars]
binary_process_data = pd.get_dummies(process_data).values

smm_process = stepmix.StepMix(n_components=3)
smm_process_fit = smm_process.fit(binary_process_data)

print("LCA Results (Processes):")
print("Number of classes:", smm_process.n_components)
class_probs = smm_process.predict_proba(binary_process_data)
print("Class probabilities:", class_probs)

connection_vars = ['Inter-Unit Collaboration', 'Multi-Function Teams', 'Collocating Business and Technical Units', 'Co-Creation with Customers', 'Partnerships with Technological Companies']
connection_data = data[connection_vars]
binary_connection_data = pd.get_dummies(connection_data).values

smm_connection = stepmix.StepMix(n_components=3)
smm_connection_fit = smm_connection.fit(binary_connection_data)

print("LCA Results (Connections):")
print("Number of classes:", smm_connection.n_components)
class_probs = smm_connection.predict_proba(binary_connection_data)
print("Class probabilities:", class_probs)

# Plot the class probabilities for each category
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.bar(range(smm_structure.n_components), smm_structure.predict_proba(binary_structure_data).mean(axis=0))
plt.title("Structure")
plt.xlabel("Class")
plt.ylabel("Probability")

plt.subplot(1, 3, 2)
plt.bar(range(smm_process.n_components), smm_process.predict_proba(binary_process_data).mean(axis=0))
plt.title("Processes")
plt.xlabel("Class")
plt.ylabel("Probability")

plt.subplot(1, 3, 3)
plt.bar(range(smm_connection.n_components), smm_connection.predict_proba(binary_connection_data).mean(axis=0))
plt.title("Connections")
plt.xlabel("Class")
plt.ylabel("Probability")

plt.show()
