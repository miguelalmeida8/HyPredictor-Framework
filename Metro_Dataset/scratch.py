import tkinter as tk
from tkinter import messagebox


# Function to apply the rule and update predictions
def apply_rule():
    global test_predictions_with_rules, y_test
    condition = (x_test['median_tp3'] * x_test['median_h1'] < 500) & \
                (x_test['median_oil_temperature'] < 65) & \
                ((x_test['median_tp3'] - x_test['median_h1'] < 1) & (test_predictions_with_rules == 1))
    test_predictions_with_rules[condition] = 0

    # Calculate evaluation metrics with the rule applied
    test_accuracy_with_rule = accuracy_score(y_test, test_predictions_with_rules)
    test_precision_with_rule = precision_score(y_test, test_predictions_with_rules)
    test_recall_with_rule = recall_score(y_test, test_predictions_with_rules)
    test_f1_score_with_rule = f1_score(y_test, test_predictions_with_rules)

    # Show a message box with the updated evaluation metrics
    messagebox.showinfo("Rule Applied", f"Test Accuracy with Rule: {test_accuracy_with_rule}\n"
                                        f"Test Precision with Rule: {test_precision_with_rule}\n"
                                        f"Test Recall with Rule: {test_recall_with_rule}\n"
                                        f"Test F1 Score with Rule: {test_f1_score_with_rule}")


# Create the main window
root = tk.Tk()
root.title("Rule Application Interface")

# Add a button to apply the rule
apply_rule_button = tk.Button(root, text="Apply Rule", command=apply_rule)
apply_rule_button.pack()

# Run the Tkinter event loop
root.mainloop()
