from sklearn.metrics import classification_report, roc_auc_score

def cls_metric(ground_truth_labels, predict_labels):
    cls_report = classification_report(
        y_true = ground_truth_labels, 
        y_pred = predict_labels,
        labels = [0, 1, 2, 3],
        target_names = ["easy", "medium", "hard", "extra-hard"],
        digits = 4, 
        output_dict = True
    )
    
    return cls_report


def auc_metric(ground_truth_labels, predict_probs):
    auc_score = roc_auc_score(ground_truth_labels, predict_probs)
    
    return auc_score