"""SParC interact match metric."""
import logging
from typing import Dict, Any
from third_party.test_suite import evaluation as cosql_evaluation


logger = logging.getLogger(__name__)

def compute_exact_match_metric(predictions, references) -> Dict[str, Any]:
    foreign_key_maps = dict()
    for reference in references:
        if reference["db_id"] not in foreign_key_maps:
            foreign_key_maps[reference["db_id"]] = cosql_evaluation.build_foreign_key_map(
                {
                    "table_names_original": reference["db_table_names"],
                    "column_names_original": list(
                        zip(
                            reference["db_column_names"]["table_id"],
                            reference["db_column_names"]["column_name"],
                        )
                    ),
                    "foreign_keys": list(
                        zip(
                            reference["db_foreign_keys"]["column_id"],
                            reference["db_foreign_keys"]["other_column_id"],
                        )
                    ),
                }
            )
    evaluator = cosql_evaluation.Evaluator(
        db_dir=references[0]["db_path"],
        kmaps=foreign_key_maps,
        etype="match",
        plug_value=False,
        keep_distinct=False,
        progress_bar_for_each_datapoint=False,
        )
    # Used on SParC and CoSQL
    turn_scores = {"exec": [], "exact": []}
    for prediction, reference in zip(predictions, references):
        
        turn_idx = reference.get("turn_idx", 0)
        # skip final utterance-query pairs
        if turn_idx < 0:
            evaluator.scores["joint_all"]["count"] += 1
            if turn_scores["exact"] != []:
                if all(v == 1 for v in turn_scores["exact"]):
                    evaluator.scores["joint_all"]["exact"] += 1
                turn_scores = {"exec": [], "exact": []}
            continue
        try:
            _ = evaluator.evaluate_one(
                reference["db_id"],
                reference["query"],
                prediction,
                turn_scores,
                idx=turn_idx,
            )
        except AssertionError as e:
            logger.warning(f"unexpected evaluation error: {e.args[0]}")
    evaluator.finalize()

    return {
        "exact_match": evaluator.scores["all"]["exact"],
        "interact_match": evaluator.scores["joint_all"]["exact"],
    }
