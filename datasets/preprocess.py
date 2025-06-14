import json
import os
import argparse
import logging
from collections import Counter
from typing import List, Dict, Any, Optional, Set
from scipy.stats import pointbiserialr
import random


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Utility Functions ---

def load_json(path: str) -> List[Dict[str, Any]]:
    """Loads a JSON file from the given path."""
    try:
        with open(path, "r", encoding="utf-8") as fin:
            obj = json.load(fin)
        logging.info(f"Successfully loaded JSON from {path}")
        return obj
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at {path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {path}. Please check file format.")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {path}: {e}")
        raise

def dump_json(obj: Any, path: str):
    """Dumps a Python object to a JSON file at the given path."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True) # Ensure output directory exists
        with open(path, "w", encoding="utf-8") as fout:
            json.dump(obj, fout, indent=2)
        logging.info(f"Successfully saved JSON to {path}")
    except Exception as e:
        logging.error(f"Error saving JSON to {path}: {e}")
        raise

# --- Main Question Processor Class ---

class QuestionProcessor:
    """
    A class to preprocess question data, calculate psychometric dimensions
    (difficulty, discrimination, distractor efficiency), and find question pairs.
    """

    REQUIRED_QUESTION_KEYS: Set[str] = {
        "q_id", "learning_mat_id", "q_text", "answer", "learning_mat", "topic", "stud_perf"
    }
    REQUIRED_STUD_PERF_KEYS: Set[str] = {"accuracy", "s_id", "choice"}
    MIN_STUDENT_ANSWERS_FOR_DISCRIMINATION: int = 10

    def __init__(self, questions_data: List[Dict[str, Any]]):
        """
        Initializes the QuestionProcessor with raw question data.
        Validates the input data structure.
        """
        if not isinstance(questions_data, list):
            raise TypeError("Input 'questions_data' must be a list of dictionaries.")
        if not questions_data:
            logging.warning("Input 'questions_data' is empty. No questions to process.")

        self.all_questions = questions_data
        self.perf_by_stud: Dict[str, List[str]] = {}
        self.total_perf_by_stud: Dict[str, float] = {}

        self._validate_input_questions()

    def _validate_input_questions(self):
        """Internal method to validate the structure of the initial question data."""
        logging.info("Validating input question data structure...")
        for i, q in enumerate(self.all_questions):
            # Check top-level question keys
            if not isinstance(q, dict):
                raise ValueError(f"Question at index {i} is not a dictionary.")
            if not self.REQUIRED_QUESTION_KEYS.issubset(q.keys()):
                missing_keys = self.REQUIRED_QUESTION_KEYS - q.keys()
                raise ValueError(f"Question (ID: {q.get('q_id', 'N/A')}) at index {i} is missing required keys: {missing_keys}")

            # Check 'stud_perf' structure
            stud_perf_list = q.get('stud_perf', [])
            if not isinstance(stud_perf_list, list):
                raise ValueError(f"Question (ID: {q.get('q_id', 'N/A')}) at index {i} has 'stud_perf' that is not a list.")
            for j, stud_perf in enumerate(stud_perf_list):
                if not isinstance(stud_perf, dict):
                    raise ValueError(f"Student performance entry at Q {i}, S {j} is not a dictionary.")
                if not self.REQUIRED_STUD_PERF_KEYS.issubset(stud_perf.keys()):
                    missing_keys = self.REQUIRED_STUD_PERF_KEYS - stud_perf.keys()
                    raise ValueError(f"Student performance entry at Q {i}, S {j} is missing required keys: {missing_keys}")
        logging.info("Input question data validated successfully.")


    def calculate_difficulty(self):
        """Calculates the difficulty (proportion of correct answers) for each question."""
        logging.info("Calculating question difficulty...")
        for q in self.all_questions:
            stud_perf_correct = [1 for stud in q['stud_perf'] if stud.get('accuracy') == "correct"]
            if not stud_perf_correct: # Handle cases with no students or no correct answers
                q['diff'] = 0.0
                logging.warning(f"Question {q['q_id']} has no correct student performances. Difficulty set to 0.")
            else:
                q['diff'] = sum(stud_perf_correct) / len(q['stud_perf'])
        logging.info("Difficulty calculation complete.")

    def _prepare_student_performance_data(self):
        """Internal method to aggregate student performance data across all questions."""
        logging.info("Aggregating student performance data...")
        for q in self.all_questions:
            for stud in q['stud_perf']:
                s_id = stud.get('s_id')
                accuracy = stud.get('accuracy')
                if s_id is not None and accuracy is not None:
                    if s_id not in self.perf_by_stud:
                        self.perf_by_stud[s_id] = []
                    self.perf_by_stud[s_id].append(accuracy)
                else:
                    logging.warning(f"Skipping student performance entry for Q {q.get('q_id', 'N/A')} due to missing 's_id' or 'accuracy'.")

        for s_id, s_perf_list in self.perf_by_stud.items():
            if s_perf_list:
                correct_count = len([acc for acc in s_perf_list if acc == "correct"])
                self.total_perf_by_stud[s_id] = correct_count / len(s_perf_list)
            else:
                self.total_perf_by_stud[s_id] = 0.0
                logging.warning(f"Student {s_id} has no valid performance entries.")
        logging.info("Student performance data aggregation complete.")

    def calculate_discrimination(self):
        """Calculates the discrimination index for each question using Point-Biserial Correlation."""
        if pointbiserialr is None:
            logging.error("Scipy (required for discrimination calculation) is not installed. Skipping discrimination calculation.")
            for q in self.all_questions:
                q['disc'] = None # Set to None if calculation is skipped
            return

        logging.info("Calculating question discrimination...")
        self._prepare_student_performance_data()

        for q in self.all_questions:
            corr_x = [] # Binary score for the current question (1 if correct, 0 if incorrect)
            corr_y = [] # Total test score for the student
            
            for stud in q['stud_perf']:
                s_id = stud.get('s_id')
                accuracy = stud.get('accuracy')
                
                if accuracy is not None and s_id is not None and s_id in self.total_perf_by_stud:
                    corr_x.append(1 if accuracy == "correct" else 0)
                    corr_y.append(self.total_perf_by_stud[s_id])
                else:
                    logging.warning(f"Skipping student {s_id} for discrimination on Q {q.get('q_id', 'N/A')} due to missing data.")

            if len(corr_x) < self.MIN_STUDENT_ANSWERS_FOR_DISCRIMINATION:
                logging.warning(f"Question {q['q_id']} has fewer than {self.MIN_STUDENT_ANSWERS_FOR_DISCRIMINATION} student answers. Discrimination not calculated.")
                q['disc'] = 0.0 # Or None, depending on desired behavior for insufficient data
            elif len(set(corr_x)) > 1: # Ensure there's variance in performance on this question
                disc = pointbiserialr(corr_x, corr_y)[0]
                q['disc'] = disc
            else:
                logging.info(f"Question {q['q_id']} has no variance in student answers (all correct/all incorrect). Discrimination set to 0.")
                q['disc'] = 0.0 # If all students got it right or all got it wrong, discrimination is 0.
        logging.info("Discrimination calculation complete.")

    def calculate_distractor_efficiency(self):
        """
        Calculates distractor efficiency based on distractors chosen by at least 5% of students.
        """
        logging.info("Calculating distractor efficiency...")
        for q in self.all_questions:
            # Filter choices to only include distractors (not the correct answer) that are strings
            choices = [
                stud['choice'] for stud in q['stud_perf']
                if stud.get('choice') != q.get('answer') and isinstance(stud.get('choice'), str)
            ]
            distractors_counts = Counter(choices)

            min_count_for_5_percent = len(q['stud_perf']) * 0.05
            
            # An efficient distractor is one chosen by at least 5% of students (excluding the correct answer)
            eff_dist = {
                dist: count for dist, count in distractors_counts.items()
                if count >= min_count_for_5_percent
            }
            q['disteff'] = len(eff_dist)
        logging.info("Distractor efficiency calculation complete.")

    def find_pairs(self, alpha: float, dim: str) -> List[tuple]:
        """
        Finds pairs of questions from the processed data that meet a
        specified difference (alpha) in a given dimension (dim).

        Args:
            alpha (float): The minimum absolute difference required between
                           the two questions' dimension values.
            dim (str): The dimension to compare (e.g., "diff", "disc", "disteff").

        Returns:
            List[tuple]: A list of tuples, where each tuple contains the
                         (q_id1, q_id2) of a matching pair.
        """
        logging.info(f"Finding pairs for dimension '{dim}' with alpha >= {alpha}...")
        possible_pairs = []
        # Create a set to store pairs to avoid duplicates (regardless of order)
        seen_pairs = set()

        for q1 in self.all_questions:
            # Skip if the dimension is missing in q1
            if dim not in q1 or q1[dim] is None:
                logging.warning(f"Question {q1.get('q_id', 'N/A')} is missing dimension '{dim}'. Skipping.")
                continue

            for q2 in self.all_questions:
                # Ensure it's not the same question and the dimension is present in q2
                if q1['q_id'] == q2['q_id'] or dim not in q2 or q2[dim] is None:
                    continue

                # Create a canonical representation of the pair to check for duplicates
                # This handles (q1_id, q2_id) and (q2_id, q1_id) as the same pair
                current_pair = tuple(sorted((q1['q_id'], q2['q_id'])))

                # Check conditions: same learning material, difference threshold, and not already seen
                if (q1['learning_mat_id'] == q2['learning_mat_id'] and
                    abs(q1[dim] - q2[dim]) >= alpha and
                    current_pair not in seen_pairs):

                    possible_pairs.append((q1['q_id'], q2['q_id']))
                    seen_pairs.add(current_pair) # Add the canonical form to the seen set
        logging.info(f"Found {len(possible_pairs)} pairs for dimension '{dim}'.")
        return possible_pairs
    
    def assemble_pair_data(self, possible_pairs: List[tuple], dim: str) -> List[Dict[str, Any]]:
        """
        Assembles detailed data for each question pair based on a specific dimension.

        For each pair, it randomly shuffles the order, extracts relevant question details,
        and determines a 'label' indicating which question is preferred based on the 'dim'.

        Args:
            possible_pairs (List[tuple]): A list of tuples, where each tuple contains
                                           the (q_id1, q_id2) of a matching pair.
            dim (str): The dimension used for pairing (e.g., "diff", "disc", "disteff").

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  represents a processed pair with all required details.

        Raises:
            ValueError: If a question ID in a pair is not found in the processed data.
            AssertionError: If learning material IDs or learning materials don't match for a pair.
        """
        logging.info(f"Assembling pair data for dimension '{dim}'...")
        pair_data = []

        # Create a mapping from q_id to question object for efficient lookup
        q_id_to_question = {q['q_id']: q for q in self.all_questions}

        # Define requirement descriptions outside the loop for clarity and efficiency
        requirement_definitions = {
            "diff": {
                "requirement": "the question that is easier to answer",
                "description": "An easier question has a higher proportion of students with a correct answer."
            },
            "disc": {
                "requirement": "the question that has high discrimination",
                "description": "A question with higher discrimination is more effective at distinguishing between high-performing and low-performing students."
            },
            "disteff": {
                "requirement": "the question that has a higher number of effective distractors",
                "description": "An effective distractor is one that is chosen by at least 5% of the students taking the quiz."
            }
        }

        if dim not in requirement_definitions:
            logging.error(f"Unknown dimension '{dim}' provided for pair assembly. Skipping.")
            return [] # Return empty if dimension is not recognized

        dim_info = requirement_definitions[dim]

        for pair_ids in possible_pairs:
            # Randomly shuffle the order of q_ids in the pair
            shuffled_pair_ids = list(pair_ids)
            random.shuffle(shuffled_pair_ids)
            first_q_id, second_q_id = shuffled_pair_ids[0], shuffled_pair_ids[1]

            # Retrieve full question objects using the mapping
            q1 = q_id_to_question.get(first_q_id)
            q2 = q_id_to_question.get(second_q_id)

            if q1 is None or q2 is None:
                logging.warning(f"Skipping pair ({first_q_id}, {second_q_id}): One or both question IDs not found in processed data.")
                continue

            # Ensure learning materials match (these are crucial for your pairing logic)
            if q1.get('learning_mat_id') != q2.get('learning_mat_id') or \
               q1.get('learning_mat') != q2.get('learning_mat'):
                logging.error(f"Assertion failed for pair ({first_q_id}, {second_q_id}): Learning material mismatch. This indicates an issue with `find_pairs` or input data. Skipping.")
                continue # Or you could raise AssertionError here if this state is truly unexpected

            l_id = q1.get('learning_mat_id')
            learning_mat = q1.get('learning_mat')

            # Determine the 'label' based on the dimension values
            label = "1" # Default to question 1 being "better" (e.g., easier, higher discrimination)
            # Ensure the dimension values exist and are comparable before comparison
            dim_val_q1 = q1.get(dim)
            dim_val_q2 = q2.get(dim)

            if dim_val_q1 is None or dim_val_q2 is None:
                logging.warning(f"Skipping pair ({first_q_id}, {second_q_id}): Missing '{dim}' value for one or both questions. Cannot determine label.")
                continue
            if dim_val_q1 < dim_val_q2: # q2 has higher discrimination/disteff, so label 2
                label = "2"

            pair_data.append({
                'test_id': f"{first_q_id}_{second_q_id}", # Use original (not shuffled) IDs for consistent test_id
                'l_id': l_id,
                'learning_mat': learning_mat,
                'question_1': q1['q_text'],
                'question_2': q2['q_text'],
                'label': label, # This implies q_1 meets requirement if 1, q_2 if 2
                'requirement': dim_info['requirement'],
                'requirement_description': dim_info['description']
            })
        logging.info(f"Assembled {len(pair_data)} pairs for dimension '{dim}'.")
        return pair_data
       

    def process_all(self):
        """Executes all preprocessing steps."""
        self.calculate_difficulty()
        self.calculate_discrimination()
        self.calculate_distractor_efficiency()
        logging.info("All preprocessing calculations complete.")

        return self.all_questions # Return the enriched data

# --- Main Execution Block ---

def main():
    """
    Main function to parse arguments, load data, process questions,
    and output results.
    """
    logging.info("Starting question preprocessing script...")

    parser = argparse.ArgumentParser(
        description="Preprocesses question data to calculate psychometric dimensions and find question pairs."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSON file containing raw question data."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed",
        help="Path to the directory where output JSON files will be saved (default: 'processed_data')."
    )
    parser.add_argument(
        "--diff_alpha",
        type=float,
        default=0.15,
        help="Alpha value for difficulty-based pair finding (default: 0.15)."
    )
    parser.add_argument(
        "--disc_alpha",
        type=float,
        default=0.15,
        help="Alpha value for discrimination-based pair finding (default: 0.15)."
    )
    parser.add_argument(
        "--disteff_alpha",
        type=float,
        default=2.0,
        help="Alpha value for distractor efficiency-based pair finding (default: 2.0)."
    )
    args = parser.parse_args()

    # Load data
    try:
        raw_questions = load_json(args.input_file)
    except Exception: # load_json already logs specifics
        logging.critical("Exiting due to error loading input file.")
        return # Exit if loading fails

    # Process questions
    try:
        processor = QuestionProcessor(raw_questions)
        processed_questions = processor.process_all()
    except Exception as e:
        logging.critical(f"Error during question processing: {e}. Exiting.")
        return
    # Find and report pairs
    difficulty_pairs = processor.find_pairs(args.diff_alpha, "diff")
    discrimination_pairs = processor.find_pairs(args.disc_alpha, "disc")
    disteff_pairs = processor.find_pairs(args.disteff_alpha, "disteff")

    logging.info(f"Number of difficulty-based pairs found: {len(difficulty_pairs)}")
    logging.info(f"Number of discrimination-based pairs found: {len(discrimination_pairs)}")
    logging.info(f"Number of distractor efficiency-based pairs found: {len(disteff_pairs)}")

    # Assemble pair data
    all_pair_data = []
    all_pair_data.extend(processor.assemble_pair_data(difficulty_pairs, "diff"))
    all_pair_data.extend(processor.assemble_pair_data(discrimination_pairs, "disc"))
    all_pair_data.extend(processor.assemble_pair_data(disteff_pairs, "disteff"))
    
    logging.info(f"Total {len(all_pair_data)} question pairs assembled across all dimensions.")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True) # Ensure the output directory exists

    processed_questions_file = os.path.join(output_dir, "processed_questions.json")
    assembled_pairs_file = os.path.join(output_dir, "pair_dataset.json")

    # Save the processed questions data
    try:
        dump_json(processed_questions, processed_questions_file)
    except Exception:
        logging.critical(f"Exiting due to error saving processed questions data to {processed_questions_file}.")
        return

    # Save the assembled pair data
    try:
        dump_json(all_pair_data, assembled_pairs_file)
    except Exception:
        logging.critical(f"Exiting due to error saving assembled pair data to {assembled_pairs_file}.")
        return

    logging.info("Preprocessing complete.")



if __name__ == "__main__":
    main()