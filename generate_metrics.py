import sqlite3
import sqlparse
import os
import re
import time
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

start_time = time.time()


# Load gold SQL queries
def load_gold_sql_queries(gold_sql_file: str) -> List[Tuple[str, str]]:
    """Loads (SQL query, db_id) pairs from the gold file."""
    queries = []
    with open(gold_sql_file, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                queries.append((parts[0], parts[1]))  # (SQL query, db_id)
    print(f"Loaded {len(queries)} queries from {gold_sql_file}")
    return queries


# Extract SQL sample indices
def extract_sql_indices(log_file_path: str) -> List[int]:
    """Extracts sample indices from log files."""
    with open(log_file_path, "r", encoding="utf-8") as log_file:
        log_content = log_file.read()

    pattern = re.findall(
        r"Train sample (\d+) generation STARTED\s+(.*?)\s+Train sample \1 generation FINISHED",
        log_content, re.DOTALL
    )

    sample_indices = [int(sample_index) for sample_index, _ in pattern]
    return sample_indices


# Read predicted SQL queries
def read_sql_file(file_path: str, sample_indices: List[int]) -> List[Tuple[int, str]]:
    """Reads SQL queries from a file and pairs them with their sample indices."""
    queries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sql_queries = [line.strip() for line in f.readlines() if line.strip()]

    assert len(sql_queries) == len(sample_indices), "Mismatch in queries and indices"
    return list(zip(sample_indices, sql_queries))  # Pairing (sample_index, SQL query)


# Normalize SQL Query
def normalize_sql(query: str) -> str:
    """Normalize SQL formatting."""
    if query is None:
        return ''
    return sqlparse.format(query, reindent=True, keyword_case='lower').strip()


import sqlite3
import threading

def execute_sql(db_path: str, query: str, timeout: int = 60):
    """Executes an SQL query with a timeout.
    
    - Returns query results if successful.
    - Returns "TIMEOUT" if execution exceeds `timeout` seconds.
    - Returns "OTHER" for any other errors.
    """

    def run_query(result_container):
        """Executes the SQL query and stores result in a shared list."""
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                result_container.append(cursor.fetchall())
        except Exception:
            result_container.append("OTHER")

    # Shared list to store result from the thread
    result_container = []
    
    # Run the SQL query in a separate thread
    thread = threading.Thread(target=run_query, args=(result_container,))
    thread.start()
    
    # Wait for the thread to complete within the timeout
    thread.join(timeout)

    # If thread is still alive after timeout, return "TIMEOUT"
    if thread.is_alive():
        return "TIMEOUT"

    return result_container[0] if result_container else "OTHER"


# Process a Single Sample
def process_sample(idx: int, db_base_path: str):
    """Computes Exact Match (EM) and Execution Match (EX) for a given SQL query."""
    #print("Index:",idx)
    gold_query = idx['SQL']
    db_id = idx['db_id']
    db_path = os.path.join(db_base_path, db_id, f"{db_id}.sqlite")
    pred_query = idx['Predicted_SQL']
    print("GOld:",gold_query)
    print("PRed:",pred_query)
    print("DB path:",db_path)
    # Exact Match (EM)
    em_success = int(normalize_sql(gold_query) == normalize_sql(pred_query))

    # Execution Match (EX)
    gold_result = execute_sql(db_path, gold_query)
    pred_result = execute_sql(db_path, pred_query)
    if gold_result == "TIMEOUT" or gold_result == "OTHER" or pred_result == "TIMEOUT" or pred_result == "OTHER":
        ex_success = "ERROR"
    else:
        ex_success = int(gold_result == pred_result)

    return (em_success, db_id), (ex_success, db_id), idx['question_id']


import logging

# Configure logging
log_file = '/home/schatterjee1/KBLAM/Results/sqlcoder_metrics_Key_Vals_SAME_EMBED_SPACE.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def compute_em_ex_batched(db_base_path: str, predicted_sql_queries: List[Tuple[int, str]], batch_size=10):
    """Computes Exact Match (EM) and Execution Match (EX) in batches and logs the aggregated results."""
    exact_match_results = {}
    execution_match_results = {}
    execution_error_results = {}
    execution_failure_indices = []

    total_samples = len(predicted_sql_queries)

    print("TOTAL SAMPLES:",total_samples)
    batch_count = (total_samples + batch_size - 1) // batch_size  # Calculate total batches

    for batch_idx in range(batch_count):
        batch_queries = predicted_sql_queries[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        print("Batch queries:",len(batch_queries))
        log_message = f"Processing batch {batch_idx + 1}/{batch_count}..."
        print(log_message)
        logging.info(log_message)

        with ThreadPoolExecutor(max_workers=min(128, len(batch_queries))) as executor:
            future_to_sample = {executor.submit(process_sample, idx, db_base_path): idx for idx in batch_queries}

            for future in as_completed(future_to_sample):
                (em_success, db_id), (ex_success, db_id), index = future.result()
                
                # Update EM and EX counts per db_id
                exact_match_results[db_id] = exact_match_results.get(db_id, 0) + em_success
                if ex_success != "ERROR":
                    execution_match_results[db_id] = execution_match_results.get(db_id, 0) + ex_success
                    if ex_success == 0:
                        execution_failure_indices.append(index)

                else:
                    execution_error_results[db_id] = execution_error_results.get(db_id, 0) + 1


        # Log aggregated results after each batch
        log_message = f"Batch {batch_idx + 1} Results - \n \
        EM: {exact_match_results}, \n \
        EX: {execution_match_results}, \n \
        Failure Indices: {execution_failure_indices}, \n \
        Error: {execution_error_results} "
        logging.info(log_message)

        # Flush logs
        with open(log_file, "a") as log_f:
            log_f.write("\n")
            log_f.flush()

    return exact_match_results, execution_match_results, execution_error_results, execution_failure_indices







#gold_sql_file = '/home/schatterjee1/Datasets/BirdSQL/train/train_gold.sql'

#sql_file_path = '/home/schatterjee1/CodeLutra/predicted_SQL_queries_SINGLE.sql'
db_base_path = '/home/schatterjee1/Datasets/BirdSQL/dev/dev_databases'


# gold_sql_queries = load_gold_sql_queries(gold_sql_file)
# sample_indices = extract_sql_indices(log_file_path)
# predicted_sql_queries = read_sql_file(sql_file_path, sample_indices)

import json


# Open and load the JSON file
with open("/home/schatterjee1/KBLAM/SQLs_Key_Vals_SAME_EMBED_SPACE.json", 'r') as f:
    dev_data = json.load(f)
print("Total samples:",len(dev_data))
exact_match_results, execution_match_results, execution_error_results, execution_failure_indices = compute_em_ex_batched(db_base_path, dev_data, batch_size=1024)
logging.info("Final results:")
logging.info("Exact Match:\n")
logging.info(exact_match_results)
logging.info("Execution Match:\n")
logging.info(execution_match_results)
logging.info("Execution Failure Indices:\n")
logging.info(execution_failure_indices)
logging.info("Execution Errors:\n")
logging.info(execution_error_results)
elapsed_time = (time.time() - start_time) / 60
print("--- %.2f minutes ---" % elapsed_time)
