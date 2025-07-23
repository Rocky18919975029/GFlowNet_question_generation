# utils/qa_logger.py

import os
from datetime import datetime

class QALogger:
    """
    A minimal, DDP-safe logger to save generated question-answer pairs
    to a simple, human-readable text file for qualitative inspection.
    """
    def __init__(self, log_path: str):
        self.log_path = log_path
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Write a header to the file to signify a new run
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"NEW RUN STARTED AT: {datetime.now().isoformat()}\n")
            f.write("="*80 + "\n\n")
        
        print(f"--- QALogger initialized. Q&A pairs will be saved to: {self.log_path} ---")

    def log_qa_batch(
        self,
        global_step: int,
        subject_text: str,
        questions: list,
        answers: list,
        n_probes_to_log: int
    ):
        """
        Formats and appends a batch of Q&A pairs to the log file.
        """
        log_content = []
        log_content.append(f"--- Step: {global_step}, Subject: {subject_text} ---")
        
        for i in range(min(n_probes_to_log, len(questions))):
            log_content.append(f"  Q: {questions[i]}")
            # Check if answers list is long enough
            answer_text = answers[i] if i < len(answers) else "[Answer not available]"
            log_content.append(f"  A: {answer_text}")
        
        log_content.append("") # Add a blank line for readability

        # Append all content for this batch to the file
        if log_content:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(log_content) + "\n")