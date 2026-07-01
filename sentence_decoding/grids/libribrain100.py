"""Run sentence decoding on PNPL LibriBrain100.

Default split:
- train: sub-0, Sherlock1, sessions 1-10
- val:   sub-0, Sherlock1, session 11, run 2
- test:  sub-0, Sherlock1, session 12, run 2

For broader PNPL-defined splits, change ``data.query``. For example,
``subject == 'LibriBrain100/0' and corpus == 'sherlock'`` uses all
deep-subject Sherlock records with PNPL's train/val/test partitions.
"""

import os

from neuraltrain.utils import run_grid, update_config
from sentence_decoding.main import Experiment

from .defaults import DATADIR, default_config

GRID_NAME = "libribrain100"

LIBRIBRAIN100_PATH = os.getenv(
    "LIBRIBRAIN100_PATH",
    os.path.join(DATADIR or ".", "LibriBrain100"),
)

LIBRIBRAIN100_50_WORD_VOCABULARY = [
    "is",
    "the",
    "a",
    "to",
    "it",
    "i",
    "not",
    "was",
    "we",
    "be",
    "he",
    "that",
    "have",
    "this",
    "they",
    "of",
    "there",
    "and",
    "are",
    "in",
    "but",
    "will",
    "so",
    "all",
    "my",
    "for",
    "she",
    "were",
    "any",
    "really",
    "at",
    "out",
    "our",
    "am",
    "its",
    "had",
    "him",
    "an",
    "very",
    "has",
    "do",
    "can",
    "time",
    "think",
    "good",
    "always",
    "new",
    "people",
    "as",
    "on",
]

SHERLOCK1_SUB0_SESSIONS_1_10_11_12 = (
    "subject == 'LibriBrain100/0' "
    "and task == 'Sherlock1' "
    "and ("
    "(session == '1' and run == '1') "
    "or (session == '2' and run == '1') "
    "or (session == '3' and run == '1') "
    "or (session == '4' and run == '1') "
    "or (session == '5' and run == '1') "
    "or (session == '6' and run == '1') "
    "or (session == '7' and run == '1') "
    "or (session == '8' and run == '1') "
    "or (session == '9' and run == '1') "
    "or (session == '10' and run == '1') "
    "or (session == '11' and run == '2') "
    "or (session == '12' and run == '2')"
    ")"
)

update = {
    "infra.job_name": GRID_NAME,
    "infra.cluster": None,
    "infra.mode": "force",
    "data.dataset": "LibriBrain100",
    "data.data_path": LIBRIBRAIN100_PATH,
    "data.query": SHERLOCK1_SUB0_SESSIONS_1_10_11_12,
    "data.n_timelines": "all",
    "data.n_subjects": "all",
    "data.n_timelines_per_subject": "all",
    "data.event_type": "Word",
    "data.start": 0.0,
    "data.duration": 3.0,
    "use_transformer": True,
    "use_target_scaler": False,
    "trainer_config.monitor": "val_retrieval_acc10_vocab=libribrain50_macro_0",
    "retrieval_set_sizes": [None],
    "retrieval_vocabularies": {
        "libribrain50": LIBRIBRAIN100_50_WORD_VOCABULARY,
    },
    "retrieval_metrics": [
        metric
        for metric in default_config["retrieval_metrics"]
        if metric["name"] != "BERTScore"
    ],
}
updated_config = update_config(default_config, update)

grid = {
    "data.query": [SHERLOCK1_SUB0_SESSIONS_1_10_11_12],
    # PNPL-defined deep Sherlock split:
    # "data.query": ["subject == 'LibriBrain100/0' and corpus == 'sherlock'"],
    # PNPL-defined all-corpus split:
    # "data.query": ["subject == 'LibriBrain100/0'"],
}

if __name__ == "__main__":
    run_grid(
        Experiment,
        GRID_NAME,
        updated_config,
        grid,
        combinatorial=True,
        overwrite=False,
        dry_run=False,
        infra_mode="force",
    )
