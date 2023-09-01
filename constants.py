COLUMN_NAMES = ['input_ids', 'labels']


DATASET_VALS = {
    'glue':
        {
            'sst2':
                {
                    'prefix': 'sst2 sentence: ',
                    'text_column_name': 'sentence',
                    'label_column_name': 'label',
                    'metric_to_optimize': 'accuracy',
                    'greater_is_better': True,
                    'metric_names': ['accuracy', 'f1', 'precision', 'recall'],
                    'labels': {
                        0: 'negative',
                        1: 'positive',
                        -1: 'other',
                    },
                },

            'cola':
                {
                    'prefix': 'cola sentence: ',
                    'text_column_name': 'sentence',
                    'label_column_name': 'label',
                    'metric_to_optimize': 'mcc',
                    'greater_is_better': True,
                    'metric_names': ['accuracy', 'f1', 'precision', 'recall', 'mcc'],
                    'labels': {
                        0: 'unacceptable',
                        1: 'acceptable',
                        -1: 'other',
                    },
                },
            'rte':
                {
                    'prefix_1': 'rte sentence1: ',
                    'prefix_2': 'sentence2: ',
                    'text_column_name_1': 'sentence1',
                    'text_column_name_2': 'sentence2',
                    'label_column_name': 'label',
                    'metric_to_optimize': 'accuracy',
                    'greater_is_better': True,
                    'metric_names': ['accuracy', 'f1', 'precision', 'recall'],
                    'labels': {
                        0: 'entailment',
                        1: 'not_entailment',
                        -1: 'other',
                    },
                },

            'mnli':
                {
                    'prefix_1': 'mnli hypothesis: ',
                    'prefix_2': 'premise: ',
                    'text_column_name_1': 'premise',
                    'text_column_name_2': 'hypothesis',
                    'label_column_name': 'label',
                    'metric_to_optimize': 'accuracy',
                    'greater_is_better': True,
                    'metric_names': ['accuracy', 'f1', 'precision', 'recall'],
                    'labels': {
                        2: 'contradiction',
                        0: 'entailment',
                        1: 'neutral',
                        -1: 'other',
                    },
                },
            'qnli':
                {
                    'prefix_1': 'qnli question: ',
                    'prefix_2': 'sentence: ',
                    'text_column_name_1': 'question',
                    'text_column_name_2': 'sentence',
                    'label_column_name': 'label',
                    'metric_to_optimize': 'accuracy',
                    'greater_is_better': True,
                    'metric_names': ['accuracy', 'f1', 'precision', 'recall'],
                    'labels': {
                        1: 'not_entailment',
                        0: 'entailment',
                        -1: 'other',
                    },
                },
            'mrpc':
                {
                    'prefix_1': 'mrpc sentence1: ',
                    'prefix_2': 'sentence2: ',
                    'text_column_name_1': 'sentence1',
                    'text_column_name_2': 'sentence2',
                    'label_column_name': 'label',
                    'metric_to_optimize': 'f1',
                    'greater_is_better': True,
                    'metric_names': ['accuracy', 'f1', 'precision', 'recall'],
                    'labels': {
                        0: 'not_equivalent',
                        1: 'equivalent',
                        -1: 'other',
                    },
                },
            'qqp':
                {
                    'prefix_1': 'qqp question1: ',
                    'prefix_2': 'question2: ',
                    'text_column_name_1': 'question1',
                    'text_column_name_2': 'question2',
                    'label_column_name': 'label',
                    'metric_to_optimize': 'accuracy',
                    'greater_is_better': True,
                    'metric_names': ['accuracy', 'f1', 'precision', 'recall'],
                    'labels': {
                        0: 'not_duplicate',
                        1: 'duplicate',
                        -1: 'other',
                    },
                },
            'stsb':
                {
                    'prefix_1': 'stsb sentence1: ',
                    'prefix_2': 'sentence1: ',
                    'text_column_name_1': 'sentence1',
                    'text_column_name_2': 'sentence2',
                    'label_column_name': 'label',
                    'metric_to_optimize': 'spearman',
                    'greater_is_better': True,
                    'metric_names': ['spearman', 'pearson', 'accuracy', 'f1', 'precision', 'recall'],
                    'labels': None,
                },
        },
}
