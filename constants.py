# TODO make this a class
DATASET_VALS = {
    'glue':
        {
            'sst2':
                {
                    'prefix': 'sst2 sentence: ',
                    'labels': {
                        0: 'negative',
                        1: 'positive',
                        -1: 'other',
                    },
                },

            'cola':
                {
                    'prefix': 'cola sentence: ',
                    'labels': {
                        0: 'unacceptable',
                        1: 'acceptable',
                        -1: 'other',
                    },
                },
            'mnli':
                {
                    'prefix_1': 'mnli hypothsis: ',
                    'prefix_2': 'preminse: ',
                    'labels': {
                        2: 'contradiction',
                        1: 'entailment',
                        0: 'neutral',
                        -1: 'other',
                    },
                },

        },

}
