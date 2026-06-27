# Global Failure Audit

## finqa

- Retrieval top1=0.6417, top3=0.8675, miss=152

- Generation buckets: `{'grounded_wrong': 159, 'ungrounded_correct': 97, 'ungrounded_wrong': 874, 'grounded_correct': 17}`

- Main retrieval misses by task: `{'ratio': 67, 'sum': 18, 'percent_change': 17, 'average': 16, 'lookup': 14, 'difference': 11, 'adjustment': 6, 'factor_sum': 3}`

## convfinqa

- Retrieval top1=0.7279, top3=0.9274, miss=251

- Generation buckets: `{'grounded_wrong': 753, 'grounded_correct': 617, 'ungrounded_wrong': 1864, 'ungrounded_correct': 224}`

- Main retrieval misses by task: `{'lookup': 93, 'sum': 52, 'difference': 52, 'ratio': 27, 'average': 17, 'percent_change': 6, 'adjustment': 4}`

## tatqa

- Retrieval top1=0.326, top3=0.6198, miss=435

- Generation buckets: `{'ungrounded_wrong': 770, 'grounded_wrong': 214, 'ungrounded_correct': 81, 'grounded_correct': 79}`

- Main retrieval misses by task: `{'difference': 108, 'lookup': 96, 'percent_change': 71, 'average': 49, 'ratio': 45, 'sum': 32, 'comparison': 30, 'factor_sum': 3, 'adjustment': 1}`
