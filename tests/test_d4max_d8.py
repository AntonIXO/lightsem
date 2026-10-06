from lightsem.experiments.d4max_d8.motifs import prune_end_states

def test_duplicate_paths_cannot_evict_a_distinct_future_state():
    candidates=[(0,.1,[1,2],2),(0,.2,[3,2],2),(1,0.,[1,5],5)]
    kept=prune_end_states(candidates)
    assert [p[3] for p in kept]==[2,5]
    assert kept[0]==candidates[0]
    # If only donor5 can join the final motif, the valid path survives.
    assert [p for p in kept if p[3]==5]==[candidates[2]]

def test_lexicographic_cost_and_tie_breaking_match_exhaustive_state_minimum():
    import random
    rng=random.Random(7001)
    paths=[(rng.randrange(10),rng.random(),[i],rng.randrange(4)) for i in range(1000)]
    result=prune_end_states(paths)
    assert len(result)==4
    for p in result:assert p==min([x for x in paths if x[3]==p[3]],key=lambda x:x[:3])
