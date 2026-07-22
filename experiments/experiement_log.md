Equilibrium Propagation [EqProp] & Catastrophic Forgetting [CF]
01: Vanilla EqProp, ClassIL problem, MNIST 14x14, 196x64x10, one class per task (10x1), Backprop / Replay as controls
    Hypothesis: Does EqProp overcome CF?
    EqProp forgets immediately
02: As above, but 2 classes per task (5x2)
    Question: Is CF caused by the algorithm or the experiment - is learning individual tasks the problem?
    Again Eqprop forgets immediately
    This is faster so will continue with it.
03: As above, "gating", or selectively update only hidden nodes that move MOST under the nudge
    Hypothesis: Active nodes most likely to encode previous task. If we freeze them do we mitigate CF?
04: Eqprop combined with replay
    Question: Is Eqprop better with replay? Is replay worse with Eqprop?
05: Generate samples from Eqprop model
    Question: Can Eqprop be used to generate synthetic samples from previous tasks
06: Eqprop combined with replay, samples synthetically generated from the model, so no storage
    Question: Can generated samples be used for replay?
