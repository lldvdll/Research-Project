"""Probes: ask what is still in the hidden layer, ignoring the output layer entirely.

The nearest-class-mean (NCM) probe is the decisive diagnostic in knowledge_base.md §4.6.
Discard the output layer, classify by whichever class's average hidden pattern is closest.

    argmax accuracy is LOW but NCM accuracy is HIGH  -> the hidden code survived; the damage
                                                        is in the output layer (calibration)
    both LOW                                          -> the hidden code was destroyed
                                                        (representation drift)

Prototypes are built from TRAINING data of every class seen so far, which is information the
network would not have during a task-free stream. That is deliberate: this is a measurement
instrument, not a continual-learning method.
"""
import torch


def class_prototypes(features_fn, train_data, class_idx, classes, per_class=100,
                     device="cpu", seed=0):
    """{class: mean hidden vector} from `per_class` training images of each class."""
    g = torch.Generator().manual_seed(int(seed))
    protos = {}
    for c in classes:
        pool = torch.as_tensor(class_idx[c])
        pick = pool[torch.randperm(len(pool), generator=g)[:per_class]]
        x = torch.stack([train_data[i][0] for i in pick.tolist()]).to(device)
        protos[c] = features_fn(x).detach().mean(0)
    return protos


def ncm_predict_fn(features_fn, protos):
    """fn(x) -> predicted class labels by nearest prototype (Euclidean)."""
    classes = sorted(protos)
    P = torch.stack([protos[c] for c in classes])                  # [n_classes, hidden]
    lookup = torch.tensor(classes)

    def predict(x):
        f = features_fn(x).detach()                                 # [batch, hidden]
        d = torch.cdist(f, P.to(f.device))                          # [batch, n_classes]
        return lookup.to(f.device)[d.argmin(1)]

    return predict


def cosine_ncm_predict_fn(features_fn, protos):
    """Direction-only variant: removes the magnitude degree of freedom that suppression corrupts."""
    classes = sorted(protos)
    P = torch.stack([torch.nn.functional.normalize(protos[c], dim=0) for c in classes])
    lookup = torch.tensor(classes)

    def predict(x):
        f = torch.nn.functional.normalize(features_fn(x).detach(), dim=1)
        return lookup.to(f.device)[(f @ P.to(f.device).t()).argmax(1)]

    return predict


def live_ncm_fn(features_fn, proto_x, proto_y, cosine=False):
    """fn(x) -> labels, rebuilding prototypes from a FIXED image set at the CURRENT weights.

    Use this as a readout inside run_classil: prototypes must track the drifting hidden code,
    otherwise you are measuring staleness rather than representation quality. Costs one extra
    forward pass over `proto_x` per evaluation.
    """
    classes = sorted(set(proto_y.tolist()))
    lookup = torch.tensor(classes)

    def predict(x):
        pf = features_fn(proto_x).detach()
        P = torch.stack([pf[proto_y == c].mean(0) for c in classes])
        f = features_fn(x).detach()
        if cosine:
            f = torch.nn.functional.normalize(f, dim=1)
            P = torch.nn.functional.normalize(P, dim=1)
            return lookup.to(f.device)[(f @ P.t()).argmax(1)]
        return lookup.to(f.device)[torch.cdist(f, P).argmin(1)]

    return predict


def prototype_images(train_data, class_idx, classes, per_class=50, device="cpu", seed=0):
    """Fixed image set for live_ncm_fn. Drawn once, shared by every method and condition."""
    g = torch.Generator().manual_seed(int(seed))
    xs, ys = [], []
    for c in classes:
        pool = torch.as_tensor(class_idx[c])
        pick = pool[torch.randperm(len(pool), generator=g)[:per_class]]
        xs.append(torch.stack([train_data[i][0] for i in pick.tolist()]))
        ys.append(torch.full((len(pick),), c))
    return torch.cat(xs).to(device), torch.cat(ys).to(device)


def frozen_ncm_fn(features_fn, protos_holder):
    """fn(x) -> labels using prototypes SNAPSHOT at some earlier moment.

    Contrast with live_ncm_fn, and note the difference carefully:
      live prototypes   -> "is the class information still linearly decodable?"  Any consistent
                           transformation of the hidden space is INVISIBLE, because the
                           prototypes rotate with it.
      frozen prototypes -> "has the code MOVED away from where it was?"  This is the one that
                           actually measures representation drift.
    `protos_holder` is a mutable dict {class: vector}, filled by a callback at the task switch.
    """
    def predict(x):
        if not protos_holder:
            return torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
        classes = sorted(protos_holder)
        P = torch.stack([protos_holder[c] for c in classes])
        f = features_fn(x).detach()
        return torch.tensor(classes, device=f.device)[torch.cdist(f, P.to(f.device)).argmin(1)]
    return predict


def restricted_argmax_fn(predict, classes):
    """argmax over ONLY the classes used in this experiment.

    With out_dim=10 but four classes in play, six output units are never a target. Under a
    masked loss they are also never suppressed, so they keep their random initial weights and
    can capture the argmax spuriously -- which penalises the masked condition and nothing
    else. Restricting the argmax removes that artefact. Report both.
    """
    idx = torch.tensor(sorted(classes))

    def fn(x):
        out = predict(x, raw=True).detach()
        return idx.to(out.device)[out[:, idx.to(out.device)].argmax(1)]
    return fn


def code_snapshot(features_fn, x):
    """Hidden code for a fixed image set, for drift measurement."""
    return features_fn(x).detach().clone()


def code_drift(before, after):
    """How far the hidden code moved. Two complementary numbers:
         cosine    mean cosine similarity per image (1.0 = direction unchanged)
         rel_l2    mean ||after - before|| / ||before||  (0.0 = identical)
    A rigid rotation gives low cosine but may leave live-NCM accuracy untouched, which is
    exactly why both are reported."""
    cos = torch.nn.functional.cosine_similarity(before, after, dim=1).mean().item()
    rel = ((after - before).norm(dim=1) / (before.norm(dim=1) + 1e-9)).mean().item()
    return dict(cosine=cos, rel_l2=rel)


def saturation(features_fn, x, thresh=0.95):
    """Fraction of hidden units with |activation| above `thresh`. EqProp's main failure mode:
       once tanh flattens, f'(h) -> 0 and the nudge can no longer reach the hidden layer."""
    f = features_fn(x).detach()
    return (f.abs() > thresh).float().mean().item()


def output_unit_stats(predict, x):
    """Per-output-unit mean raw score. Watch an absent class's score drift DOWN during a task
       it does not belong to -- this is logit suppression, made visible."""
    with torch.no_grad():
        out = predict(x, raw=True)
    return dict(mean=out.mean(0).tolist(), std=out.std(0).tolist())


# ------------------------------------------------------------------ update probes
# Both wrap train_step rather than hooking the runner, so a measurement is opt-in per run and
# the training loop stays the same for every rule. Pass them as run(..., wrap=...).

def alignment_probe(train_step, predict, arch, obj, device="cpu", every=1, ref=None):
    """Wrap train_step to record TARGET ALIGNMENT for each update. [R1] Fig 3b.

        d_target  = target - out_before
        d_learn   = out_after - out_before      <- the SECOND forward pass sees no target
        alignment = cos(d_target, d_learn)

    In words: the update moved the output somewhere. Alignment asks how much of that movement
    was toward where the target actually was. 1.0 = straight at it, 0.0 = sideways, negative =
    away. This is [R1]'s own measure of whether a rule configures itself prospectively, so it
    tests their mechanism claim on our networks rather than restating it.

    `ref=(x_ref, y_ref)` MEASURES THE SAME COSINE ON A FIXED BATCH THAT IS NOT BEING TRAINED.
    Point it at task-1 data during task 2 and it becomes an INTERFERENCE measure: how much of
    each update's movement, on data the update was not computed from, goes toward that data's
    own targets. Positive means the update happens to help task 1 as well; negative means it
    actively pushes task 1's outputs away, which is forgetting caught per update rather than
    inferred from an endpoint. That is the version this project needs: it is a rate, so unlike
    every accuracy metric it does not inherit the training budget.

    `every=k` measures on every k-th update only. Alignment is averaged over many updates, so
    subsampling costs precision in the mean and nothing else -- and it is what makes the probe
    affordable for EqProp, where `predict` is a full relaxation rather than a forward pass.

    Returns (wrapped_train_step, log) or, when `ref` is given, (wrapped, log, ref_log). Each
    log is a list of (update_index, mean-over-batch value), so subsampled runs still say WHEN
    each measurement was taken and can be aligned to the task switch.

    COST: two extra forward passes per measured update, or four with `ref`. Negligible for
    backprop and PC. For EqProp each one is a settling, so use `every` there.
    """
    from .model import make_target

    log, ref_log, step = [], [], [0]
    cos = torch.nn.functional.cosine_similarity
    if ref is not None:
        x_ref, y_ref = ref
        tgt_ref = make_target(y_ref, arch, obj, device=device)

    def _measure(x, tgt, out_before):
        with torch.no_grad():
            out_after = predict(x, raw=True).detach()
        return cos(tgt - out_before, out_after - out_before, dim=-1).mean().item()

    def wrapped(x, y, active=None):
        i = step[0]
        step[0] = i + 1
        take = (i % every == 0)
        if not take:
            return train_step(x, y, active=active)
        with torch.no_grad():
            out_before = predict(x, raw=True).detach().clone()
            ref_before = (predict(x_ref, raw=True).detach().clone()
                          if ref is not None else None)
        train_step(x, y, active=active)
        tgt = make_target(y, arch, obj, device=device)
        log.append((i, _measure(x, tgt, out_before)))
        if ref is not None:
            ref_log.append((i, _measure(x_ref, tgt_ref, ref_before)))

    return (wrapped, log, ref_log) if ref is not None else (wrapped, log)


def weight_path_probe(train_step, params):
    """Wrap train_step to accumulate the L1 PATH LENGTH of every weight. [R31] Li & van Rossum.

    Their metabolic cost is M = sum_i sum_t |w_i(t) - w_i(t-1)| -- how far each synapse actually
    travelled, counting every reversal. Compare it against |w_i(T) - w_i(0)|, how far it needed
    to travel, and the ratio is INEFFICIENCY: 1.0 is a straight line, higher is more wandering.
    See metrics.inefficiency, which takes what this returns.

    Returns (wrapped_train_step, path) where `path` is a dict {name: tensor} of accumulated
    |dw|, same shape as the weights, filling as the run proceeds. Per-synapse, so the
    DISTRIBUTION is available and not just the mean -- that distribution is the object of
    interest in scripts 47-49.

    COST: one clone and one subtraction per weight per update. Negligible.
    """
    path = {k: torch.zeros_like(v) for k, v in params.named().items() if v is not None}

    def wrapped(x, y, active=None):
        before = {k: v.detach().clone() for k, v in params.named().items() if v is not None}
        train_step(x, y, active=active)
        for k, v in params.named().items():
            if v is not None:
                path[k] += (v.detach() - before[k]).abs()

    return wrapped, path
