"""
Comparison of two totally ordered event logs through embeddings.

A trace is a sequence of activities <a_1, ..., a_k>, totally ordered by time:
one event, one activity, no ties.

The embeddings of the c context activities are CONCATENATED and the next activity
is predicted.

The two logs are compared by the compass method: one model is fitted on the
joint log and its E_out kept as the compass, then the model is retrained per
log with E_out frozen. The resulting embeddings are aligned, so their cosine
locates a difference.

Vocabulary convention -- ONE padding index, ``PAD_IDX = 0``, in every embedding
matrix:
    start marker -> 0      the zero vector, via padding_idx=0
    activity_i   -> i+1
    end marker   -> |A|+1  a candidate target only, never an input
so E_in holds the ids 0..|A| and E_out the candidate targets Act(L) u {end},
the ids 1..|A|+1 shifted down by ``TARGET_OFFSET``.

Public API
----------
    compare_cwindow_event_logs   two event logs
                                 -> (model_i, model_j, compass)
    activity_similarity          cross-log similarity per activity
    save_model / load_model      one model per file, at a given path

A model carries its own vocabulary and Act(L), so comparing, inspecting and
saving need nothing but the model.
"""

import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from tqdm.auto import tqdm

PAD_IDX = 0  # the ONLY padding index, in every embedding matrix
START_TOKEN = "#START#"  # the start marker |> of Definition "Training sample"
END_TOKEN = "#END#"  # the end marker [] of the same definition

# E_in is indexed by ``word_to_ix``, E_out by the CANDIDATE TARGETS Act(L) u {[]}
# alone: the start marker is padding, never a target, so it has no output column.
# Shifting the input id by this offset gives the target's column in E_out.
TARGET_OFFSET = 1


# some useful helpers


def set_seed(seed=None):
    """Seed Python, NumPy and PyTorch.

    :param seed: ``None`` draws one from the OS, independently of any earlier seeding.
    :returns: the seed applied, so a random run stays reproducible.
    """
    if seed is None:
        seed = random.SystemRandom().randrange(2**32)  # numpy caps seeds at 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _get_device(device=None):
    """The device to train on.

    :param device: ``None`` picks CUDA, then MPS, then CPU.
    :returns: the :class:`torch.device`.
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _activities_of(traces):
    """Act(L), the activities occurring in a log.

    :param traces: the log, as ``{trace: count}``.
    :returns: the set of activities.
    """
    return {a for trace in traces for a in trace}


# combining and balancing two logs for the compass stage


def _balance_case_counts(traces, balance):
    """Balance the logs' contribution to the compass stage.

    Where the log sizes differ substantially, the imbalance is corrected so that
    neither log dominates the compass. We reweight the case counts rather than
    downsampling the larger log.

    :param traces: ``{log name: log}``.
    :param balance: ``False`` returns the logs unchanged, every factor 1.0.
    :returns: ``(rescaled logs, per-log factors)``; the counts turn fractional.
    """
    if not balance:
        return dict(traces), dict.fromkeys(traces, 1.0)
    target_n = max(sum(t.values()) for t in traces.values())
    factors = {n: target_n / sum(t.values()) for n, t in traces.items()}
    return (
        {
            n: Counter({trace: count * factors[n] for trace, count in t.items()})
            for n, t in traces.items()
        },
        factors,
    )


def _combine_logs(traces):
    """The joint log L_1 + L_2 the compass stage is trained on.

    :param traces: ``{log name: log}``.
    :returns: one log, adding up the counts of a shared trace variant.
    """
    combined = Counter()
    for t in traces.values():
        for trace, count in t.items():
            combined[trace] += count
    return combined


# training functions


def _plot_loss(losses, title):
    """Plot a run's per-epoch loss.

    :param losses: the per-epoch loss, in order.
    :param title: the plot's title.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("cross entropy loss")
    plt.title(title)
    plt.grid(True)
    plt.show()


def _fit(
    model,
    contexts,
    targets,
    weights,
    *,
    learning_rate,
    epochs,
    device,
    verbose,
    desc="",
):
    """Fit a model on encoded tensors, full batch. Only parameters left with
    ``requires_grad`` are updated.

    :param model: the model, fitted in place.
    :param contexts: the context activities, ``[N, c]``.
    :param targets: the target of each sample, as an E_out column, ``[N]``.
    :param weights: each sample's multiplicity in the training multiset, ``[N]``.
    :param learning_rate: the Adam learning rate.
    :param epochs: how many full-batch steps.
    :param device: the device to train on.
    :param verbose: show the progress bar.
    :param desc: its label.
    :returns: the per-epoch cross-entropy of Definition "CWINDOW embedding model".
        The step descends the weighted mean, which differs from it by the constant
        total weight, so the gradient is the same either way.
    """
    contexts, targets, weights = (
        contexts.to(device),
        targets.to(device),
        weights.to(device),
    )

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss(reduction="none")

    epoch_losses = []
    n_samples = contexts.shape[0]
    progress_bar = tqdm(range(epochs), disable=not verbose, desc=desc)

    for _ in progress_bar:
        model.train()
        p = torch.randperm(n_samples, device=device)
        logits = model(contexts[p])
        w = weights[p]
        total = (criterion(logits, targets[p]) * w).sum()
        loss = total / w.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(total.item())
        progress_bar.set_postfix(loss=f"{total.item():.1f}")

    model.eval()
    return epoch_losses


def _freeze_output(model):
    """Freeze the compass E_out for the retraining stage.

    :param model: the model, frozen in place.
    """
    model.output.weight.requires_grad = False
    model.output.bias.requires_grad = False


# comparison functions


def _cosine(weight_i, weight_j, ids_i, ids_j):
    """Row-wise cosine similarity of two embedding matrices.

    :param weight_i: the first matrix.
    :param weight_j: the second.
    :param ids_i: the rows to take from ``weight_i``.
    :param ids_j: the rows to take from ``weight_j``, aligned with ``ids_i``.
    :returns: one cosine per row pair.
    """
    with torch.no_grad():
        a = F.normalize(weight_i[torch.tensor(ids_i)].cpu().float(), p=2, dim=1)
        b = F.normalize(weight_j[torch.tensor(ids_j)].cpu().float(), p=2, dim=1)
    return (a * b).sum(dim=1).numpy()


# persistence functions


def save_model(model, path):
    """Save a model to ``path``

    :param model: the model to save.
    :param path: the file to write, extension included.
    :returns: the path written.
    """
    torch.save(model, path)
    return path


def load_model(path, device=None):
    """Load a model saved by :func:`save_model`.

    :param path: the file to read.
    :param device: ``None`` picks the best available.
    :returns: the model, in eval mode.
    """
    model = torch.load(path, map_location=_get_device(device), weights_only=False)
    model.eval()
    return model


################################ CONTROL_FLOW ##################################


class CWINDOW(nn.Module):
    """The embeddings of the c context activities are concatenated into the vector
    h, and E_out scores h against the candidate targets Act(L) u {end}.

    Concatenation is the whole point: a context keeps its order, so <a, b> and
    <b, a> give different vectors -- unlike CBOW, which would sum them into one.

    ``word_to_ix`` covers the start marker and the activities -- the rows of E_in
    -- plus the end marker, which is a target only. The candidate targets are its
    ids from 1 upwards, i.e. column ``word_to_ix[token] - TARGET_OFFSET`` of E_out.

    :param word_to_ix: ``{token: id}`` over start marker, activities, end marker.
    :param embedding_dim: d, the embedding dimension.
    :param context_window: c, the window size.
    :param activities: Act(L) of the log the model is fitted on, which may be
        narrower than ``word_to_ix``. Required: it is the row set
        :meth:`activity_embeddings` returns.
    :param unit_norm: keep the activity embeddings on the unit sphere -- every context
        embedding is L2-normalised before it enters h, and :meth:`activity_embeddings`
        returns normalised rows. The model can then no longer spend capacity on the
        length of an embedding, so an activity is described by its direction alone,
        which is what the cosine of the comparison reads.
    """

    def __init__(
        self, word_to_ix, embedding_dim, context_window, activities, unit_norm=False
    ):
        super().__init__()
        self.word_to_ix = dict(word_to_ix)
        self.activities = set(activities)
        self.context_window = context_window  # c   (activities of context)
        self.unit_norm = unit_norm
        # E_in in R^((|A|+1) x d): the start marker (the zero row) and the
        # activities; #END# is a target only and gets no input row
        self.embeddings = nn.Embedding(
            num_embeddings=len(self.word_to_ix) - 1,
            embedding_dim=embedding_dim,
            padding_idx=PAD_IDX,
        )
        # E_out in R^((c*d) x (|A|+1)) on the concatenated context vector
        # h in R^(c*d): one column per candidate target Act(L) u {#END#}, none
        # for #START#
        self.output = nn.Linear(
            context_window * embedding_dim, len(self.word_to_ix) - 1
        )

    def forward(self, context_batch):
        """The logits [N, |A|+1] over the candidate targets, in the column order
        ``word_to_ix[token] - TARGET_OFFSET``.

        :param context_batch: the context activities as ids, ``[N, c]``, left-padded
            with the start marker. Its embedding is 0, so the padding contributes
            the zero vector in its own c slots without disturbing the others.
        """
        embeds = self.embeddings(context_batch)  # [N, c, d]
        if self.unit_norm:
            # per activity, not per context: the zero padding row has no direction and
            # F.normalize leaves it at zero, so the start marker stays the null vector
            embeds = F.normalize(embeds, p=2, dim=2)
        # h = ( emb(a_(t-c)), ..., emb(a_(t-1)) ):  concat ACROSS the window
        h = embeds.view(embeds.shape[0], -1)  # [N, c*d]
        return self.output(h)

    def activity_embeddings(self):
        """The model's own activities and their embeddings, the rows of E_in as they
        are, sorted by activity.

        :returns: ``(activities, [n, d] matrix)``; under ``unit_norm`` the rows are the unit
            vectors the forward pass scores, not the raw stored weights.
        """
        activities = sorted(self.activities)
        ids = torch.tensor([self.word_to_ix[a] for a in activities])
        with torch.no_grad():
            embs = self.embeddings.weight[ids].cpu().float()
            if self.unit_norm:
                embs = F.normalize(embs, p=2, dim=1)
        return activities, embs.numpy()


def _convert_to_simpleEventLog(
    log,
    case_col="case:concept:name",
    activity_col="concept:name",
    time_col="time:timestamp",
):
    """Converts an event log into a simple, totally ordered event log.

    Per case, the events are put in time order and their activities read off as a
    tuple, so equal traces hash equally. Where the timestamp ties, the log's own
    row order breaks it -- the sort is stable.

    :param log: the event log.
    :param case_col: the case id column.
    :param activity_col: the activity column.
    :param time_col: the timestamp column.
    :returns: L, as ``{trace: count}``.
    """

    def _case_to_trace(case_df):
        return tuple(case_df.sort_values(time_col, kind="stable")[activity_col])

    return Counter(log.groupby(case_col).apply(_case_to_trace))


def _encode_cwindow_traces(traces, c, word_to_ix=None):
    """The training multiset S_c(L) of Definition "Training sample", as tensors: the
    vocabulary, the padded traces, the samples, and the tensors.

    Every trace is padded into sigma', and each activity sigma'(t) is one sample
    against the c preceding ones. Equal samples are folded into one row weighted by
    their multiplicity in L.

    :param traces: ONE log; the compass stage gets the joint log.
    :param c: the window size.
    :param word_to_ix: a vocabulary to reuse -- the compass' in the retraining
        stage, where it may be wider than L; ``None`` builds it from Act(L).
    :returns: ``(contexts, targets, weights, word_to_ix)``. ``targets`` holds E_out
        columns, so a sample scores against Act(L) u {end} alone.
    :raises ValueError: if ``c < 1``.
    """
    if c < 1:
        raise ValueError(f"context window c must be >= 1, got {c}.")

    # the vocabulary: #START# on the zero row, the activities sorted, #END# last
    if word_to_ix is None:
        word_to_ix = {START_TOKEN: PAD_IDX}
        word_to_ix.update(
            {a: i + 1 for i, a in enumerate(sorted(_activities_of(traces)))}
        )
        word_to_ix[END_TOKEN] = len(word_to_ix)
    end_id = word_to_ix[END_TOKEN]

    # every trace's samples, equal ones folded into one row and its summed weight
    sample_counts = {}  # (context, target) -> weight
    for trace, trace_count in traces.items():
        # sigma' = < #START#*c, a_1, ..., a_k, #END# >, as ids
        padded = [PAD_IDX] * c + [word_to_ix[a] for a in trace] + [end_id]
        for t in range(c + 1, len(padded) + 1):  # t = c+1, ..., c+k+1
            # the all-#START# context at t = c+1 is a sample like any other: it is
            # what the model learns the log's opening distribution from
            key = (tuple(padded[t - c - 1 : t - 1]), padded[t - 1])
            sample_counts[key] = sample_counts.get(key, 0) + trace_count

    return (
        torch.tensor([list(key[0]) for key in sample_counts], dtype=torch.long),
        torch.tensor(
            [key[1] - TARGET_OFFSET for key in sample_counts], dtype=torch.long
        ),
        torch.tensor(list(sample_counts.values()), dtype=torch.float),
        word_to_ix,
    )


def _train_cwindow(
    traces,
    c,
    embedding_dim,
    learning_rate,
    epochs,
    *,
    word_to_ix=None,
    compass=None,
    unit_norm=False,
    device=None,
    verbose=True,
    desc="compass",
):
    """Fit the CWINDOW model on ONE log.

    Without a ``compass`` this is the compass stage. With one it is the retraining
    stage: the model starts from the compass and its E_out is frozen, so every log
    is fitted against identical target directions.

    :param traces: the log to fit.
    :param c: the window size.
    :param embedding_dim: d, the embedding dimension.
    :param learning_rate: the Adam learning rate.
    :param epochs: how many full-batch steps.
    :param word_to_ix: a vocabulary to reuse -- the compass' in the retraining stage.
    :param compass: the compass model; ``None`` for the compass stage.
    :param unit_norm: keep the activity embeddings on the unit sphere (see :class:`CWINDOW`).
    :param device: the device to train on.
    :param verbose: print the setup and show the loss curve.
    :param desc: the progress bar's label.
    :returns: ``(model, final loss)``.
    """
    contexts, targets, weights, word_to_ix = _encode_cwindow_traces(
        traces, c, word_to_ix=word_to_ix
    )
    device = _get_device(device)

    model = CWINDOW(
        word_to_ix,
        embedding_dim,
        c,
        activities=_activities_of(traces),
        unit_norm=unit_norm,
    ).to(device)
    if compass is not None:
        model.load_state_dict(compass.state_dict())
        _freeze_output(model)

    if verbose:
        print(
            f"Using device: {device} — {epochs} epochs, lr={learning_rate:.1e}"
            + (", unit norm" if unit_norm else "")
            + (", frozen output" if compass is not None else "")
        )
    losses = _fit(
        model,
        contexts,
        targets,
        weights,
        learning_rate=learning_rate,
        epochs=epochs,
        device=device,
        verbose=verbose,
        desc=desc,
    )
    if verbose:
        _plot_loss(losses, f"CWINDOW — {desc}")
    return model, losses[-1]


def compare_cwindow_event_logs(
    log_i,
    log_j,
    names=("log_i", "log_j"),
    c=2,
    embedding_dim=32,
    compass_learning_rate=1e-2,
    compass_epochs=500,
    log_learning_rate=1e-2,
    log_epochs=500,
    unit_norm=False,
    device=None,
    verbose=False,
    balance_compass=False,
    case_col="case:concept:name",
    activity_col="concept:name",
    time_col="time:timestamp",
):
    """Train comparable activity embeddings for two event logs, by the compass method::

        model_i, model_j, compass = compare_cwindow_event_logs(log_i, log_j)

    One model is fitted on the joint log and its E_out kept as the compass; the
    model is then retrained per log with that E_out frozen, which yields the
    aligned embeddings. Read the difference off them with
    :func:`activity_similarity`.

    :param log_i: the first event log.
    :param log_j: the second. Both are converted with
        :func:`_convert_to_simpleEventLog`.
    :param names: how the logs are labelled in the verbose output.
    :param c: the window size, in activities.
    :param embedding_dim: d, the embedding dimension.
    :param compass_learning_rate: the compass stage's learning rate.
    :param compass_epochs: its epochs.
    :param log_learning_rate: the retraining stage's learning rate.
    :param log_epochs: its epochs.
    :param unit_norm: keep the activity embeddings on the unit sphere (see :class:`CWINDOW`).
        Applies to every stage, so the compass and both per-log models agree.
    :param device: the device to train on.
    :param verbose: print each stage and show its loss curve.
    :param balance_compass: balance the logs for the compass stage (see
        :func:`_balance_case_counts`); each log is retrained on all of its data.
    :param case_col: the case id column.
    :param activity_col: the activity column.
    :param time_col: the timestamp column.
    :returns: ``(model_i, model_j, compass)``.
    """
    name_i, name_j = names
    traces_i = _convert_to_simpleEventLog(log_i, case_col, activity_col, time_col)
    traces_j = _convert_to_simpleEventLog(log_j, case_col, activity_col, time_col)

    # the compass is trained on ONE log: both logs, rescaled if they are to weigh
    # equally, and merged
    balanced, factors = _balance_case_counts(
        {name_i: traces_i, name_j: traces_j}, balance_compass
    )
    if verbose:
        print(
            "--- CWINDOW COMPASS on the combined log"
            + (
                f", case counts scaled by {
                    ({n: round(f, 2) for n, f in factors.items()})
                }"
                if balance_compass
                else ""
            )
            + " ---"
        )
    compass, _ = _train_cwindow(
        _combine_logs(balanced),
        c,
        embedding_dim,
        compass_learning_rate,
        compass_epochs,
        unit_norm=unit_norm,
        device=device,
        verbose=verbose,
        desc="compass",
    )

    # one model per log, initialised from the compass with its output frozen
    def per_log(traces, name):
        if verbose:
            print(f"--- CWINDOW RETRAINING on log {name} ---")
        return _train_cwindow(
            traces,
            c,
            embedding_dim,
            log_learning_rate,
            log_epochs,
            word_to_ix=compass.word_to_ix,
            compass=compass,
            unit_norm=unit_norm,
            device=device,
            verbose=verbose,
            desc=name,
        )[0]

    return per_log(traces_i, name_i), per_log(traces_j, name_j), compass


def activity_similarity(model_i, model_j, activities=None):
    """Computes cross-log similarity sim(a) of the shared activities.

    :param model_i: the model of the first log.
    :param model_j: the model of the second.
    :param activities: which activities; ``None`` takes the shared ones, sorted.
    :returns: sim, indexed by activity.
    """
    if activities is None:
        activities = sorted(model_i.activities & model_j.activities)
    sims = _cosine(
        model_i.embeddings.weight,
        model_j.embeddings.weight,
        [model_i.word_to_ix[a] for a in activities],
        [model_j.word_to_ix[a] for a in activities],
    )
    return pd.Series(sims, index=list(activities), name="cosine_similarity")
