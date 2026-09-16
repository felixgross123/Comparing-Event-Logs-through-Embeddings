"""
Comparison of two simple block-structured event logs through embeddings.

A trace is a sequence of blocks, each a non-empty multiset of activities: the
activities within a block are mutually unordered -- the events share a
timestamp -- while every activity of the i-th block precedes every activity of
the j-th for i < j.

The model of Definition "Block-structured embedding model" follows that
structure: within a block the embeddings are summed, which is commutative;
across the c context blocks the block vectors are concatenated, which keeps
their positions apart; and the members of the target block are predicted
individually. The target window is the next block alone, so there is one output
matrix E_out.

The two logs are compared by the compass method: one model is fitted on the
joint log and its E_out kept as the compass, then the model is retrained per
log with E_out frozen. The resulting embeddings are aligned, so their cosine
locates a difference, and the distributions the two models assign to one
context explain it.

The same method slices a log along time: its periods are the windows, the
compass is fitted on all of them together, and one model per window is
retrained against it. Adjacent windows then compare as directly as two logs --
a log against its own past -- and so do two logs within one window.

Case attributes condition the prediction distributed-memory style: the value of
a case becomes a case token in an input matrix E_in' of its own dimension d',
concatenated to the context vector. ONE attribute per model -- independent
attributes in one model share the projection layer and mix their effects.

Vocabulary convention -- ONE padding index, ``PAD_IDX = 0``, in every embedding
matrix:
    start marker -> 0      the zero vector, via padding_idx=0
    activity_i   -> i+1
    end marker   -> |A|+1  a candidate target only, never an input
so E_in holds the ids 0..|A| and E_out the candidate targets Act(L) u {end},
the ids 1..|A|+1 shifted down by ``TARGET_OFFSET``. The case tokens mirror it,
with the missing value on the zero row.

Public API
----------
    compare_block_event_logs             two event logs
                                         -> (model_i, model_j, compass)
    compare_block_event_logs_over_time   the logs sliced into periods
                                         -> ({(log, period): model}, compass)
    compare_block_event_logs_pvdm        the first, conditioned on ONE case attribute
    activity_similarity                  cross-log similarity per activity
    drift_scores                         a log against its own past, per window
    cross_log_scores                     two logs against each other, per window
    attribute_similarity                 cross-log similarity per case token
    save_model / load_model              one model per file, at a given path

A model carries its own vocabularies and Act(L), so comparing, inspecting and
saving need nothing but the model.

Layout
------
    1. SHARED             seeding and device, the encoding primitives, the
                          compass' joint log, the training loop, the cosine,
                          and the persistence
    2. CONTROL-FLOW       the block-structured model on activities alone
    3. TIME               the same, one model per period of the log
    4. DATA               the same, conditioned on ONE case attribute

Both perspective sections run the same chain in the same order: model, log
conversion, encoding, training, entry point, comparison.
"""

import random
from collections import Counter
from itertools import pairwise
from statistics import median

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


#################################### SHARED #####################################

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


def _activities_of(traces, with_attrs=False):
    """Act(L), the activities occurring in a block log.

    :param traces: the block log.
    :param with_attrs: its keys are ``(trace, case token)`` pairs, not traces alone.
    :returns: the set of activities.
    """
    return {
        a
        for key in traces
        for block in (
            key[0] if with_attrs else key
        )  # if traces features attributes, the activities are in the 0. component
        for a in block
    }


# combining and balancing two logs for the compass stage


def _balance_case_counts(traces, balance):
    """Balance the logs' contribution to the compass stage.

    Where the log sizes differ substantially, the imbalance is corrected so that
    neither log dominates the compass. We reweight the case counts rather than
    downsampling the larger log.

    :param traces: ``{log name: block log}``.
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

    :param traces: ``{log name: block log}``.
    :returns: one block log, adding up the counts of a shared trace variant.
    """
    combined = Counter()
    for t in traces.values():
        for trace, count in t.items():
            combined[trace] += count
    return combined


# training funtions


def _plot_loss(losses, title):
    """Plot a run's per-epoch loss.

    :param losses: the per-epoch loss, in order.
    :param title: the plot's title.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Total weighted cross entropy")
    plt.title(title)
    plt.grid(True)
    plt.show()


def _fit(
    model,
    contexts,
    targets,
    weights,
    attrs=None,
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
    :param contexts: the context blocks, ``[N, c, max block length]``.
    :param targets: the target of each sample, as an E_out column, ``[N]``.
    :param weights: each sample's multiplicity in the training multiset, ``[N]``.
    :param attrs: the case token of each sample ``[N]``, ``None`` without one.
    :param learning_rate: the Adam learning rate.
    :param epochs: how many full-batch steps.
    :param device: the device to train on.
    :param verbose: show the progress bar.
    :param desc: its label.
    :returns: the per-epoch cross-entropy of Definition "Block-structured embedding
        model". The step descends the weighted mean, which differs from it by the
        constant total weight, so the gradient is the same either way.
    """
    contexts, targets, weights = (
        contexts.to(device),
        targets.to(device),
        weights.to(device),
    )
    if attrs is not None:
        attrs = attrs.to(device)

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
        logits = model(contexts[p]) if attrs is None else model(contexts[p], attrs[p])
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
    # nothing to freeze when the model was built with ``output_bias=False``
    if model.output.bias is not None:
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


# persistance functions


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
    # a model pickled before ``unit_norm`` existed carries no such attribute, and was
    # by definition fitted without it
    if not hasattr(model, "unit_norm"):
        model.unit_norm = False
    model.eval()
    return model


################################ CONTROL_FLOW ##################################


class BLOCK_STRUCTURED(nn.Module):
    """ "the embeddings of a context block are summed, the c block vectors concatenated
    in to vector h, and E_out scores h against the candidate targets Act(L) u {end}.

    ``word_to_ix`` covers the start marker and the activities -- the rows of E_in
    -- plus the end marker, which is a target only. The candidate targets are its
    ids from 1 upwards, i.e. column ``word_to_ix[token] - TARGET_OFFSET`` of E_out.

    :param word_to_ix: ``{token: id}`` over start marker, activities, end marker.
    :param embedding_dim: d, the embedding dimension.
    :param context_window: c, the window size.
    :param activities: Act(L) of the log the model is fitted on, which may be
        narrower than ``word_to_ix``.  Required: it is the row set
        :meth:`activity_embeddings` returns.
    :param extra_context_dim: d', room in h for the case token of the subclass.
    :param unit_norm: keep the activity embeddings on the unit sphere -- every embedding
        is L2-normalised before the within-block sum, and :meth:`activity_embeddings`
        returns normalised rows. The model can then no longer spend capacity on the
        length of an embedding, so an activity is described by its direction alone,
        which is what the cosine of the comparison reads. A block still grows with the
        number of activities in it: the normalisation is per activity, not per block.
    :param output_bias: give E_out a bias -- one learned offset per target, added to
        every logit whatever the context. word2vec, and the compass method as it is
        usually written, have none; ``False`` drops it and leaves E_out the matrix
        alone. Kept, the bias absorbs how often a target occurs, which a drill-down of
        one activity in an otherwise empty context then reads as that activity's pull.
    """

    def __init__(
        self,
        word_to_ix,
        embedding_dim,
        context_window,
        activities,
        extra_context_dim=0,
        unit_norm=False,
        output_bias=True,
    ):
        super().__init__()
        self.word_to_ix = dict(word_to_ix)
        self.activities = set(activities)
        self.context_window = context_window  # c   (blocks of context)
        self.unit_norm = unit_norm
        # E_in in R^((|A|+1) x d): the start marker (the zero row) and the
        # activities; #END# is a target only and gets no input row
        self.embeddings = nn.Embedding(
            num_embeddings=len(self.word_to_ix) - 1,
            embedding_dim=embedding_dim,
            padding_idx=PAD_IDX,
        )
        # E_out in R^((c*d) x (|A|+1)) on the concatenated context vector
        # h in R^(c*d) (widened by ``extra_context_dim`` for the PV-DM subclass):
        # one column per candidate target Act(L) u {#END#}, none for #START#
        self.output = nn.Linear(
            context_window * embedding_dim + extra_context_dim,
            len(self.word_to_ix) - 1,
            bias=output_bias,
        )

    def forward(self, context_batch):
        """The logits [N, |A|+1] over the candidate targets, in the column order
        ``word_to_ix[token] - TARGET_OFFSET``.

        :param context_batch: the context blocks as ids, ``[N, c, max block length]``,
            right-padded with the start marker. Its embedding is 0, so the padding drops
            out of emb(B) and an all-padding block contributes the zero vector.
        """
        embeds = self.embeddings(context_batch)  # [N, c, L, d]
        if self.unit_norm:
            # per activity, not per block: the padding row has no direction and
            # F.normalize leaves it at zero, so it still drops out of the sum below
            embeds = F.normalize(embeds, p=2, dim=3)
        # emb(B) = sum_{a in B} B(a) emb(a): a block's ids are listed WITH their
        # multiplicity, so the plain sum is the multiplicity-weighted one
        block_vecs = embeds.sum(dim=2)  # [N, c, d]  sum WITHIN blocks
        # h = ( emb(B_1), ..., emb(B_c) ):  concat ACROSS blocks
        h = block_vecs.view(block_vecs.shape[0], -1)  # [N, c*d]
        return self.output(h)

    def activity_embeddings(self):
        """The model's own activities and their embeddings, the rows of E_in as they
        are, sorted by activity.

        :returns: ``(activities, [n, d] matrix)``; under ``unit_norm`` the rows are the
            unit vectors the forward pass sums, not the raw stored weights.
        """
        activities = sorted(self.activities)
        ids = torch.tensor([self.word_to_ix[a] for a in activities])
        with torch.no_grad():
            embs = self.embeddings.weight[ids].cpu().float()
            if self.unit_norm:
                embs = F.normalize(embs, p=2, dim=1)
        return activities, embs.numpy()


def _convert_to_blockEventLog(
    log,
    case_col="case:concept:name",
    activity_col="concept:name",
    time_col="time:timestamp",
):
    """converts an event log into a simple block-structured event log.

    Per case, the events sharing a timestamp form one block -- the multiset of their
    activities -- and the blocks follow one another in time. A block is a sorted
    tuple, multiplicities kept, so equal traces hash equally.

    :param log: the event log.
    :param case_col: the case id column.
    :param activity_col: the activity column.
    :param time_col: the timestamp column, which is only weakly ordering.
    :returns: L, as ``{trace: count}``.
    """

    def _case_to_blocks(case_df):
        return tuple(
            tuple(sorted(group[activity_col]))
            for _, group in case_df.groupby(time_col, sort=True)
        )

    return Counter(log.groupby(case_col).apply(_case_to_blocks))


def _encode_block_traces(traces, c, word_to_ix=None):
    """The training multiset S_c(L) of Definition "Training sample", as tensors: the
    vocabulary, the padded traces, the samples, and the tensors.

    Every trace is padded into sigma', and each activity of the target block
    sigma'(t) is one sample against the c preceding context blocks. Equal samples
    are folded into one row weighted by their multiplicity in L and in sigma'(t).

    :param traces: ONE block log; the compass stage gets the joint log.
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
        # sigma' = < [#START#]*c, B_1, ..., B_k, [#END#] >, blocks as sorted ids
        padded = (
            [(PAD_IDX,)] * c
            + [tuple(sorted(word_to_ix[a] for a in block)) for block in trace]
            + [(end_id,)]
        )
        for t in range(c + 1, len(padded) + 1):  # t = c+1, ..., c+k+1
            context = tuple(padded[t - c - 1 : t - 1])  # <sigma'(t-c),...,sigma'(t-1)>
            # every activity of the target block is one sample, with its
            # multiplicity as the weight it carries into the loss
            for target, mult in Counter(padded[t - 1]).items():
                key = (context, target)
                sample_counts[key] = sample_counts.get(key, 0) + trace_count * mult

    # the tensors: every block right-padded to the longest one in the batch.
    # PAD_IDX embeds to the zero vector, so the filling drops out of the block sum
    contexts = [key[0] for key in sample_counts]
    max_block_len = max(len(block) for ctx in contexts for block in ctx)
    return (
        torch.tensor(
            [
                [list(b) + [PAD_IDX] * (max_block_len - len(b)) for b in ctx]
                for ctx in contexts
            ],
            dtype=torch.long,
        ),
        torch.tensor(
            [key[1] - TARGET_OFFSET for key in sample_counts], dtype=torch.long
        ),
        torch.tensor(list(sample_counts.values()), dtype=torch.float),
        word_to_ix,
    )


def _train_block(
    traces,
    c,
    embedding_dim,
    learning_rate,
    epochs,
    *,
    word_to_ix=None,
    compass=None,
    unit_norm=False,
    output_bias=True,
    device=None,
    verbose=True,
    desc="compass",
):
    """Fit the block-structured model on ONE block log.

    Without a ``compass`` this is the compass stage. With one it is the retraining
    stage: the model starts from the compass and its E_out is frozen, so every log
    is fitted against identical target directions.

    :param traces: the block log to fit.
    :param c: the window size.
    :param embedding_dim: d, the embedding dimension.
    :param learning_rate: the Adam learning rate.
    :param epochs: how many full-batch steps.
    :param word_to_ix: a vocabulary to reuse -- the compass' in the retraining stage.
    :param compass: the compass model; ``None`` for the compass stage.
    :param unit_norm: keep the activity embeddings on the unit sphere (see
        :class:`BLOCK_STRUCTURED`).
    :param output_bias: give E_out a bias (see :class:`BLOCK_STRUCTURED`).
    :param device: the device to train on.
    :param verbose: print the setup and show the loss curve.
    :param desc: the progress bar's label.
    :returns: ``(model, final loss)``.
    """
    contexts, targets, weights, word_to_ix = _encode_block_traces(
        traces, c, word_to_ix=word_to_ix
    )
    device = _get_device(device)

    model = BLOCK_STRUCTURED(
        word_to_ix,
        embedding_dim,
        c,
        activities=_activities_of(traces),
        unit_norm=unit_norm,
        output_bias=output_bias,
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
        _plot_loss(losses, f"BLOCK_STRUCTURED — {desc}")
    return model, losses[-1]


def compare_block_event_logs(
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
    output_bias=True,
    device=None,
    verbose=False,
    balance_compass=False,
    case_col="case:concept:name",
    activity_col="concept:name",
    time_col="time:timestamp",
):
    """Train comparable activity embeddings for two event logs, by the compass method::

        model_i, model_j, compass = compare_block_event_logs(log_i, log_j)

    One model is fitted on the joint log and its E_out kept as the compass; the
    model is then retrained per log with that E_out frozen, which yields the
    aligned embeddings. Read the difference off them with
    :func:`activity_similarity`.

    :param log_i: the first event log.
    :param log_j: the second. Both are converted with
        :func:`_convert_to_blockEventLog`.
    :param names: how the logs are labelled in the verbose output.
    :param c: the window size, in blocks.
    :param embedding_dim: d, the embedding dimension.
    :param compass_learning_rate: the compass stage's learning rate.
    :param compass_epochs: its epochs.
    :param log_learning_rate: the retraining stage's learning rate.
    :param log_epochs: its epochs.
    :param unit_norm: keep the activity embeddings on the unit sphere (see
        :class:`BLOCK_STRUCTURED`). Applies to every stage, so the compass and both
        per-log models agree.
    :param output_bias: give E_out a bias (see :class:`BLOCK_STRUCTURED`). Applies to
        every stage, so the compass and both per-log models agree.
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
    traces_i = _convert_to_blockEventLog(log_i, case_col, activity_col, time_col)
    traces_j = _convert_to_blockEventLog(log_j, case_col, activity_col, time_col)

    # the compass is trained on ONE log: both block logs, rescaled if they are to
    # weigh equally, and merged
    balanced, factors = _balance_case_counts(
        {name_i: traces_i, name_j: traces_j}, balance_compass
    )
    if verbose:
        print(
            "--- BLOCK_STRUCTURED COMPASS on the combined log"
            + (
                f", case counts scaled by {
                    ({n: round(f, 2) for n, f in factors.items()})
                }"
                if balance_compass
                else ""
            )
            + " ---"
        )
    compass, _ = _train_block(
        _combine_logs(balanced),
        c,
        embedding_dim,
        compass_learning_rate,
        compass_epochs,
        unit_norm=unit_norm,
        output_bias=output_bias,
        device=device,
        verbose=verbose,
        desc="compass",
    )

    # one model per log, initialised from the compass with its output frozen
    def per_log(traces, name):
        if verbose:
            print(f"--- BLOCK_STRUCTURED RETRAINING on log {name} ---")
        return _train_block(
            traces,
            c,
            embedding_dim,
            log_learning_rate,
            log_epochs,
            word_to_ix=compass.word_to_ix,
            compass=compass,
            unit_norm=unit_norm,
            output_bias=output_bias,
            device=device,
            verbose=verbose,
            desc=name,
        )[0]

    return per_log(traces_i, name_i), per_log(traces_j, name_j), compass


def activity_similarity(model_i, model_j, activities=None):
    """Computes cross-log similarity sim(a) of the shared activities,

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


################################### TIME #######################################


def _convert_to_dated_blockEventLog(
    log,
    case_col="case:concept:name",
    activity_col="concept:name",
    time_col="time:timestamp",
):
    """converts an event log into a DATED block-structured event log.

    :func:`_convert_to_blockEventLog` keeping each block's period, so the samples
    can be sliced into windows afterwards.

    :param log: the event log.
    :param case_col: the case id column.
    :param activity_col: the activity column.
    :param time_col: the timestamp column, whose values are the periods; they are
        ordered by their own sort order, so a sortable label ("2018/1") works.
    :returns: L, as ``{dated trace: count}`` with a dated trace
        ``((p_1, B_1), ..., (p_k, B_k))``, p_1 < ... < p_k.
    """

    def _case_to_dated_blocks(case_df):
        return tuple(
            (period, tuple(sorted(group[activity_col])))
            for period, group in case_df.groupby(time_col, sort=True)
        )

    return Counter(log.groupby(case_col).apply(_case_to_dated_blocks))


def _dated_activities_of(traces):
    """Act(L) of a dated block log, whose blocks come paired with their period.

    :param traces: the dated block log.
    :returns: the set of activities.
    """
    return {a for trace in traces for (_, block) in trace for a in block}


def _encode_dated_block_traces(traces, c, word_to_ix=None):
    """:func:`_encode_block_traces` on a dated block log, tagging every sample with
    the window it belongs to.

    :param traces: ONE dated block log.
    :param c: the window size, in blocks.
    :param word_to_ix: a vocabulary to reuse -- the shared one, so the windows of
        every log are encoded against identical ids; ``None`` builds it from Act(L).
    :returns: ``(contexts, targets, weights, windows, word_to_ix)``, where
        ``windows`` holds the period of each sample's target block and ``None``
        where that target is the end marker, which belongs to no window.
    :raises ValueError: if ``c < 1``.
    """
    if c < 1:
        raise ValueError(f"context window c must be >= 1, got {c}.")

    if word_to_ix is None:
        word_to_ix = {START_TOKEN: PAD_IDX}
        word_to_ix.update(
            {a: i + 1 for i, a in enumerate(sorted(_dated_activities_of(traces)))}
        )
        word_to_ix[END_TOKEN] = len(word_to_ix)
    end_id = word_to_ix[END_TOKEN]

    # as in _encode_block_traces, with the period of the target block in the key:
    # samples that agree on (context, target) but predict different periods stay
    # apart, so a window's subset is exact
    sample_counts = {}  # (context, target, window) -> weight
    for trace, trace_count in traces.items():
        padded = (
            [(PAD_IDX,)] * c
            + [tuple(sorted(word_to_ix[a] for a in block)) for (_, block) in trace]
            + [(end_id,)]
        )
        # the periods of sigma': the start padding and the end marker have none
        periods = [None] * c + [period for (period, _) in trace] + [None]
        for t in range(c + 1, len(padded) + 1):
            context = tuple(padded[t - c - 1 : t - 1])
            for target, mult in Counter(padded[t - 1]).items():
                key = (context, target, periods[t - 1])
                sample_counts[key] = sample_counts.get(key, 0) + trace_count * mult

    contexts = [key[0] for key in sample_counts]
    max_block_len = max(len(block) for ctx in contexts for block in ctx)
    return (
        torch.tensor(
            [
                [list(b) + [PAD_IDX] * (max_block_len - len(b)) for b in ctx]
                for ctx in contexts
            ],
            dtype=torch.long,
        ),
        torch.tensor(
            [key[1] - TARGET_OFFSET for key in sample_counts], dtype=torch.long
        ),
        torch.tensor(list(sample_counts.values()), dtype=torch.float),
        [key[2] for key in sample_counts],
        word_to_ix,
    )


def _balance_window_weights(weights, windows, balance):
    """Balance the windows' contribution to the compass stage.

    A window's samples are as many as its cases, so without this the busy periods
    fix E_out on their own and the quiet ones are read against a space they had no
    part in. Every window is scaled UP to the busiest one, which therefore stays at
    1.0 and no real sample is ever weighed below its own count -- the anchor
    :func:`_balance_case_counts` uses too. It is cosmetic either way: :func:`_fit`
    descends the WEIGHTED MEAN, so a factor common to every sample cancels and only
    the ratios matter.

    The end-marker samples belong to no window, so there is no window total to put
    them on; they follow the windows as a whole instead, by the factor the window
    mass received, which keeps their share of the log where it was.

    :param weights: the sample weights, ``[N]``.
    :param windows: the period of each sample, ``None`` outside every window.
    :param balance: ``False`` returns the weights unchanged.
    :returns: ``(the weights to train the compass on, {period: factor})``.
    """
    if not balance:
        return weights, {}

    totals = {}
    for w, window in zip(weights.tolist(), windows, strict=True):
        if window is not None:
            totals[window] = totals.get(window, 0.0) + w
    if not totals:
        return weights, {}

    busiest = max(totals.values())
    factors = {window: busiest / total for window, total in totals.items()}
    # what the window mass as a whole was scaled by, for the end markers
    outside = len(totals) * busiest / sum(totals.values())
    return (
        weights
        * torch.tensor(
            [outside if w is None else factors[w] for w in windows], dtype=torch.float
        ),
        factors,
    )


def _present_activities(contexts, ix_to_word):
    """Act(W), the activities a window is actually fitted on.

    A window constrains the input embedding of an activity only where that
    activity occurs in one of its context blocks, so this is the set its
    embeddings may be compared over.

    :param contexts: the window's context blocks, ``[N, c, max block length]``.
    :param ix_to_word: the vocabulary, inverted.
    :returns: the set of activities.
    """
    tokens = {ix_to_word[i] for i in torch.unique(contexts).tolist()}
    return tokens - {START_TOKEN, END_TOKEN}


def compare_block_event_logs_over_time(
    logs,
    compass_log,
    c=2,
    embedding_dim=32,
    compass_learning_rate=1e-2,
    compass_epochs=500,
    window_learning_rate=1e-2,
    window_epochs=500,
    exclude_windows=(),
    unit_norm=False,
    output_bias=True,
    balance_compass=False,
    device=None,
    compass_verbose=False,
    window_verbose=False,
    case_col="case:concept:name",
    activity_col="concept:name",
    time_col="time:timestamp",
):
    """Train comparable activity embeddings per log and period, by the compass method::

        models, compass = compare_block_event_logs_over_time(
            {"on-time": log_i, "late": log_j}, everyone)

    ONE compass is fitted on ``compass_log`` -- the log every other is a part of --
    and its E_out kept; one model per ``(log, period)`` is then retrained from it
    with that E_out frozen. Every window of every log is therefore scored against
    the same target directions and lives in one space: read a log against its own
    past with :func:`drift_scores`, and two logs against each other within a window
    with :func:`cross_log_scores`.

    Passing the whole, unfiltered log as the compass is what keeps the balancing
    simple. The compass sees no cohorts -- only periods -- so the windows are the
    one thing to weigh against each other, and a cohort never has to be weighed
    against another cohort. The cohorts enter afterwards, in the retraining, where
    each is fitted on its own samples alone.

    :param logs: ``{name: event log}``, the logs to resolve over time. Include
        ``compass_log`` under a name of its own to get its windows as a reference.
    :param compass_log: the event log the compass is fitted on.
    :param c: the window size, in blocks.
    :param embedding_dim: d, the embedding dimension.
    :param compass_learning_rate: the compass stage's learning rate.
    :param compass_epochs: its epochs.
    :param window_learning_rate: the retraining stage's learning rate.
    :param window_epochs: its epochs.
    :param exclude_windows: periods to model no window for -- the first c periods,
        which have no full context, or a period too thin to fit. Only their
        samples are dropped; their events stay in the traces and still serve as
        CONTEXT for the neighbouring windows.
    :param unit_norm: keep the activity embeddings on the unit sphere (see
        :class:`BLOCK_STRUCTURED`). Applies to the compass and to every window, so all
        of them agree.
    :param output_bias: give E_out a bias (see :class:`BLOCK_STRUCTURED`). Applies to
        the compass and to every window, so all of them agree.
    :param balance_compass: weigh every window equally in the compass stage (see
        :func:`_balance_window_weights`); a window is retrained on its own samples
        either way.
    :param device: the device to train on.
    :param compass_verbose: print the compass stage with the rescaling factors it
        applies, and show its progress bar and loss curve.
    :param window_verbose: show each window's own progress bar and loss curve -- one
        figure per window, so of the order of the grid. The bar over the windows is
        drawn either way, as the run's own progress.
    :param case_col: the case id column.
    :param activity_col: the activity column.
    :param time_col: the timestamp column, holding the periods.
    :returns: ``({(name, period): model}, compass)``, the windows sorted by period.
    """

    def convert(log):
        return _convert_to_dated_blockEventLog(log, case_col, activity_col, time_col)

    compass_traces = convert(compass_log)
    traces = {name: convert(log) for name, log in logs.items()}

    # ONE vocabulary over the compass log and every log read against it, so all
    # windows share their ids and E_out
    activities = _dated_activities_of(compass_traces).union(
        *(_dated_activities_of(t) for t in traces.values())
    )
    word_to_ix = {START_TOKEN: PAD_IDX}
    word_to_ix.update({a: i + 1 for i, a in enumerate(sorted(activities))})
    word_to_ix[END_TOKEN] = len(word_to_ix)
    ix_to_word = {i: a for a, i in word_to_ix.items()}

    exclude = set(exclude_windows)
    device = _get_device(device)

    # the compass: the whole log, minus the samples of the periods no window is
    # fitted for, with every remaining window weighing the same
    contexts, targets, weights, periods, _ = _encode_dated_block_traces(
        compass_traces, c, word_to_ix=word_to_ix
    )
    keep = torch.tensor(
        [p is None or p not in exclude for p in periods], dtype=torch.bool
    )
    compass_weights, factors = _balance_window_weights(
        weights[keep],
        [p for p, k in zip(periods, keep.tolist(), strict=True) if k],
        balance_compass,
    )
    if compass_verbose:
        print(
            "--- BLOCK_STRUCTURED COMPASS on the whole log"
            + (", every window weighing equally" if balance_compass else "")
            + " ---"
        )
        if factors:
            scaled = sorted(factors.values())
            print(
                f"    {len(scaled)} windows scaled x{scaled[0]:.2f} to "
                f"x{scaled[-1]:.2f} (median x{median(scaled):.2f})"
            )
    compass = BLOCK_STRUCTURED(
        word_to_ix,
        embedding_dim,
        c,
        activities=_dated_activities_of(compass_traces),
        unit_norm=unit_norm,
        output_bias=output_bias,
    ).to(device)
    losses = _fit(
        compass,
        contexts[keep],
        targets[keep],
        compass_weights,
        learning_rate=compass_learning_rate,
        epochs=compass_epochs,
        device=device,
        verbose=compass_verbose,
        desc="compass",
    )
    if compass_verbose:
        _plot_loss(losses, "BLOCK_STRUCTURED — compass on the whole log")

    # one model per (log, period), initialised from the compass with E_out frozen
    encoded = {
        name: _encode_dated_block_traces(t, c, word_to_ix=word_to_ix)[:4]
        for name, t in traces.items()
    }
    windows = sorted(
        {
            (name, p)
            for name, (_, _, _, ps) in encoded.items()
            for p in ps
            if p is not None and p not in exclude
        },
        key=lambda key: (key[1], key[0]),
    )
    models = {}
    for name, period in tqdm(windows, desc="windows"):
        ctx, tgt, wgt, ps = encoded[name]
        mask = torch.tensor([p == period for p in ps], dtype=torch.bool)
        model = BLOCK_STRUCTURED(
            word_to_ix,
            embedding_dim,
            c,
            # Act(W): what this window constrains, and so what it may be compared over
            activities=_present_activities(ctx[mask], ix_to_word),
            unit_norm=unit_norm,
            output_bias=output_bias,
        ).to(device)
        model.load_state_dict(compass.state_dict())
        _freeze_output(model)
        losses = _fit(
            model,
            ctx[mask],
            tgt[mask],
            wgt[mask],
            learning_rate=window_learning_rate,
            epochs=window_epochs,
            device=device,
            verbose=window_verbose,
            desc=f"{name} {period}",
        )
        if window_verbose:
            _plot_loss(losses, f"BLOCK_STRUCTURED — {name} {period}")
        models[(name, period)] = model

    return models, compass


def _windows_of(models, name):
    """The periods one log is modelled over, in order.

    :param models: the windows, from :func:`compare_block_event_logs_over_time`.
    :param name: the log.
    :returns: its periods, sorted.
    """
    return sorted(period for (n, period) in models if n == name)


def drift_scores(models, names=None):
    """The drift of each log against its OWN past, window by window.

    Between two adjacent windows and over the activities both are fitted on, the
    per-activity drift is the halved cosine distance ``(1 - sim) / 2`` in the
    shared compass space, in ``[0, 1]``: 0 = the activity sits in the same
    behaviour in both windows, 1 = it points the opposite way. A peak in the mean
    marks a period where the log's behaviour changed.

    :param models: the windows, from :func:`compare_block_event_logs_over_time`.
    :param names: which logs, in order; ``None`` takes all of them, sorted.
    :returns: ``(mean drift as a frame, {log: per-activity frame})``, both indexed
        by the LATER window of the pair, ``NaN`` where a pair shares no activity.
    """
    if names is None:
        names = sorted({name for (name, _) in models})
    later = sorted({p for name in names for p in _windows_of(models, name)[1:]})

    avg = pd.DataFrame(index=later, columns=list(names), dtype=float)
    per_activity = {}
    for name in names:
        periods = _windows_of(models, name)
        columns = {}
        for previous, period in pairwise(periods):
            sim = activity_similarity(models[(name, previous)], models[(name, period)])
            if sim.empty:
                continue
            # the cosine is clipped first: rounding can return 1 + eps for two
            # near-identical embeddings, which would put the drift below 0
            columns[period] = (1.0 - sim.clip(-1.0, 1.0)) / 2.0
            avg.at[period, name] = float(columns[period].mean())
        per_activity[name] = pd.DataFrame(columns).reindex(columns=later)
    return avg, per_activity


def cross_log_scores(models, name_i, name_j):
    """The similarity of two logs WITHIN each window -- :func:`activity_similarity`
    resolved over time.

    Over the activities both logs are fitted on in that window, the cosine of
    their embeddings: high = the two treat the activity alike at that point,
    low = they part there. A different question from :func:`drift_scores`, which
    compares a log with its own past rather than with the other log.

    :param models: the windows, from :func:`compare_block_event_logs_over_time`.
    :param name_i: the first log.
    :param name_j: the second.
    :returns: ``(mean similarity per window, per-activity frame)``, both indexed by
        window, ``NaN`` where the window is missing for a log or shares no activity.
    """
    periods = sorted(
        set(_windows_of(models, name_i)) & set(_windows_of(models, name_j))
    )
    columns, avg = {}, {}
    for period in periods:
        sim = activity_similarity(models[(name_i, period)], models[(name_j, period)])
        avg[period] = float("nan") if sim.empty else float(sim.mean())
        if not sim.empty:
            columns[period] = sim
    return (
        pd.Series(avg, name="cosine_similarity"),
        pd.DataFrame(columns).reindex(columns=periods),
    )


############################### DATA ###########################################

MISSING_VALUE = "?"  # rendering of a missing case-attribute value


def _fmt_attr_value(v):
    """Render a case-attribute value as a token, ``2`` not ``2.0``, so equal values
    hash to one case token.

    :param v: the raw value.
    :returns: the token; a missing value renders as :data:`MISSING_VALUE`.
    """
    if pd.isna(v):
        return MISSING_VALUE
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, np.integer):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    return str(v)


class BLOCK_STRUCTURED_PVDM(BLOCK_STRUCTURED):
    """:class:`BLOCK_STRUCTURED` conditioned on ONE case attribute, distributed-memory
    style: the value of the case becomes a case token in an input matrix E_in' of
    its own dimension d', looked up per case and concatenated to the context
    vector, h = ( emb(B_1), ..., emb(B_c), emb(g) ), with E_out enlarged
    accordingly.

    One attribute per model: independent attributes in one model share the
    projection layer and mix their effects.

    :param word_to_ix: the activity vocabulary, as in the base class.
    :param embedding_dim: d, the embedding dimension.
    :param context_window: c, the window size.
    :param attr_name: the case attribute conditioned on.
    :param attr_value_to_ix: ``{case token: id}``. Row 0 is the missing value, whose
        embedding stays 0, so the attribute says nothing about a case that lacks it.
    :param attr_dim: d', the dimension of E_in'.
    :param activities: Act(L) of the log the model is fitted on. Required.
    :param unit_norm: keep BOTH embedding matrices on the unit sphere -- the activities
        as in the base class, and the case tokens of E_in' with them, since
        :func:`attribute_similarity` reads those by cosine too. The missing value is the
        padding row: it has no direction, so it stays the zero vector and the attribute
        still says nothing about a case that lacks it.
    :param output_bias: give E_out a bias (see :class:`BLOCK_STRUCTURED`).
    """

    def __init__(
        self,
        word_to_ix,
        embedding_dim,
        context_window,
        attr_name,
        attr_value_to_ix,
        attr_dim,
        activities,
        unit_norm=False,
        output_bias=True,
    ):
        super().__init__(
            word_to_ix,
            embedding_dim,
            context_window,
            activities,
            extra_context_dim=int(attr_dim),
            unit_norm=unit_norm,
            output_bias=output_bias,
        )
        self.attr_name = attr_name
        self.attr_value_to_ix = dict(attr_value_to_ix)
        # the input embedding matrix E_in^n of the attribute's values
        self.attr_embeddings = nn.Embedding(
            num_embeddings=len(self.attr_value_to_ix),
            embedding_dim=attr_dim,
            padding_idx=PAD_IDX,
        )

    @property
    def attr_dim(self):
        return self.attr_embeddings.embedding_dim

    def forward(self, context_batch, attr_batch):
        """The logits [N, |A|+1], as in the base class but off the widened context
        vector h = ( emb(B_1), ..., emb(B_c), emb(g) ).

        :param context_batch: the context blocks as ids, ``[N, c, max block length]``,
            right-padded with the start marker, as in the base class.
        :param attr_batch: the value id of each sample's case, ``[N]``.
        """
        embeds = self.embeddings(context_batch)  # [N, c, L, d]
        attr_vecs = self.attr_embeddings(attr_batch)  # [N, d']  emb(g)
        if self.unit_norm:
            embeds = F.normalize(
                embeds, p=2, dim=3
            )  # per activity, as in the base class
            attr_vecs = F.normalize(attr_vecs, p=2, dim=1)  # the missing row stays zero
        block_vecs = embeds.sum(dim=2)  # [N, c, d]  sum WITHIN blocks
        h = torch.cat(
            [
                block_vecs.view(block_vecs.shape[0], -1),  # [N, c*d]
                attr_vecs,
            ],
            dim=1,
        )
        return self.output(h)  # [N, c*d + d'] -> [N, |A|+1]

    def attribute_embeddings(self, values=None):
        """The case token embeddings, the rows of E_in' as they are.

        :param values: which case tokens, in order; ``None`` takes all. The missing
            value is the zero vector and is left out.
        :returns: ``(case tokens, [k, d'] matrix)``; under ``unit_norm`` the rows are the
            unit vectors the forward pass concatenates, not the raw stored weights.
        """
        mapping = self.attr_value_to_ix
        if values is None:
            values = [
                v for v in sorted(mapping, key=mapping.get) if mapping[v] != PAD_IDX
            ]
        ids = torch.tensor([mapping[v] for v in values])
        with torch.no_grad():
            embs = self.attr_embeddings.weight[ids].cpu().float()
            if self.unit_norm:
                embs = F.normalize(embs, p=2, dim=1)
        return list(values), embs.numpy()


def _convert_to_blockEventLog_with_attr(
    log,
    attr_col,
    case_col="case:concept:name",
    activity_col="concept:name",
    time_col="time:timestamp",
):
    """:func:`_convert_to_blockEventLog` with the case token of each case kept, so the
    model can condition on it.

    Two cases with equal traces but different values stay separate variants, so the
    attribute survives the variant collapse.

    :param log: the event log.
    :param attr_col: the case attribute, constant across a case; the first row's
        value becomes the case token.
    :param case_col: the case id column.
    :param activity_col: the activity column.
    :param time_col: the timestamp column.
    :returns: ``{(trace, case token): count}``.
    """

    def _case_to_blocks_and_attr(case_df):
        blocks = tuple(
            tuple(sorted(group[activity_col]))
            for _, group in case_df.groupby(time_col, sort=True)
        )
        token = f"{attr_col}={_fmt_attr_value(case_df.iloc[0][attr_col])}"
        return (blocks, token)

    return Counter(log.groupby(case_col).apply(_case_to_blocks_and_attr))


def _encode_block_traces_with_attr(
    traces, c, attr_name, word_to_ix=None, attr_value_to_ix=None
):
    """:func:`_encode_block_traces` with the case token id of each sample's case, and
    its vocabulary built the same way.

    :param traces: ONE block log with case tokens.
    :param c: the window size.
    :param attr_name: the case attribute, the prefix of its tokens.
    :param word_to_ix: an activity vocabulary to reuse.
    :param attr_value_to_ix: a case token vocabulary to reuse; ``None`` builds it,
        the missing value on row 0.
    :returns: ``(contexts, targets, attrs, weights, word_to_ix, attr_value_to_ix)``,
        ``attrs`` the case token of each sample.
    :raises ValueError: if ``c < 1``.
    """
    if c < 1:
        raise ValueError(f"context window c must be >= 1, got {c}.")

    # the activity vocabulary, as in :func:`_encode_block_traces`
    if word_to_ix is None:
        word_to_ix = {START_TOKEN: PAD_IDX}
        word_to_ix.update(
            {
                a: i + 1
                for i, a in enumerate(sorted(_activities_of(traces, with_attrs=True)))
            }
        )
        word_to_ix[END_TOKEN] = len(word_to_ix)
    end_id = word_to_ix[END_TOKEN]

    # the attribute vocabulary, mirroring it: the MISSING value takes the zero
    # row, so the module needs no padding index beyond the global one
    if attr_value_to_ix is None:
        missing = f"{attr_name}={MISSING_VALUE}"
        values = sorted({token for (_, token) in traces} - {missing})
        attr_value_to_ix = {
            missing: PAD_IDX,
            **{v: i + 1 for i, v in enumerate(values)},
        }

    sample_counts = {}  # (context, target, attr) -> weight
    for (trace, attr_token), trace_count in traces.items():
        attr_id = attr_value_to_ix[attr_token]
        padded = (
            [(PAD_IDX,)] * c
            + [tuple(sorted(word_to_ix[a] for a in block)) for block in trace]
            + [(end_id,)]
        )
        for t in range(c + 1, len(padded) + 1):  # t = c+1, ..., c+k+1
            context = tuple(padded[t - c - 1 : t - 1])
            for target, mult in Counter(padded[t - 1]).items():
                key = (context, target, attr_id)
                sample_counts[key] = sample_counts.get(key, 0) + trace_count * mult

    contexts = [key[0] for key in sample_counts]
    max_block_len = max(len(block) for ctx in contexts for block in ctx)
    return (
        torch.tensor(
            [
                [list(b) + [PAD_IDX] * (max_block_len - len(b)) for b in ctx]
                for ctx in contexts
            ],
            dtype=torch.long,
        ),
        torch.tensor(
            [key[1] - TARGET_OFFSET for key in sample_counts], dtype=torch.long
        ),
        torch.tensor([key[2] for key in sample_counts], dtype=torch.long),
        torch.tensor(list(sample_counts.values()), dtype=torch.float),
        word_to_ix,
        attr_value_to_ix,
    )


def _train_pvdm(
    traces,
    c,
    embedding_dim,
    attr_name,
    learning_rate,
    epochs,
    *,
    attr_dim=None,
    word_to_ix=None,
    attr_value_to_ix=None,
    compass=None,
    unit_norm=False,
    output_bias=True,
    device=None,
    verbose=True,
    desc="compass",
):
    """:func:`_train_block` for the case-attribute model: compass stage without a
    ``compass``, retraining stage with one. Activity and case token embeddings are
    both refit against the frozen E_out, so both are comparable across logs.

    :param traces: the block log with case tokens.
    :param c: the window size.
    :param embedding_dim: d, the embedding dimension.
    :param attr_name: the case attribute conditioned on.
    :param learning_rate: the Adam learning rate.
    :param epochs: how many full-batch steps.
    :param attr_dim: d'; ``None`` takes d.
    :param word_to_ix: an activity vocabulary to reuse.
    :param attr_value_to_ix: a case token vocabulary to reuse.
    :param compass: the compass model; ``None`` for the compass stage.
    :param unit_norm: keep both embedding matrices on the unit sphere (see
        :class:`BLOCK_STRUCTURED_PVDM`).
    :param output_bias: give E_out a bias (see :class:`BLOCK_STRUCTURED`).
    :param device: the device to train on.
    :param verbose: print the setup and show the loss curve.
    :param desc: the progress bar's label.
    :returns: ``(model, final loss)``.
    """
    contexts, targets, attrs, weights, word_to_ix, attr_value_to_ix = (
        _encode_block_traces_with_attr(
            traces,
            c,
            attr_name,
            word_to_ix=word_to_ix,
            attr_value_to_ix=attr_value_to_ix,
        )
    )
    device = _get_device(device)

    model = BLOCK_STRUCTURED_PVDM(
        word_to_ix,
        embedding_dim,
        c,
        attr_name,
        attr_value_to_ix,
        compass.attr_dim
        if compass is not None
        else (embedding_dim if attr_dim is None else attr_dim),
        activities=_activities_of(traces, with_attrs=True),
        unit_norm=unit_norm,
        output_bias=output_bias,
    ).to(device)

    if compass is not None:
        # initialise from the compass (activity + attribute inputs + output);
        # the missing-value row is zero in the compass and stays zero here
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
        attrs,
        learning_rate=learning_rate,
        epochs=epochs,
        device=device,
        verbose=verbose,
        desc=desc,
    )
    if verbose:
        _plot_loss(losses, f"BLOCK_STRUCTURED_PVDM — {desc}")
    return model, losses[-1]


def compare_block_event_logs_pvdm(
    log_i,
    log_j,
    attr_col,
    names=("log_i", "log_j"),
    c=2,
    embedding_dim=32,
    attr_dim=None,
    compass_learning_rate=1e-2,
    compass_epochs=500,
    log_learning_rate=1e-2,
    log_epochs=500,
    unit_norm=False,
    output_bias=True,
    device=None,
    verbose=False,
    balance_compass=False,
    case_col="case:concept:name",
    activity_col="concept:name",
    time_col="time:timestamp",
):
    """:func:`compare_block_event_logs` conditioned on ONE case attribute::

        model_i, model_j, compass = compare_block_event_logs_pvdm(
        log_i, log_j, "gender")

    Every value of the attribute receives one aligned embedding per log, so case
    tokens become comparable across the logs just as activities do. A case without
    a value takes the missing token, whose embedding is 0: it still fits the
    activity embeddings, and the attribute says nothing about it.

    :param log_i: the first event log.
    :param log_j: the second.
    :param attr_col: the case attribute, constant across a case. One attribute per
        model; a second attribute means a second call.
    :param names: how the logs are labelled in the verbose output.
    :param c: the window size, in blocks.
    :param embedding_dim: d, the embedding dimension.
    :param attr_dim: d'; ``None`` takes d.
    :param compass_learning_rate: the compass stage's learning rate.
    :param compass_epochs: its epochs.
    :param log_learning_rate: the retraining stage's learning rate.
    :param log_epochs: its epochs.
    :param unit_norm: keep both embedding matrices on the unit sphere (see
        :class:`BLOCK_STRUCTURED_PVDM`). Applies to every stage, so the compass and
        both per-log models agree.
    :param output_bias: give E_out a bias (see :class:`BLOCK_STRUCTURED`). Applies to
        every stage, so the compass and both per-log models agree.
    :param device: the device to train on.
    :param verbose: print each stage and show its loss curve.
    :param balance_compass: balance the logs for the compass stage.
    :param case_col: the case id column.
    :param activity_col: the activity column.
    :param time_col: the timestamp column.
    :returns: ``(model_i, model_j, compass)``. Read them with
        :func:`activity_similarity` / :func:`attribute_similarity`.
    """
    name_i, name_j = names
    traces_i = _convert_to_blockEventLog_with_attr(
        log_i, attr_col, case_col, activity_col, time_col
    )
    traces_j = _convert_to_blockEventLog_with_attr(
        log_j, attr_col, case_col, activity_col, time_col
    )

    balanced, factors = _balance_case_counts(
        {name_i: traces_i, name_j: traces_j}, balance_compass
    )
    if verbose:
        print(
            f"--- PV-DM COMPASS on the combined log, attribute {attr_col!r}"
            + (
                f", case counts scaled by {
                    ({n: round(f, 2) for n, f in factors.items()})
                }"
                if balance_compass
                else ""
            )
            + " ---"
        )
    compass, _ = _train_pvdm(
        _combine_logs(balanced),
        c,
        embedding_dim,
        attr_col,
        compass_learning_rate,
        compass_epochs,
        attr_dim=attr_dim,
        unit_norm=unit_norm,
        output_bias=output_bias,
        device=device,
        verbose=verbose,
        desc="compass",
    )

    def per_log(traces, name):
        if verbose:
            print(f"--- PV-DM RETRAINING on log {name} ---")
        return _train_pvdm(
            traces,
            c,
            embedding_dim,
            attr_col,
            log_learning_rate,
            log_epochs,
            word_to_ix=compass.word_to_ix,
            attr_value_to_ix=compass.attr_value_to_ix,
            compass=compass,
            unit_norm=unit_norm,
            output_bias=output_bias,
            device=device,
            verbose=verbose,
            desc=name,
        )[0]

    return per_log(traces_i, name_i), per_log(traces_j, name_j), compass


def attribute_similarity(model_i, model_j, values=None):
    """The cross-log similarity of the case tokens -- :func:`activity_similarity` for
    the case attribute.

    :param model_i: the model of the first log.
    :param model_j: the model of the second.
    :param values: which case tokens; ``None`` takes all with an embedding. The
        missing value is the zero vector and is left out.
    :returns: the similarity, indexed by case token.
    """
    mapping = model_i.attr_value_to_ix
    if values is None:
        values = [v for v in sorted(mapping, key=mapping.get) if mapping[v] != PAD_IDX]
    sims = _cosine(
        model_i.attr_embeddings.weight,
        model_j.attr_embeddings.weight,
        [mapping[v] for v in values],
        [model_j.attr_value_to_ix[v] for v in values],
    )
    return pd.Series(sims, index=list(values), name="cosine_similarity")
