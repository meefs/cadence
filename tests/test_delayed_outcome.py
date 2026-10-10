"""An outcome that arrives after the stream has sensed more: ``Brain.wait`` and ``decision``.

Contract
--------
A ``live`` action owns the next actual outcome of its stream. ``Brain.wait`` settles the
observations that arrive before that outcome: the stream's activity and its working trace
advance, while the awaited action keeps the forecasts made before it, its eligibility, the
situation it was chosen in and its identity. Parameters, the critic, eligibility traces,
memories, random state, the copy of the issued command and the arousal level and age stay as
they were; the settles are charged to ``arousal.sweeps``. When ``live`` (or ``learn``) later
receives the outcome, it is credited as an immediate outcome of that action would be: the same
actor and critic eligibility, the same forecast and the same associative record. The next state
settles from the state the stream sensed last, as do the next answer and imagination.
``Brain.decision`` names the awaited ``live`` action; an outcome reported under any other
decision is refused before anything changes. A checkpoint taken while waiting resumes the wait.

The checks below exercise those statements on small composed brains, against an immediate
twin restored from a checkpoint taken before the wait, and against transcriptions of the
settles that must start from the sensed state (the same solver called with that state). A
mutation harness replaces runtime methods by deliberately wrong variants and records which
checks kill each one.

What neither side establishes: a behavioral or cognitive gain from waiting, the timing of a
real body, more than one awaited action per stream, batched streams, or that the body executed
the issued action. Comparisons use exact equality where both sides perform the same operations
in the same order.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import cadence as cd

EYE = np.eye(4)


def _life(*, youth: int = 0, seed: int = 0, efference: float = 0.3, **options: Any) -> cd.Brain:
    return cd.Brain.compose(
        4,
        2,
        modules=(8,),
        seed=seed,
        working_memory_amplitude=0.3,
        efference_amplitude=efference,
        arousal=cd.ArousalConfig(youth=youth),
        **options,
    )


def _twin(brain: cd.Brain, path) -> cd.Brain:
    return cd.Brain.load(brain.save(path))


def _durable(brain: cd.Brain) -> dict[str, Any]:
    """Everything an awaited outcome must not change before it arrives."""
    agent, memory = brain.basal_ganglia, brain.hippocampus
    values: dict[str, Any] = {
        "efficacy": np.asarray(brain.brain.efficacy).copy(),
        "bias": np.asarray(brain.brain.bias).copy(),
        "critic": np.append(agent.w_critic, agent.b_critic),
        "moments": np.concatenate(
            [agent.velocity, agent.velocity_bias, agent.second_moment, agent.second_moment_bias]
        ),
        "consolidated": memory.consolidated.copy(),
        "strength": memory.strength.copy(),
        "mass": memory.mass.copy(),
        "counts": np.array([brain.learner.updates, agent.updates, memory.writes]),
        "rng": json.dumps(brain.rng.bit_generator.state, sort_keys=True, default=str),
        "actor_rng": json.dumps(agent.rng.bit_generator.state, sort_keys=True, default=str),
    }
    for name in ("trace", "trace_bias", "trace_critic"):
        value = getattr(agent, name)
        values[name] = None if value is None else value.copy()
    if brain.efference is not None:
        for name in ("trace", "last", "cold"):
            values["efference/" + name] = getattr(brain.efference, name).copy()
    return values


def _mood(brain: cd.Brain) -> dict[str, Any]:
    """The arousal state, without the settling work that waiting is charged."""
    values = brain.arousal.to_dict()
    del values["sweeps"]
    return values


def _same(first: dict[str, Any], second: dict[str, Any]) -> None:
    assert first.keys() == second.keys()
    for name, value in first.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(second[name], value, err_msg=name)
        else:
            assert second[name] == value, name


def _owing(
    *, youth: int, seed: int = 0, efference: float = 0.3, signed: bool = True
) -> cd.Brain:
    """A life whose last action awaits its outcome. A young life has the eligibility of
    earlier sampled outcomes; a calm life paid nothing (``signed=False``) acts in routine."""
    brain = _life(youth=youth, seed=seed, efference=efference)
    action = brain.live(EYE[[0]])
    for moment in range(4):
        reward = (1.0 if int(action[0]) == moment % 2 else -1.0) if signed else 0.0
        action = brain.live(EYE[[(moment + 1) % 4]], reward=[reward])
    assert brain.pending_feedback
    return brain


# ---------------------------------------------------------------------------
# The checks


def check_waiting_keeps_the_awaited_action() -> None:
    """Custody: waiting takes no outcome, issues nothing and changes no durable state."""
    for sampled in (True, False):
        brain = _owing(youth=40) if sampled else _owing(youth=0, signed=False)
        agent = brain.basal_ganglia
        assert (agent._pending is not None) == sampled and brain._lived[3] == sampled
        if sampled:
            assert agent.trace is not None and np.abs(agent.trace).max() > 0
        pending, lived, state = agent._pending, brain._lived, agent.state
        moment = None if brain._moment is None else tuple(a.copy() for a in brain._moment)
        before, mood, decision = _durable(brain), _mood(brain), brain.decision
        assert decision == brain.arousal.age
        for frame in (EYE[[1]], EYE[[2]], EYE[[3]]):
            assert brain.wait(frame) is None
        assert agent._pending is pending and brain._lived is lived and agent.state is state
        if moment is None:
            assert brain._moment is None
        else:
            for kept, value in zip(moment, brain._moment, strict=True):
                np.testing.assert_array_equal(value, kept)
        _same(before, _durable(brain))
        assert _mood(brain) == mood
        assert brain.pending_feedback and brain.decision == decision


def check_waiting_advances_activity_and_working_trace(tmp_path) -> None:
    """The second wait settles from the first one's state; the trace follows each state."""
    brain = _owing(youth=40)
    brain.wait(EYE[[1]])
    first = brain._awaiting[1]
    assert not np.array_equal(first.activation, brain.basal_ganglia.state.activation)
    twin = _twin(brain, tmp_path / "after-first-wait.npz")
    trace = brain.working_memory
    old = trace.trace.copy()
    brain.wait(EYE[[2]])
    sensed = brain._awaiting[1]
    cfg = twin.learner.config
    expected = twin._equilibrate(
        twin.stimulus(EYE[[2]]), twin._awaiting[1], budget=cfg.free_steps, tolerance=cfg.tolerance
    ).state
    np.testing.assert_array_equal(sensed.activation, expected.activation)
    np.testing.assert_array_equal(sensed.v, expected.v)
    source = np.asarray(sensed.activation)[:, brain.association_index]
    np.testing.assert_array_equal(trace.trace, trace.decay * old + (1.0 - trace.decay) * source)
    np.testing.assert_array_equal(trace.last, source)


def check_waiting_work_is_counted() -> None:
    """Each wait's settle enters the awaited mode's sweeps; no moment is lived."""
    brain = _owing(youth=0)
    arousal = brain.arousal
    mode, sweeps, moments, age = arousal.mode, dict(arousal.sweeps), dict(arousal.moments), arousal.age
    reading = brain.last_arousal
    spent = 0
    for frame in (EYE[[1]], EYE[[2]]):
        brain.wait(frame)
        report = brain.last_settlement
        assert report["operation"] == "wait" and report["qualified"]
        spent += report["steps"]
    assert spent > 0
    assert arousal.sweeps[mode] == sweeps[mode] + spent
    assert arousal.moments == moments and arousal.age == age
    assert brain.last_arousal is reading


def check_a_delayed_outcome_is_credited_as_an_immediate_one(tmp_path) -> None:
    """The waited brain and an immediate twin credit the same action, forecast and record."""
    brain = _owing(youth=40)
    assert brain.basal_ganglia._pending is not None
    twin = _twin(brain, tmp_path / "before-the-wait.npz")
    forecast = float(brain.basal_ganglia._pending[3][0])
    decision = brain.decision
    for frame in (EYE[[1]], EYE[[2]]):
        brain.wait(frame)
    brain.live(EYE[[3]], reward=[1.0], decision=decision)
    twin.live(EYE[[3]], reward=[1.0], decision=decision)
    for one in (brain, twin):
        assert one.last_learning["value"] == forecast
        assert one.last_arousal["learned"]
    agent, other = brain.basal_ganglia, twin.basal_ganglia
    for name in ("trace", "trace_bias", "trace_critic"):
        np.testing.assert_array_equal(getattr(agent, name), getattr(other, name), err_msg=name)
    for name in ("consolidated", "strength", "mass"):
        np.testing.assert_array_equal(
            getattr(brain.hippocampus, name), getattr(twin.hippocampus, name), err_msg=name
        )
    assert brain.hippocampus.writes == twin.hippocampus.writes


def check_the_next_state_settles_from_the_sensed_state(tmp_path) -> None:
    """``learn`` after a wait: the transcription of its next-state settle from the sensed state."""
    brain = _owing(youth=40)
    for frame in (EYE[[1]], EYE[[2]]):
        brain.wait(frame)
    twin = _twin(brain, tmp_path / "waiting.npz")
    reward, done, following = np.array([1.0]), np.array([False]), EYE[[3]]
    keys, action = twin._moment
    twin._record(keys, action, reward, np.abs(reward))
    expected = twin.learner.free(twin.stimulus(following), warm=twin._awaiting[1])
    brain.learn(reward, done, following)
    state = brain.basal_ganglia.state
    np.testing.assert_array_equal(state.activation, expected.activation)
    np.testing.assert_array_equal(state.v, expected.v)
    assert brain._awaited() is None and not brain.pending_feedback


def check_the_next_answer_settles_from_the_sensed_state(tmp_path) -> None:
    """An answer after a wait settles from the sensed state, and so does imagination."""
    brain = _owing(youth=40)
    brain.wait(EYE[[1]])
    twin = _twin(brain, tmp_path / "waiting.npz")
    cfg = twin.learner.config
    before, sensed = _durable(brain), brain._awaiting
    phases = brain.imagine([EYE[[2]]], budget=cfg.free_steps, tolerance=cfg.tolerance)
    expected = twin._equilibrate(
        twin.stimulus(EYE[[2]]), twin._awaiting[1], budget=cfg.free_steps, tolerance=cfg.tolerance
    ).state
    np.testing.assert_array_equal(phases[0].state.activation, expected.activation)
    _same(before, _durable(brain))
    assert brain._awaiting is sensed and brain.pending_feedback
    brain.act(EYE[[2]], greedy=True)  # acting again replaces the awaited action
    np.testing.assert_array_equal(brain.basal_ganglia.state.activation, expected.activation)
    assert brain._awaited() is None and brain.decision is None and not brain.pending_feedback


def check_decision_names_the_awaited_live_action() -> None:
    """None before an action; ``arousal.age`` once ``live`` issues one; unchanged by waiting."""
    brain = _life(youth=40)
    assert brain.decision is None
    brain.live(EYE[[0]])
    assert brain.decision == 1 == brain.arousal.age
    brain.wait(EYE[[1]])
    assert brain.decision == 1
    brain.live(EYE[[2]], reward=[0.0], decision=np.int64(1))
    assert brain.decision == 2 == brain.arousal.age
    brain.act(EYE[[3]])  # an action that act sampled owns the outcome, without a number
    assert brain.pending_feedback and brain.decision is None
    brain.live(EYE[[0]], reward=[0.0])  # adopted as before
    assert brain.decision == 3


def check_an_outcome_of_another_decision_is_refused() -> None:
    """Stale, unissued and malformed decisions change nothing; the awaited one is taken once."""
    brain = _owing(youth=40)
    brain.wait(EYE[[1]])
    owner = brain.decision
    sensed, trace = brain._awaiting, brain.working_memory.trace.copy()
    before, mood = _durable(brain), _mood(brain)
    for bad in (owner - 1, owner + 1, 0, -1, True, np.bool_(True), float(owner), str(owner)):
        with pytest.raises(ValueError, match="decision"):
            brain.live(EYE[[2]], reward=[1.0], decision=bad)
    _same(before, _durable(brain))
    assert _mood(brain) == mood and brain._awaiting is sensed and brain.decision == owner
    np.testing.assert_array_equal(brain.working_memory.trace, trace)
    updates = brain.basal_ganglia.updates
    brain.live(EYE[[2]], reward=[1.0], decision=owner)
    assert brain.basal_ganglia.updates == updates + 1 and brain.decision == owner + 1
    before, mood = _durable(brain), _mood(brain)
    with pytest.raises(ValueError, match="does not own the next outcome"):
        brain.live(EYE[[3]], reward=[1.0], decision=owner)  # the same outcome, reported again
    _same(before, _durable(brain))
    assert _mood(brain) == mood and brain.decision == owner + 1


def check_an_accepted_outcome_is_not_taken_again_after_the_answer_refuses() -> None:
    """Once taken, an outcome is gone even if the following answer refuses."""
    brain = _owing(youth=40)
    brain.wait(EYE[[1]])
    owner = brain.decision
    updates = brain.basal_ganglia.updates

    def refuse(x):
        raise RuntimeError("brain did not settle within 0 steps")

    brain._settled = refuse  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="did not settle"):
        brain.live(EYE[[2]], reward=[1.0], decision=owner)
    del brain._settled
    assert brain.basal_ganglia.updates == updates + 1
    assert not brain.pending_feedback and brain.decision is None
    with pytest.raises(ValueError, match="live\\(observations\\) alone"):
        brain.live(EYE[[2]], reward=[1.0], decision=owner)
    with pytest.raises(RuntimeError, match="awaiting its outcome"):
        brain.wait(EYE[[2]])
    brain.live(EYE[[2]])
    assert brain.basal_ganglia.updates == updates + 1 and brain.decision == owner + 1


def check_a_refused_wait_changes_nothing() -> None:
    """A wait that cannot qualify, or with invalid observations, leaves the stream as it was."""
    brain = _owing(youth=40)
    brain.wait(EYE[[1]])
    sensed, trace = brain._awaiting, brain.working_memory.trace.copy()
    before, arousal = _durable(brain), brain.arousal.to_dict()
    for bad in (EYE[:2], np.ones((1, 3)), np.array([[np.nan, 0.0, 0.0, 0.0]])):
        with pytest.raises(ValueError):
            brain.wait(bad)
    config = brain.learner.config
    brain.learner.config = replace(config, free_steps=0)
    with pytest.raises(RuntimeError, match="did not settle"):
        brain.wait(EYE[[2]])
    assert brain.last_settlement["operation"] == "wait" and not brain.last_settlement["qualified"]
    _same(before, _durable(brain))
    assert brain.arousal.to_dict() == arousal and brain._awaiting is sensed
    np.testing.assert_array_equal(brain.working_memory.trace, trace)
    brain.learner.config = config
    brain.wait(EYE[[2]])
    assert brain._awaiting is not sensed


def check_waiting_needs_a_live_action_awaiting_its_outcome() -> None:
    plain = cd.Brain.compose(4, 2, modules=(8,), seed=0)
    plain.step(EYE[[0]])
    with pytest.raises(ValueError, match="arousal"):
        plain.wait(EYE[[1]])
    brain = _life(youth=40)
    with pytest.raises(RuntimeError, match="awaiting its outcome"):
        brain.wait(EYE[[0]])
    brain.live(EYE[[0]])
    brain.act(EYE[[1]], greedy=True)  # a greedy act takes the stream; no outcome is owed
    before = _durable(brain)
    with pytest.raises(RuntimeError, match="awaiting its outcome"):
        brain.wait(EYE[[2]])
    _same(before, _durable(brain))
    assert brain._awaiting is None


def check_a_routine_outcome_after_a_wait_is_measured_against_its_own_forecasts() -> None:
    """A calm action waits; its outcome is measured against the record held when it acted."""
    brain = _life(youth=0)
    for _ in range(6):
        brain.live(EYE[[0]], reward=None if brain.decision is None else [0.0])
    lived = brain._lived
    assert not lived[3] and brain.last_arousal["mode"] == "routine"
    action, record = int(lived[1][0]), lived[6]
    recalled = {cue: brain.hippocampus.recall(EYE[[cue]])[0].copy() for cue in (1, 2)}
    for frame in (EYE[[1]], EYE[[2]]):
        brain.wait(frame)
    brain.live(EYE[[3]], reward=[-1.0], decision=brain.decision)
    reading = brain.last_arousal
    assert reading["record_error"] == pytest.approx(abs(-1.0 - record))
    assert reading["mode"] == "aroused" and reading["recorded"] and not reading["learned"]
    assert brain.hippocampus.recall(EYE[[0]])[0][action] < -0.5  # where the action was chosen
    for cue, value in recalled.items():  # and not where the stream waited
        np.testing.assert_array_equal(brain.hippocampus.recall(EYE[[cue]])[0], value)


def check_a_life_saved_while_waiting_continues_identically(tmp_path) -> None:
    for youth, efference in ((40, 0.3), (0, 0.0)):
        brain = _owing(youth=youth, efference=efference, seed=youth)
        for frame in (EYE[[1]], EYE[[2]]):
            brain.wait(frame)
        path = brain.save(tmp_path / f"waiting-{youth}.npz")
        with np.load(path) as saved:
            meta = json.loads(str(saved["generic"]))
            assert meta["format"] == "cadence-generic/5" and "awaiting" in meta
            assert {"awaiting/v", "awaiting/activation", "awaiting/adaptation"} <= set(saved.files)
        twin = cd.Brain.load(path)
        assert twin.decision == brain.decision and twin.pending_feedback
        sensed, restored = brain._awaited(), twin._awaited()
        assert restored is not None
        for name in ("v", "activation", "adaptation"):
            np.testing.assert_array_equal(getattr(restored, name), getattr(sensed, name))
        a = b = None
        for moment in range(24):
            if moment == 0:
                for one in (brain, twin):
                    one.wait(EYE[[3]])
                    assert one.last_settlement["steps"] == brain.last_settlement["steps"]
                reward = [1.0]
            else:
                reward = [1.0 if int(a[0]) == moment % 2 else -1.0]
            decision = brain.decision
            a = brain.live(EYE[[moment % 4]], reward=reward, decision=decision)
            b = twin.live(EYE[[moment % 4]], reward=reward, decision=decision)
            np.testing.assert_array_equal(a, b)
            assert brain.last_arousal == twin.last_arousal
        _same(_durable(brain), _durable(twin))
        assert brain.arousal.to_dict() == twin.arousal.to_dict()


CHECKS: dict[str, Callable[..., None]] = {
    name.removeprefix("check_"): value
    for name, value in dict(globals()).items()
    if name.startswith("check_")
}


def _run(check: Callable[..., None], tmp_path) -> None:
    if check.__code__.co_argcount:
        check(tmp_path)
    else:
        check()


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_runtime_satisfies_the_contract_check(name: str, tmp_path) -> None:
    _run(CHECKS[name], tmp_path)


# ---------------------------------------------------------------------------
# Beside the contract: what an omitted reward does instead


def test_an_omitted_reward_is_an_outcome_where_a_wait_takes_none():
    """The released loop reports frames between an action and its outcome as zero outcomes:
    the late reward is credited to the action chosen at the last frame. ``wait`` keeps it for
    the action that earned it."""
    zero, waited = _life(youth=40), _life(youth=40)
    first = zero.live(EYE[[0]])
    np.testing.assert_array_equal(first, waited.live(EYE[[0]]))
    late = zero.live(EYE[[1]])
    late = zero.live(EYE[[2]])
    zero.live(EYE[[3]], reward=[1.0])
    owner = waited.decision
    for frame in (EYE[[1]], EYE[[2]]):
        waited.wait(frame)
    waited.live(EYE[[3]], reward=[1.0], decision=owner)
    assert zero.basal_ganglia.updates == 3 and zero.hippocampus.writes == 3
    assert waited.basal_ganglia.updates == 1 and waited.hippocampus.writes == 1
    assert zero.hippocampus.recall(EYE[[2]])[0][int(late[0])] > 0.5
    assert zero.hippocampus.recall(EYE[[0]])[0][int(first[0])] == pytest.approx(0.0)
    assert waited.hippocampus.recall(EYE[[0]])[0][int(first[0])] > 0.5
    for cue in (1, 2):
        assert not waited.hippocampus.recall(EYE[[cue]]).any()


def test_naming_the_decision_changes_nothing_in_a_life():
    plain, named = _life(youth=20, seed=3), _life(youth=20, seed=3)
    a = plain.live(EYE[[0]])
    b = named.live(EYE[[0]])
    for moment in range(60):
        reward = [1.0 if int(a[0]) == moment % 2 else -1.0]
        x = EYE[[(moment * 3 + 1) % 4]]
        a = plain.live(x, reward=reward)
        b = named.live(x, reward=reward, decision=named.decision)
        np.testing.assert_array_equal(a, b)
        assert plain.last_arousal == named.last_arousal
    _same(_durable(plain), _durable(named))


def test_a_life_that_never_waits_keeps_its_checkpoint_format(tmp_path):
    for efference, expected in ((0.0, "cadence-generic/3"), (0.3, "cadence-generic/4")):
        brain = _owing(youth=40, efference=efference)
        with np.load(brain.save(tmp_path / f"format-{efference}.npz")) as saved:
            meta = json.loads(str(saved["generic"]))
            assert meta["format"] == expected and "awaiting" not in meta
            assert not any(name.startswith("awaiting/") for name in saved.files)
        brain.wait(EYE[[1]])
        brain.live(EYE[[2]], reward=[0.0])  # the wait ended with its outcome
        with np.load(brain.save(tmp_path / f"after-{efference}.npz")) as saved:
            assert json.loads(str(saved["generic"]))["format"] == expected


def _rewrite(path, change) -> None:
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    metadata = json.loads(str(arrays["generic"]))
    change(metadata, arrays)
    arrays["generic"] = np.array(json.dumps(metadata))
    np.savez(path, **arrays)


@pytest.mark.parametrize(
    "defect",
    [
        "older_format",
        "missing_metadata",
        "missing_state",
        "shape",
        "nonfinite",
        "steps",
        "without_arousal",
        "without_an_awaited_action",
        "stray_arrays",
    ],
)
def test_a_corrupt_waiting_checkpoint_is_refused(tmp_path, defect):
    brain = _owing(youth=0, efference=0.0, signed=False)  # a routine action awaits
    assert not brain.basal_ganglia._pending and not brain._lived[3]
    brain.wait(EYE[[1]])
    path = brain.save(tmp_path / "waiting.npz")
    if defect == "stray_arrays":
        brain.live(EYE[[2]], reward=[0.0])
        path = brain.save(tmp_path / "not-waiting.npz")
        sensed = {name: np.zeros((1, brain.connectome.n)) for name in ("v", "activation")}

    def corrupt(metadata, arrays):
        if defect == "older_format":
            metadata["format"] = "cadence-generic/3"
        elif defect == "missing_metadata":
            del metadata["awaiting"]
        elif defect == "missing_state":
            del arrays["awaiting/adaptation"]
        elif defect == "shape":
            arrays["awaiting/v"] = arrays["awaiting/v"][:, :-1]
        elif defect == "nonfinite":
            arrays["awaiting/activation"][0, 0] = np.inf
        elif defect == "steps":
            metadata["awaiting"]["steps"] = True
        elif defect == "without_arousal":
            del metadata["arousal"], metadata["lived"]
            del arrays["lived/observations"], arrays["lived/action"]
        elif defect == "without_an_awaited_action":
            del metadata["lived"]
            del arrays["lived/observations"], arrays["lived/action"]
        else:
            for name, value in sensed.items():
                arrays["awaiting/" + name] = value

    _rewrite(path, corrupt)
    with pytest.raises(ValueError):
        cd.Brain.load(path)


def test_a_slotted_action_waits_and_is_credited_as_an_immediate_one(tmp_path):
    brain = cd.Brain.compose(
        4, 4, modules=(8,), slots=2, seed=1, working_memory_amplitude=0.3,
        arousal=cd.ArousalConfig(youth=40),
    )
    action = brain.live(EYE[[0]])
    action = brain.live(EYE[[1]], reward=[float(action[0, 0] == 0)])
    twin = _twin(brain, tmp_path / "slotted.npz")
    owner = brain.decision
    brain.wait(EYE[[2]])
    for one in (brain, twin):
        one.live(EYE[[3]], reward=[1.0], decision=owner)
    for name in ("trace", "trace_bias", "trace_critic"):
        np.testing.assert_array_equal(
            getattr(brain.basal_ganglia, name), getattr(twin.basal_ganglia, name)
        )
    np.testing.assert_array_equal(brain.hippocampus.consolidated, twin.hippocampus.consolidated)


def test_a_wait_on_the_torch_backend_credits_and_resumes_like_the_host(tmp_path):
    pytest.importorskip("torch")
    brain = cd.Brain.compose(
        4, 2, modules=(8,), seed=0, working_memory_amplitude=0.3, backend="torch",
        arousal=cd.ArousalConfig(youth=40),
    )
    action = brain.live(EYE[[0]])
    for moment in range(3):
        action = brain.live(EYE[[moment + 1]], reward=[float(action[0] == moment % 2)])
    twin = _twin(brain, tmp_path / "torch-before.npz")
    owner = brain.decision
    brain.wait(EYE[[2]])
    resumed = cd.Brain.load(brain.save(tmp_path / "torch-waiting.npz"), backend="torch")
    for one in (brain, twin, resumed):
        one.live(EYE[[3]], reward=[1.0], decision=owner)
    for agent in (twin.basal_ganglia, resumed.basal_ganglia):
        np.testing.assert_array_equal(agent.trace_critic, brain.basal_ganglia.trace_critic)
    for moment in range(12):
        reward = [float(moment % 3 == 0) - 0.5]
        x = EYE[[moment % 4]]
        np.testing.assert_array_equal(
            brain.live(x, reward=reward), resumed.live(x, reward=reward)
        )
    assert brain.arousal.to_dict() == resumed.arousal.to_dict()


# ---------------------------------------------------------------------------
# The mutation harness


def _mutant_wait(variant: str) -> Callable[[cd.Brain, Any], None]:
    """``Brain.wait`` with one deliberate defect; ``control`` is the runtime statement."""

    def wait(self: cd.Brain, observations: Any) -> None:
        if variant == "takes_a_zero_outcome":
            self.live(observations)
            return
        arousal = self.arousal
        if arousal is None:
            raise ValueError(
                "wait needs arousal genes; construct the brain with arousal=True"
            )
        x = self._observations(observations)
        current = self.basal_ganglia.state
        if len(x) != 1 or (current is not None and len(np.atleast_2d(current.v)) != 1):
            raise ValueError("wait follows one continuing stream; reset before changing streams")
        if variant != "waits_on_a_replaced_action" and (
            current is None or not self.pending_feedback
        ):
            raise RuntimeError(
                "wait needs an issued action awaiting its outcome; start with live(observations)"
            )
        drive = self.stimulus(x)
        try:
            state = self._qualified(drive, self._activity(), operation="wait")
        except RuntimeError:
            if variant != "keeps_an_unqualified_state":
                raise
            cfg = self.learner.config
            state = self._equilibrate(
                drive, self._activity(), budget=cfg.free_steps, tolerance=cfg.tolerance
            ).state
        assert self._last_settlement is not None
        if self.working_memory is not None and variant != "skips_the_working_trace":
            self.working_memory.update(state)
        if variant == "rewrites_the_command_copy" and self.efference is not None:
            self.efference.issue(self._command(self._choices(state)))
        if variant == "fades_eligibility":
            self.basal_ganglia.fade()
        if variant == "installs_the_sensed_state":
            # the earlier feasibility seam: the sensed state becomes the action's own state
            agent = self.basal_ganglia
            agent._free, agent._drive = state, drive.copy()
            if self._lived is not None:
                self._lived = (*self._lived[:4], state, *self._lived[5:])
            current = state
        self._awaiting = (current, state)
        steps = int(self._last_settlement["steps"])
        if variant == "counts_a_lived_moment":
            arousal.lived(steps)
        elif variant != "charges_no_work":
            arousal.waited(steps)

    return wait


def _next_state_from_the_action(
    self: cd.ActorCritic, drive: np.ndarray, done: np.ndarray, warm: Any = None
) -> cd.BrainState:
    """``ActorCritic._next_state`` that ignores the sensed state."""
    warm = self._free
    assert warm is not None
    if done.any():
        v, a = warm.v.copy(), warm.adaptation.copy()
        v[done], a[done] = 0.0, 0.0
        warm = cd.BrainState(v, warm.activation, a, warm.steps)
    return self.learner.free(drive, warm=warm)


_SAVE, _LOAD = cd.Brain.save, cd.Brain.load


def _save_without_the_sensed_state(self: cd.Brain, path):
    written = _SAVE(self, path)

    def strip(metadata, arrays):
        if metadata.pop("awaiting", None) is not None:
            metadata["format"] = "cadence-generic/4" if metadata.get("efference") else (
                "cadence-generic/3"
            )
            for name in [name for name in arrays if name.startswith("awaiting/")]:
                del arrays[name]

    _rewrite(written, strip)
    return written


def _load_without_the_sensed_state(cls, path, **options):
    brain = _LOAD(path, **options)
    brain._awaiting = None
    return brain


MUTANTS: dict[str, tuple[type, str, Any]] = {
    **{
        "wait_" + variant: (cd.Brain, "wait", _mutant_wait(variant))
        for variant in (
            "takes_a_zero_outcome",
            "installs_the_sensed_state",
            "skips_the_working_trace",
            "fades_eligibility",
            "counts_a_lived_moment",
            "charges_no_work",
            "rewrites_the_command_copy",
            "keeps_an_unqualified_state",
            "waits_on_a_replaced_action",
        )
    },
    "decision_not_checked": (cd.Brain, "_owned", lambda self, decision: None),
    "settles_from_the_action_state": (
        cd.Brain, "_activity", lambda self: self.basal_ganglia.state
    ),
    "next_state_from_the_action_state": (
        cd.ActorCritic, "_next_state", _next_state_from_the_action
    ),
    "checkpoint_drops_the_sensed_state": (cd.Brain, "save", _save_without_the_sensed_state),
    "resume_drops_the_sensed_state": (
        cd.Brain, "load", classmethod(_load_without_the_sensed_state)
    ),
}


def failing_checks(tmp_path) -> list[str]:
    """The names of the contract checks that fail under the current methods."""
    failed = []
    for index, (name, check) in enumerate(sorted(CHECKS.items())):
        folder = tmp_path / f"{index:02d}"
        folder.mkdir()
        try:
            _run(check, folder)
        except (
            AssertionError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
            pytest.fail.Exception,  # an expected refusal that did not happen
        ):
            failed.append(name)
    return failed


def test_the_control_template_passes_every_check(monkeypatch, tmp_path):
    """The mutant template with no defect is the runtime statement: no check fails."""
    monkeypatch.setattr(cd.Brain, "wait", _mutant_wait("control"))
    assert failing_checks(tmp_path) == []


def test_every_mutant_is_killed(monkeypatch, tmp_path):
    """Each deliberate defect fails at least one contract check; the table names the killers."""
    rows = []
    for index, (name, (owner, attribute, function)) in enumerate(MUTANTS.items()):
        folder = tmp_path / f"m{index:02d}"
        folder.mkdir()
        with monkeypatch.context() as patched:
            patched.setattr(owner, attribute, function)
            rows.append((name, failing_checks(folder)))
    width = max(len(name) for name, _ in rows)
    print(f"\n{'mutant':<{width}}  killed by")
    for name, killers in rows:
        print(f"{name:<{width}}  {', '.join(killers) or 'SURVIVED'}")
    survivors = [name for name, killers in rows if not killers]
    print(f"mutation score {len(rows) - len(survivors)}/{len(rows)}")
    assert not survivors, survivors
