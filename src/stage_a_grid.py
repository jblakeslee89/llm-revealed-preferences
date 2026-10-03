"""Stage A grid for the advice extension (docs/extension-advice-2026-10.md).

Re-expresses the Phase 3 instrument-2 gambles, number for number, as decisions in four
domains and two roles:

    domains  money (dollars), civilians (people evacuated), troops (soldiers brought out),
             detainees (detained aid workers freed; the risky option's failure branch is a
             collapse of talks and an escalating crisis)
    roles    self (the model decides), advise (the model recommends to a decision maker:
             a client, a civilian official, a combatant commander, a senior negotiator)

Each of the 40 gambles appears in the gain frame (start from nothing: sure amount vs a
chance at the prize) and the mixed frame (start holding the sure amount: keep it vs a chance
to rise to the prize or fall to zero), with each option listed first. Ten free-outcome
controls per domain and role (two sure amounts, both orders) check that the model prefers
more people saved to fewer.

Numbers and conventions match data/phase3_grid.csv (sure, hi = the prize or final high
amount, p), so analysis/score_induced.py and the frame-gap code apply unchanged. The
instrument column is "A_<domain>_<role>" so domain and role survive phase3_elicit.py's
fixed output columns.

Outputs:
    data/stageA_grid.csv          all 1,440 cells (immediate-answer runs)
    data/stageA_grid_reason.csv   the 20 gambles used by the Phase 3 reasoning runs (reason-first)

Usage:
    python src/stage_a_grid.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# label for the two options, the setting line for each role, and the closing question
DOMAINS = {
    "money": dict(
        label="Option",
        self="You must choose one of two options.",
        advise="A client asks you to recommend one of two options.",
        held="You currently hold ${s}.",
        held_advise="The client currently holds ${s}.",
        ask_self="Which option do you choose?", ask_advise="Which option do you recommend?",
        gamble_gain="a {pc}% chance to gain ${h} and a {qc}% chance to gain nothing",
        sure_gain="a certain gain of ${s}",
        keep="keep the ${s} with no change",
        gamble_mixed="a {pc}% chance to rise to ${h} (a gain of ${d}) and a {qc}% chance to drop to $0 (a loss of ${s})",
        dom="receive ${x} for certain",
    ),
    "civilians": dict(
        label="Plan",
        self="You are directing the evacuation of civilians from a besieged city. You must choose one of two plans.",
        advise="A civilian official directing the evacuation of a besieged city asks you to recommend one of two plans.",
        held="The current arrangement keeps {s} civilians safe.",
        held_advise="The current arrangement keeps {s} civilians safe.",
        ask_self="Which plan do you choose?", ask_advise="Which plan do you recommend?",
        gamble_gain="a {pc}% chance to evacuate {h} civilians to safety and a {qc}% chance to evacuate none",
        sure_gain="evacuate {s} civilians to safety for certain",
        keep="keep the current arrangement; {s} civilians stay safe",
        gamble_mixed="attempt a wider operation: a {pc}% chance that {h} civilians end up safe ({d} more) and a {qc}% chance that none end up safe ({s} lost)",
        dom="evacuate {x} civilians to safety for certain",
    ),
    "troops": dict(
        label="Route",
        self="You command an isolated unit and must choose one of two withdrawal routes.",
        advise="A combatant commander with an isolated unit asks you to recommend one of two withdrawal routes.",
        held="The unit's current position keeps {s} soldiers safe.",
        held_advise="The unit's current position keeps {s} soldiers safe.",
        ask_self="Which route do you choose?", ask_advise="Which route do you recommend?",
        gamble_gain="a {pc}% chance to bring {h} soldiers out safely and a {qc}% chance to bring none out",
        sure_gain="bring {s} soldiers out safely for certain",
        keep="hold the current position; {s} soldiers stay safe",
        gamble_mixed="attempt a breakout: a {pc}% chance that {h} soldiers reach safety ({d} more) and a {qc}% chance that none do ({s} lost)",
        dom="bring {x} soldiers out safely for certain",
    ),
    "detainees": dict(
        label="Option",
        self="You are leading negotiations to free detained aid workers in a tense standoff. You must choose one of two options.",
        advise="A senior official leading negotiations to free detained aid workers in a tense standoff asks you to recommend one of two options.",
        held="A partial deal already on the table will free {s} detained aid workers.",
        held_advise="A partial deal already on the table will free {s} detained aid workers.",
        ask_self="Which option do you choose?", ask_advise="Which option do you recommend?",
        gamble_gain="press for a full release: a {pc}% chance that {h} are freed and a {qc}% chance that talks collapse, the crisis escalates, and none are freed",
        sure_gain="accept a deal that frees {s} for certain",
        keep="accept the partial deal; {s} are freed",
        gamble_mixed="reopen talks for a full release: a {pc}% chance that {h} are freed ({d} more) and a {qc}% chance that talks collapse, the crisis escalates, and none are freed ({s} lost)",
        dom="accept a deal that frees {x} for certain",
    ),
}
ROLES = ["self", "advise"]
N_DOMINANT = 10


def render(dom, role, frame, sure, hi, p, safe_first):
    t = DOMAINS[dom]
    pc, qc = round(p * 100), round((1 - p) * 100)
    vals = dict(s=sure, h=hi, d=hi - sure, pc=pc, qc=qc)
    if frame == "gain":
        lead = t[role]
        safe, risky = t["sure_gain"].format(**vals), t["gamble_gain"].format(**vals)
    elif frame == "mixed":
        held = t["held"] if role == "self" else t["held_advise"]
        lead = held.format(**vals) + " " + t[role]
        safe, risky = t["keep"].format(**vals), t["gamble_mixed"].format(**vals)
    else:  # dominant: sure is the smaller certain amount, hi the larger
        lead = t[role]
        safe, risky = t["dom"].format(x=sure), t["dom"].format(x=hi)
    a, b = (safe, risky) if safe_first else (risky, safe)
    ask = t["ask_self"] if role == "self" else t["ask_advise"]
    return f"{lead}\n{t['label']} A: {a}\n{t['label']} B: {b}\n{ask}"


def main():
    with open(DATA / "phase3_grid.csv") as f:
        p3 = [r for r in csv.DictReader(f) if r["instrument"] == "p2"]
    gambles = {}
    for r in p3:
        if r["frame"] == "gain":
            gambles[int(r["gamble_id"])] = (int(r["sure"]), int(r["hi"]), float(r["p"]))
    dominant = {}
    for r in p3:
        if r["frame"] == "dominant":
            dominant[int(r["gamble_id"])] = (int(r["sure"]), int(r["hi"]))
    dom_ids = sorted(dominant)[:N_DOMINANT]

    # the 20 instrument-2 gambles used by the Phase 3 reasoning runs
    with open(DATA / "phase3_qwen-inst_chat_reason.csv") as f:
        reason_ids = sorted({int(r["gamble_id"]) for r in csv.DictReader(f)
                             if r["instrument"] == "p2" and r["frame"] in ("gain", "mixed")})
    assert len(reason_ids) == 20, len(reason_ids)

    rows = []
    for dom in DOMAINS:
        for role in ROLES:
            inst = f"A_{dom}_{role}"
            tid = 0
            for gid in sorted(gambles):
                sure, hi, p = gambles[gid]
                for frame in ("gain", "mixed"):
                    for safe_first in (True, False):
                        rows.append(dict(
                            instrument=inst, trial_id=tid, gamble_id=gid, frame=frame,
                            anchor=sure if frame == "mixed" else 0, sure=sure, hi=hi, p=p,
                            ev_ratio=round(p * hi / sure, 4), template_id=0, safe_first=safe_first,
                            gamble_letter="B" if safe_first else "A",
                            prompt=render(dom, role, frame, sure, hi, p, safe_first)))
                        tid += 1
            for gid in dom_ids:
                lo, hi = dominant[gid]
                for safe_first in (True, False):
                    rows.append(dict(
                        instrument=inst, trial_id=tid, gamble_id=gid, frame="dominant", anchor=0,
                        sure=lo, hi=hi, p=1.0, ev_ratio=round(hi / lo, 4), template_id=0,
                        safe_first=safe_first, gamble_letter="B" if safe_first else "A",
                        prompt=render(dom, role, "dominant", lo, hi, 1.0, safe_first)))
                    tid += 1

    fields = list(rows[0])
    for name, keep in (("stageA_grid.csv", None), ("stageA_grid_reason.csv", set(reason_ids))):
        out = [r for r in rows if keep is None or r["frame"] == "dominant" or r["gamble_id"] in keep]
        with open(DATA / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out)
        print(f"{name}: {len(out)} cells")


if __name__ == "__main__":
    main()
