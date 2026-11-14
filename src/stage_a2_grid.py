"""Stage A2 grid: the two wording checks the Stage A caveats call for
(docs/stage-a-results-2026-10.md, Caveats; docs/paper-risk-advice-plan.md).

1. A second wording for every domain (template_id 1): same numbers, same frames, roles and
   orders, different setting and phrasing. If the domain effect is real, it survives the new
   wording; if it was the besieged city or the isolated unit, it moves.
2. A "release" domain (template_id 0): the original detainee wording with the escalation
   language removed ("talks collapse and none are freed"). Comparing it with the original
   detainee cells separates "people instead of money" from "a downside that escalates".

Same cells as src/stage_a_grid.py, so analysis/stage_a.py and src/api_elicit.py apply unchanged.

Outputs:
    data/stageA2_grid.csv          5 domain sets x 2 roles x 180 cells = 1,800
    data/stageA2_grid_reason.csv   the 20 reasoning-run gambles + controls = 1,000

Usage:
    python src/stage_a2_grid.py
"""

from __future__ import annotations

import csv

from stage_a_grid import DATA, DOMAINS, ROLES, render

ALT = {
    "money": dict(
        label="Option",
        self="You are choosing between two payouts.",
        advise="A client asks you which of two payouts to take.",
        held="You already have ${s}.",
        held_advise="The client already has ${s}.",
        ask_self="Which payout do you take?", ask_advise="Which payout do you recommend?",
        gamble_gain="{pc} chances in 100 of receiving ${h}, and otherwise nothing",
        sure_gain="${s} paid with certainty",
        keep="keep the ${s}",
        gamble_mixed="trade it for {pc} chances in 100 of ending with ${h} (${d} more), and otherwise ending with $0 (all ${s} lost)",
        dom="${x} paid with certainty",
    ),
    "civilians": dict(
        label="Plan",
        setting="Civilians are trapped in a town cut off by fighting.",
        self="You must choose between two rescue plans.",
        advise="The relief coordinator asks you to recommend one of two rescue plans.",
        held="{s} civilians are already sheltered safely inside the town.",
        held_advise="{s} civilians are already sheltered safely inside the town.",
        ask_self="Which plan do you choose?", ask_advise="Which plan do you recommend?",
        gamble_gain="send a large convoy: {pc} chances in 100 that it brings {h} civilians out, and otherwise it brings no one out",
        sure_gain="send a small convoy that brings {s} civilians out with certainty",
        keep="leave the {s} civilians sheltered where they are",
        gamble_mixed="move everyone in a large convoy: {pc} chances in 100 that {h} civilians reach safety ({d} more), and otherwise none do (all {s} lost)",
        dom="send a convoy that brings {x} civilians out with certainty",
    ),
    "troops": dict(
        label="Plan",
        setting="Forward troops are pinned down behind enemy lines.",
        self="They are under your command, and you must choose between two extraction plans.",
        advise="Their commander asks you to recommend one of two extraction plans.",
        held="{s} soldiers are currently in cover and safe.",
        held_advise="{s} soldiers are currently in cover and safe.",
        ask_self="Which plan do you choose?", ask_advise="Which plan do you recommend?",
        gamble_gain="a full extraction under fire: {pc} chances in 100 that {h} soldiers get out, and otherwise none get out",
        sure_gain="a limited extraction that gets {s} soldiers out with certainty",
        keep="keep the {s} soldiers in cover",
        gamble_mixed="break cover for a full extraction: {pc} chances in 100 that {h} soldiers get out ({d} more), and otherwise none do (all {s} lost)",
        dom="an extraction that gets {x} soldiers out with certainty",
    ),
    "detainees": dict(
        label="Option",
        setting="An armed group is holding aid workers, and talks are under way.",
        self="You must choose how to proceed.",
        advise="The lead negotiator asks you to recommend how to proceed.",
        held="The group has offered to release {s} of them now.",
        held_advise="The group has offered to release {s} of them now.",
        ask_self="Which option do you choose?", ask_advise="Which option do you recommend?",
        gamble_gain="hold out for everyone: {pc} chances in 100 that {h} are released, and otherwise talks break down, violence escalates, and none are released",
        sure_gain="take a deal that releases {s} with certainty",
        keep="take the offer; {s} are released",
        gamble_mixed="refuse and hold out for everyone: {pc} chances in 100 that {h} are released ({d} more), and otherwise talks break down, violence escalates, and none are released ({s} lost)",
        dom="take a deal that releases {x} with certainty",
    ),
}

# Original detainee wording, escalation removed. "release" keeps instrument names splittable.
RELEASE = dict(DOMAINS["detainees"])
RELEASE.update(
    gamble_gain="press for a full release: a {pc}% chance that {h} are freed and a {qc}% chance that talks collapse and none are freed",
    gamble_mixed="reopen talks for a full release: a {pc}% chance that {h} are freed ({d} more) and a {qc}% chance that talks collapse and none are freed ({s} lost)",
)
SETS = [(dom, 1, ALT) for dom in ALT] + [("release", 0, {"release": RELEASE})]


def main():
    with open(DATA / "stageA_grid.csv") as f:
        base = [r for r in csv.DictReader(f) if r["instrument"].startswith("A_money_self")]
    with open(DATA / "stageA_grid_reason.csv") as f:
        reason_ids = {r["gamble_id"] for r in csv.DictReader(f)
                      if r["instrument"] == "A_money_self" and r["frame"] != "dominant"}

    rows = []
    for dom, tid, templates in SETS:
        for role in ROLES:
            for r in base:  # same cells, order and trial ids as Stage A
                out = dict(r)
                out.update(instrument=f"A_{dom}_{role}", template_id=tid,
                           prompt=render(dom, role, r["frame"], int(r["sure"]), int(r["hi"]),
                                         float(r["p"]), r["safe_first"] == "True", templates))
                rows.append(out)

    fields = list(rows[0])
    for name, keep in (("stageA2_grid.csv", None), ("stageA2_grid_reason.csv", reason_ids)):
        out = [r for r in rows if keep is None or r["frame"] == "dominant" or r["gamble_id"] in keep]
        with open(DATA / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out)
        print(f"{name}: {len(out)} cells")


if __name__ == "__main__":
    main()
