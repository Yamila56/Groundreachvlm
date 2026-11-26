"""Certified question and counterfactual pair counts."""

from collections import Counter, defaultdict


def counts(rows, values):
    known = [row for row in rows if row["truth"] != "U"]
    views, targets, maps = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        views[row["scene"]].append(row)
        targets[row["scene"], row["side"]].append(row)
        maps[row["scene"], row["target"]].append(row)

    def correct(row):
        return values[row["id"]] == row["truth"]

    def joint(groups, size, opposite=False):
        selected = [group for group in groups.values()
                    if len(group) == size and all(row["truth"] != "U" for row in group)
                    and (not opposite or {row["truth"] for row in group} == {"Y", "N"})]
        return dict(total=len(groups), certified=len(selected),
                    correct=sum(all(map(correct, group)) for group in selected))

    return dict(certified=len(known), correct=sum(map(correct, known)),
                unanswered=sum(values[row["id"]] == "U" for row in known),
                confusion=dict(Counter(f"{row['truth']}-{values[row['id']]}" for row in rows)),
                target_flip=joint(targets, 2, opposite=True), workspace_flip=joint(maps, 2, opposite=True),
                four=joint(views, 4))
