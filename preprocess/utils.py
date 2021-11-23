import re


def is_ent(tp_str):
    if len(tp_str) < 3:
        return False
    if tp_str.startswith("m.") or tp_str.startswith("g."):
        # print(tp_str)
        return True
    return False


def find_entity(sparql_str):
    str_lines = sparql_str.split("\n")
    ent_set = set()
    for line in str_lines[1:]:
        if "ns:" not in line:
            continue
        literals = re.findall('".*"@en', line)
        if literals:
            literals = [x[1:-4] for x in literals]
            print(literals)
            ent_set.update(literals)
        spline = line.strip().split()
        for item in spline:
            ent_str = item[3:].replace("(", "")
            ent_str = ent_str.replace(")", "")
            if is_ent(ent_str):
                ent_set.add(ent_str)

    return ent_set


def load_dict(path):
    obj2idx, idx2obj = dict(), list()
    with open(path) as f:
        for line in f:
            line = line.strip()
            obj2idx[line] = len(obj2idx)
            idx2obj.append(line)
    return obj2idx, idx2obj
