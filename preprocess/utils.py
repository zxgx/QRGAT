

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
        spline = line.strip().split(" ")
        for item in spline:
            ent_str = item[3:].replace("(", "")
            ent_str = ent_str.replace(")", "")
            if is_ent(ent_str):
                ent_set.add(ent_str)

    return ent_set
