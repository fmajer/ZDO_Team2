import xmltodict


def load_anns(path):
    with open(path) as fd:
        anns = xmltodict.parse(fd.read())
    return anns
