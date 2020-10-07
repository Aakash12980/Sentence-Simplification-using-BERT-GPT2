def open_file(file_path):
    data = []
    with open(file_path, 'r', encoding="utf-8") as f:
        sents = f.readlines()
        for s in sents:
            data.append(s.strip())
    return data


