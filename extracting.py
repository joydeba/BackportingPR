'''
This model.py python file is part of ReBack, licensed under the CC0 1.0 Universal.
Details of the license can be found in the LICENSE file.
The current version of the ReBack can be always found at https://github.com/joydeba/BackportingPR
'''

def commit_id(commit):
    id = commit[0].strip().split(":")[1].strip()
    return id


def commit_port(commit):
    port = commit[1].strip().split(":")[1].strip()
    return port


def commit_date(commit):
    committer_date = commit[3].strip().split(":")[1].strip()
    return committer_date

def commit_meta(commit):
    commit_meta = commit[9].strip()
    return commit_meta

def commit_msg(commit):
    commit_msg = commit[6].strip()
    return commit_msg


def extract_hunk_code(code, sign):
    dict_hunk = {}
    for i in xrange(1, len(code)):
        l = code[i]
        if sign in l:
            try:
                hunk_idx = int(l.strip().split(":")[0])
            except ValueError:
                print "something wrong here"
                exit()
            line = l.strip().split(":")[3].strip()
            prop_line = l.strip().split(":")[2].strip()
            new_line = prop_line + ":" + line
            if hunk_idx not in dict_hunk.keys():
                dict_hunk[hunk_idx] = [new_line]
            else:
                dict_hunk[hunk_idx].append(new_line)
    return dict_hunk


def hunk_code(code):
    added_code = extract_hunk_code(code=code, sign="+")
    removed_code = extract_hunk_code(code=code, sign="-")
    return added_code, removed_code


def commit_code(commit):
    all_code = commit[12:]  
    file_index = [i for i, c in enumerate(all_code) if c.startswith("file:")]
    dicts = list()
    for i in xrange(0, len(file_index)):
        dict_code = {}
        if i == len(file_index) - 1:
            added_code, removed_code = hunk_code(all_code[file_index[i]:])
        else:
            added_code, removed_code = hunk_code(all_code[file_index[i]:file_index[i + 1]])
        dict_code[i] = all_code[file_index[i]].split(":")[1].strip()
        dict_code["added"] = added_code
        dict_code["removed"] = removed_code
        dicts.append(dict_code)
    return dicts













def extract_msg(commits):
    msgs = [" ".join(c["msg"].split(",")) for c in commits]
    return msgs

def extract_meta(commits):
    metas = [" ".join(p["meta"].split(",")) for p in commits]
    return metas

def extract_line_code(dict_code):
    lines = list()
    for k in dict_code.keys():
        for l in dict_code[k]:
            lines += l.split(":")[1].split(",")
            lines = [l.split(":")[0]] + lines
    return lines


def extract_code(commits):
    codes = list()
    for c in commits:
        line = list()
        for t in c["code"]:
            added_line, removed_line = extract_line_code(t["added"]), extract_line_code(t["removed"])
            line += added_line + removed_line
        codes.append(" ".join(line))
    return codes


def dictionary(data):
    # Dictionary for PR discussion
    lists = list()
    for m in data:
        lists += m.split()
    lists = list(set(lists))
    lists.append("NULL")
    new_dict = dict()
    for i in xrange(len(lists)):
        new_dict[lists[i]] = i
    return new_dict





