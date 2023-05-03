def get_lines(filename):
    with open(filename,'r') as f:
        return f.readlines()
    
def preprocess(filename):
    returnable_list = []
    input_lines = get_lines(filename)

    for line in input_lines:
        if line.startswith('\n'):
            abstract_lines_split = abstract_lines.splitlines()
            for line_num,line in enumerate(abstract_lines_split):
                dictionary = {}
                tag_and_line = line.split('\t')
                dictionary['target'] = tag_and_line[0]
                dictionary['text'] = tag_and_line[1]
                dictionary['line_number'] = line_num + 1
                dictionary['total_lines'] = len(abstract_lines_split)
                returnable_list.append(dictionary)
        elif line.startswith('###'):
            abstract_lines = ''
            pass
        else:
            abstract_lines += line
    return returnable_list

def split_chars(sentence):
    return ' '.join(list(sentence))