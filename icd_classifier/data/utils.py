from collections import defaultdict
from icd_classifier.settings import DATA_DIR


def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits, 
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code


def load_code_descriptions():
    # load description lookup from the appropriate data files
    desc_dict = defaultdict(str)
    with open("%s/D_ICD_DIAGNOSES.csv" % (DATA_DIR), 'r') as descfile:
        r = csv.reader(descfile)
        #header
        next(r)
        for row in r:
            code = row[1]
            desc = row[-1]
            desc_dict[reformat(code, True)] = desc
    with open("%s/D_ICD_PROCEDURES.csv" % (DATA_DIR), 'r') as descfile:
        r = csv.reader(descfile)
        #header
        next(r)
        for row in r:
            code = row[1]
            desc = row[-1]
            if code not in desc_dict.keys():
                desc_dict[reformat(code, False)] = desc
    with open('%s/ICD9_descriptions' % DATA_DIR, 'r') as labelfile:
        for i,row in enumerate(labelfile):
            row = row.rstrip().split()
            code = row[0]
            if code not in desc_dict.keys():
                desc_dict[code] = ' '.join(row[1:])
    return desc_dict