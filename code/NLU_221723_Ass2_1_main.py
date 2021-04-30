import spacy
from spacy.tokens import Doc
from sklearn.metrics import classification_report
import pandas as pd
#place modules in current directory:
import conll
import NLU_221723_Ass2_1_functions as q1

#Global vars:
nlp = spacy.load('en_core_web_sm')
labels = {'B-ORG': 0, 'I-ORG': 0, 'B-LOC': 0, 'I-LOC': 0, 'B-PER': 0, 'I-PER': 0, 'B-MISC': 0, 'I-MISC': 0, 'O': 0}

#Functions are imported from module "NLU_221723_Ass2_1_functions" (q1).

#MAIN MODULE:
#place 'src' folder (available in "data" subrep) in current directory:
src = './src/conll2003/dev.txt'

#1) Convert conll_2003 dataset to list of strings:
conll_str, src_count, ner_conll, refs, tup_refs = q1.conll_to_str(src)
#save to file (for next questions):
q1.export_to_file(conll_str, 'conll_str')

#2) Convert list of strings to aligned list of Spacy docs:
conll_ret, ret_count = q1.spacy_retok(conll_str)

#3) Remap labels in spacy retokenized list of doc sentences:
ner_spacy, hyps, tup_hyps = q1.labels_remap(conll_ret)

#PRINT KEY OUTPUT INFO:
print("SRC COUNT: {}".format(src_count))
print("RET COUNT: {}".format(ret_count))
print(ner_conll)
print(ner_spacy)

#1.1 EVALUATION: TOKEN-LEVEL & CLASS-LEVEL:
report = classification_report(refs, hyps)
print("\n1.1 EVALUATION: TOKEN-LEVEL & CLASS-LEVEL:")
print(report)

#1.2 EVALUATION: CHUNK-LEVEL:
chunk_res = conll.evaluate(tup_refs, tup_hyps)

pd_tbl = pd.DataFrame().from_dict(chunk_res, orient='index')
print("\n1.2 EVALUATION: CHUNK-LEVEL:")
print(pd_tbl.round(decimals=3))
