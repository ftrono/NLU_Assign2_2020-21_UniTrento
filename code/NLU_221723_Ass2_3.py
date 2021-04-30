import spacy
from spacy.tokens import Span
import pandas as pd
#place modules in current directory:
import conll
import NLU_221723_Ass2_1_functions as q1

#Global vars:
nlp = spacy.load('en_core_web_sm')
labels = {'B-ORG': 0, 'I-ORG': 0, 'B-LOC': 0, 'I-LOC': 0, 'B-PER': 0, 'I-PER': 0, 'B-MISC': 0, 'I-MISC': 0, 'O': 0}

#Functions from previous question 1 are imported from module "NLU_221723_Ass2_1_functions" (q1).

#FUNCTIONS:
#Companion function (input: single doc sentence):
def sent_extend(sent):

    #vars:
    comps = []
    ents_ls = []
    orig_ent = []
    label = ''

    #bools:
    comp_is_ent = False
    head_is_ent = False
    diff_gran = False
    unlock = False

    #span indexes:
    #the "1000" value for "a" means that no initial span position has been already set:
    a = 1000
    b = 0


    #0) Store compounds:
    for tok in list(sent):
        if str(tok.dep_) == 'compound':
            comps.append(tok)

    #Store ents:
    for ent in list(sent.ents):
        ents_ls.append(list(ent))
    
    #KEY ASSUMPTION: compound is always on the left, so I iterate from first token on:
    #For each compound:
    for comp in comps:
        i = comp.i

        #1) Check:
        #a) if current compound is a NE:
        if str(comp.ent_type_) == '':
            comp_is_ent = False
        else:
            comp_is_ent = True
            for ent in ents_ls:
                if comp in ent:
                    orig_ent = ent

        #b) if head is a NE:
        head = comp.head
        if str(head.ent_type_) == '':
            head_is_ent = False
        else:
            head_is_ent = True
            #priority to head ent over comp ent, if both:
            for ent in ents_ls:
                if head in ent:
                    orig_ent = ent

        #c) if head is a compound:
        if head in comps:
            #check head.head.ent_type:
            gran = head.head
            if (str(gran.ent_type_) != '') and ((head_is_ent == True) and ((str(gran.ent_type_) != str(head.ent_type_))) or ((comp_is_ent == True) and (str(gran.ent_type_) != str(comp.ent_type_)))):
                diff_gran = True

        #Choose label to assign:
        if label == '':
            #if both ents:
            if (comp_is_ent == True) and (head_is_ent == True):
                #check if same type:
                if str(comp.ent_type_) == str(head.ent_type_):
                    label = str(head.ent_type_)
            #if one only is ent:
            elif (comp_is_ent == True) and (head_is_ent == False):
                label = str(comp.ent_type_)
            elif (comp_is_ent == False) and (head_is_ent == True):
                label = str(head.ent_type_)

        #EARLY STOP CASE:
        #separate if all out of a NE:
        if (a != 1000) and (comp_is_ent == False) and (head_is_ent == False) and (diff_gran == False):
            b = i
            unlock = True

        #CONSTRUCT SPAN:
        #if can go ahead:
        if label != '':
            #compound is always on the left.
            #So, I define the start of the new entity span ("a") as the current compound position, if this has not been set before:
            if a == 1000:
                a = comp.i

            
            #Checks for span end:
            #next token:
            nxt = sent[i+1]
            #If out of current span to extend:
            if (nxt not in comps):
                b = nxt.i
                unlock = True
            #other conditions:
            elif diff_gran == True:
                b = i
                unlock = True
            elif unlock == True:
                b = i

            if unlock == True:
                #PREPARE MERGE:
                new_ent = Span(sent, a, b+1, label=label)

                #if the entity will be actually expanded, MERGE:
                if len(list(new_ent)) > len(orig_ent):
                    #set ent (don't modify tokens outside new_ent):
                    sent.set_ents([new_ent], default='unmodified')

                a = 1000
                b = 0
                label = ''
                diff_gran = False
                unlock = False

    return sent

#Key function: NE_EXTENDER (input: list of doc sentences):
def ne_extend(sent_ls):
    ext_ls = []

    for sent in sent_ls:
        ext_sent = sent_extend(sent)
        ext_ls.append(ext_sent)

    return ext_ls

#Label visualizer (input: doc sentence, output: list of tuples):
def label_visualizer(sent):
    ls = []
    comps = []
    ents_ls = []

    for tok in sent:
        if tok.ent_iob_ != 'O':
            str2 = '-' + str(tok.ent_type_)
        else:
            str2 = ''
        str1 = str(tok.ent_iob_) + str2
        ls.append((tok.text, str1))

        if tok.dep_ == 'compound':
            comps.append([tok, tok.head, tok.dep_])

    for ent in sent.ents:
        ents_ls.append(ent)

    return ls, comps, ents_ls


#MAIN:
#1) SAMPLE SENTENCE:
print("1) SAMPLE SENTENCE:")
#Various test sentences to try:
test = nlp("The press office explained that a regulation draft was requested last month by EU President Jean Claude Junker.")

#test = nlp("How beautiful is the Rio Grande river bank, the EU President said.")

#test = nlp("Juventus Team will play today at the new Turin stadium.")

#test = nlp("The new Turin stadium owner Sheik O'Dollar will build additional spaces for the audience.")

#BEFORE:
ls, bef_comps, bef_ents  = label_visualizer(test)
print("BEFORE:")
print(ls)

print('COMPOUNDS:')
for l in bef_comps:
    print(l)

print("BEFORE ENTITIES:")
print(bef_ents)

#AFTER:
#using function for single sentence:
doc_ext = sent_extend(test)
new_ls, aft_comps, aft_ents = label_visualizer(doc_ext)
print("AFTER:")
print(new_ls)

print("AFTER ENTITIES:")
print(aft_ents)


#2) EVALUATION ON CONLL_2003:
print("\n2) EVALUATION ON CONLL_2003:")
#Repeating steps of q1 from scratch:
#place 'src' folder (available in "data" subrep) in current directory:
src = './src/conll2003/dev.txt'

#1) Convert conll_2003 dataset to list of strings. Obtain: list of string sentences from conll_2003 and array tup_refs for conll.evaluate() function:
conll_str, src_count, _, _, tup_refs = q1.conll_to_str(src)

#2) Convert list of strings to aligned list of Spacy docs. Obtain: list of docs sentences retokenized:
conll_ret, ret_count = q1.spacy_retok(conll_str)

#Check tokenization alignment:
print("SRC COUNT: {}".format(src_count))
print("RET COUNT: {}".format(ret_count))

#3) Launch NEW FUNCTION ne_extend() to extend NEs with relevant compounds. Obtain: list of sentence docs with new NE labels for the tokens:
conll_ext = ne_extend(conll_ret)

#4) Remap labels in spacy retokenized and NE-extended list of doc sentences:
ner_spacy, _, tup_hyps = q1.labels_remap(conll_ext)

#5) Launching conll.evaluate() on NE-extended input:
ext_res = conll.evaluate(tup_refs, tup_hyps)

pd_tbl = pd.DataFrame().from_dict(ext_res, orient='index')
print("CONLL EVALUATION RESULTS:")
print(pd_tbl.round(decimals=3))

