import spacy
from spacy.tokens import Doc
#place module in same directory:
import conll

#Global vars:
nlp = spacy.load('en_core_web_sm')
labels = {'B-ORG': 0, 'I-ORG': 0, 'B-LOC': 0, 'I-LOC': 0, 'B-PER': 0, 'I-PER': 0, 'B-MISC': 0, 'I-MISC': 0, 'O': 0}

#FUNCTIONS MODULE:

#Conll2003 conversion to list of strings:
def conll_to_str(src):
    #read file:
    f = open(src, 'r')

    #buffers:
    words = []
    buf = []

    #return vars:
    sent_ls = []
    tok_count = 0

    #ner_conll is dictionary with frequency count for all classes:
    ner_conll = labels.copy()
    #refs is full list of occurrencies of IOB tags:
    refs = []
    #tup_refs is list of lists, where each sentence is a list made of tuples (token text, IOB):
    tup_refs = []

    #Convert file to list of string sentences:
    for line in f:
        line = line.strip()
        #if line of interest, store word and IOB tag into a buffer:
        if len(line) > 0:
            spl = line.split()
            #print(spl[0])
            #ignore DOCSTART:
            if spl[0] != '-DOCSTART-':
                words.append(spl[0])
                tok_count = tok_count + 1
                #update count in frequency dict:
                ner_conll[spl[3]] = ner_conll[spl[3]] + 1
                #store occurrency:
                refs.append(spl[3])
                buf.append((spl[0], spl[3]))
            
        else:
            #If end of sentence, save buffer to return vars:
            if words != []:
                #convert words buffer to string:
                str1 = " "
                str1 = str1.join(words)
                #save string & tuple, then empty buffers:
                sent_ls.append(str1)
                words = []
                tup_refs.append(buf)
                buf = []

    f.close()
    #returns: list of strings (set_ls), token count, dict of NER labels & occurrencies, ref array for accuracy, ref array of tuples for conll.evaluate():
    return sent_ls, tok_count, ner_conll, refs, tup_refs

#Save to file (input: list of strings, name for the file):
def export_to_file(ls, name):
    fname = './' + str(name) + '.txt'
    outf = open(fname, 'w')

    for s in ls:
        outf.write(str(s)+'\n')

    outf.close()

    return 0

#Helper function to spacy_retok(): doc retokenizer by sentence:
def sent_retok(doc_sent):
    #the "10000" value for "a" means that no initial span position has been already set:
    a = 10000
    b = 0
    i = 0
    last = len(list(doc_sent))

    #Check for spans to merge into single token:
    with doc_sent.retokenize() as retokenizer:
        for token in doc_sent:
            #If token is joint to other words:
            if token.whitespace_ == '':
                #accumulate indexes:
                if a == 10000:
                    #set start index:
                    a = i
                    b = i
                else:
                    #update end index:
                    b = i

            #if come out of span or reached sentence end:
            if (token.whitespace_ != '') or (i == (last - 1)):
                #check if accumulate span to merge:
                if a != 10000:
                    #merge span:
                    b = b+2
                    retokenizer.merge(doc_sent[a:b])
                    #reset indexes:
                    a = 10000
                    b = 0

            i = i+1

    #returns sentence doc:
    return doc_sent

#SPACY DOC RETOKENIZER (align tokenization to original input):
def spacy_retok(conll_str):
    conll_ret = []
    ret_count = 0

    #retokenization is done sentence by sentence:
    for i in range(len(conll_str)):
        #convert to doc & retokenize sentence by sentence:
        doc_sent = nlp(conll_str[i])
        sent_ret = sent_retok(doc_sent)
        #save to list of docs:
        conll_ret.append(sent_ret)
        #save count of tokens after retokenization:
        ret_count = ret_count + len(list(sent_ret))
    
    #returns: list of sentence docs (conll_ret), new token count:
    return conll_ret, ret_count

#LABEL REMAPPING: returns dict with count of occurrencies of remapped labels & hyps array:
def labels_remap(conll_ret):
    #remapping legend:
    remap = {'CARDINAL': 'MISC', 'DATE': 'MISC', 'EVENT': 'MISC', 'FAC': 'LOC', 'GPE': 'LOC', 'LANGUAGE': 'MISC', 'LAW': 'MISC', 'LOC': 'LOC', 'MONEY': 'MISC', 'NORP': 'ORG', 'ORDINAL': 'MISC', 'ORG': 'ORG', 'PERCENT': 'MISC', 'PERSON': 'PER', 'PRODUCT': 'MISC', 'QUANTITY': 'MISC', 'TIME': 'MISC', 'WORK_OF_ART': 'MISC', '': '', 'O': ''}

    #ner_spacy is dictionary with frequency count for all classes:
    ner_spacy = labels.copy()
    #buffer:
    buf = []
    #hyps is full list of occurrencies of IOB tags:
    hyps = []
    #tup_hyps is list of lists, where each sentence is a list made of tuples (token text, IOB):
    tup_hyps = []

    #remapping labels sentence by sentence:
    for s in conll_ret:
        #REMAPPING:
        for t in s:
            #label remapping:
            if str(t.ent_iob_) == 'O':
                str1 = str(t.ent_iob_)
            else:
                str1 = str(t.ent_iob_) + "-" + str(remap[t.ent_type_])

            #saving to dict:
            ner_spacy[str1] = ner_spacy[str1] + 1
            hyps.append(str1)
            buf.append((str(t.text), str1))

        tup_hyps.append(buf)
        buf = []

    #returns: dict of remapped NER labels & occurrencies, hyp array for accuracy, hyp array of tuples for conll.evaluate():
    return ner_spacy, hyps, tup_hyps

