import spacy
import pandas as pd

#Global vars:
nlp = spacy.load('en_core_web_sm')

#GROUP ENTITIES:
#Companion function (input: doc sentence):
def group_entities(sent):
    chunks = []
    buf = []
    out = []

    #1) Store chunks to waiting list:
    for chunk in sent.noun_chunks:
        chunks.append(chunk)

    #2) Iterate through NEs in sentence:
    for ent in sent.ents:
        #If there are chunks in waiting list:
        if chunks != []:
            #Coordinates of first chunk:
            fc_st = chunks[0].start
            fc_end = chunks[0].end

            #CHECK start/end indexes of NE vs chunk:
            #a) isolated NE located before chunk:
            if (ent.start < fc_st):
                #store NE as definitive (cast to list):
                buf.append(ent[0].ent_type_)
                out.append(buf)
                buf = []
            
            #b) NE is part of chunk:
            if (ent.start >= fc_st) and (ent.start < fc_end):
                #store NE group as provisional:
                buf.append(ent[0].ent_type_)
                #check if end of chunk:
                if ent.end >= fc_end:
                    #remove chunk from waiting list:
                    chunks.remove(chunks[0])
                    #store NE group as definitive:
                    out.append(buf)
                    buf = []

        else:
            #Isolated NE (no chunks):
            #store NE group as definitive:
            buf.append(ent[0].ent_type_)
            out.append(buf)
            buf = []

    return out

#Key function (input: string path to file):
def groups_freqcount(pathfile):
    src = open(pathfile, 'r')
    freq_counts = {}

    for line in src:
        line = line.strip()
        sent_cl = nlp(line)
        groups_cl = group_entities(sent_cl)

        #save occurrence and count frequency:
        for g in groups_cl:
            gs = str(g)
            if gs in freq_counts.keys():
                freq_counts[gs] = freq_counts[gs] + 1
            else:
                freq_counts[gs] = 1
    src.close()
    return freq_counts

#extract max frequencies (function seen in class):
def nbest(d, n=1):
    #get n max values from a dict
    #param d: input dict (values are numbers, keys are stings)
    #param n: number of values to get (int)
    #return: dict of top n key-value pairs
    
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])


#MAIN:
#Part 1:
text = "Apple's Steve Jobs died in 2011 in Palo Alto, California."
sent = nlp(text)

groups = group_entities(sent)
print("Sample sentence: {}".format(groups))

#Part2:
#Reading file conll_str.txt exported in question 1 (conll2003 dataset converted to full string sentences):
freq_counts = groups_freqcount('./conll_str.txt')
#print(freq_counts)

#Total number of NEs:
t=0
for key in freq_counts:
    t = t + int(freq_counts[key])

#Most frequent groups / NE:
fq_best = nbest(freq_counts, 12)

#Normalization:
fq_best_norm = {}
for key in fq_best:
    fq_best_norm[key] = fq_best[key] / t

#Visualize freq analysis through Pandas DataFrame:
data = {'Count': fq_best, 'Normalized': fq_best_norm}
pd_tbl = pd.DataFrame().from_dict(data, orient='columns')

print("Full dataset:")
print(pd_tbl.round(decimals=2))
