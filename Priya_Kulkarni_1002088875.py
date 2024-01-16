#Name: Priya Kulkarni
#UTA_ID: 1002088875

import os
import math
from math import log10, sqrt
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

fld= "US_Inaugural_Addresses" # folder name containing the datasets
vect={}
term = Counter()
r_tfs={}
lengths={}
final = Counter()

# getidf function
def getidf(tok): # caluculate the inverse document frequency for given token
    #tok = PorterStemmer().stem(tok) # stem the token
    #print(tok)
    return log10(len(r_tfs) / term[tok]) if tok in term.keys() and term[tok] != 0 else -1 

def getweight(fname,tok):
    root_dir = os.getcwd()
    fld_P = os.path.join(root_dir,fld,fname)
    #print(vect)
    #t = PorterStemmer().stem(tok) # stem the token
    #print(t)
    #print(vect[fld_P])
    return vect[fld_P][tok]  # get the  weights from the vector

# document query matching
def query(q):
    # lower case the query
    q = q.lower()
    ten = {} 
    cos=Counter() # cosin value
    q_len = 0 # length
    # raw term frequenct dictionary
    q_rtf={}
    docs= {} # docs dictionary
    for t in q.split(): # split the query to get the terms
        #print(t)
        t = PorterStemmer().stem(t) # stem the terms
        #print(final)
        if t not in final:
            continue
        # get the weight
        if getidf(t) == 0:  # if getidf is 0 
            docs[t], w = zip(*final[t].most_common()) 
        else:
            docs[t], w = zip(*final[t].most_common(10))  #top 10 most common documents.
        #print(w[-1])
        ten[t] = w[-1] #least common document
        count_t = q.count(t) # raw term frequency
        #print(count_t)
        if count_t != 0:
            q_rtf[t] = 1 + log10(count_t)  #applying a logarithmic scaling term frequency
        else:
            q_rtf[t] = 0
        #print(q_rtf[t])
        q_len += q_rtf[t]**2 #the squares of the term frequencies to caslulate the length
    q_len=sqrt(q_len)
    # Cosine similarity calculated
    for d in vect:
         cos[d] = sum(float(q_rtf[t] / q_len) * (final[t][d] if d in docs[t] else ten[t]) for t in q_rtf)
    #print(type(cos))
    #finds the document with the highest cosine similarity score 
    result = cos.most_common(1)
    f_res, weig = zip(*result)
    #return None if highest score is 0
    return ("None", weig[0]) if weig[0] == 0 else (f_res[0], weig[0])

# retrieve the files path recursively
def list_txt_files_recursive(directory):
    txt_file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            full_path = os.path.join(root, filename)
            _, file_extension = os.path.splitext(filename)
            if file_extension.lower() == '.txt':
                txt_file_paths.append(full_path)
    return txt_file_paths
  
def Preprocess():
    root_dir = os.getcwd()
    f_path = os.path.join(root_dir,fld, "*.txt")
    #print(f_path)
    # get path of every file in the folder
    allfile_paths= list_txt_files_recursive(os.path.join(root_dir,fld))
    #print(allfile_paths)
    
    for i in allfile_paths: # each file 
        with open(i) as f1:
            line = f1.read()
            lowcased = line.lower() # convert into lower case
        #print(lowcased)
        
        # Tokenzie
        token= RegexpTokenizer(r'[a-zA-Z]+').tokenize(lowcased) # tokenzie \
        
        # remove stop words
        stopwords_removed = [] # create list to store the stopwords removed
        for j in token: 
                if j not in stopwords.words('english'):
                    stopwords_removed.append(j)   # append to the stop words
        
        #stemming process
        stem_tok =[] # stemmed tokens list
        for j in stopwords_removed:
            stemm = PorterStemmer().stem(j) # apply stemmer
            stem_tok.append(stemm)
        #print(stem_tok)

        # Get the Raw Term frequency
        raw_tf= Counter()  # Create an empty rwo term frequency dictionary to store counts
        for item in stem_tok:
            if item in raw_tf:
                raw_tf[item] += 1
            else:
                raw_tf[item] = 1
        #print(raw_tf)

        # to get the unique term list for the document
        global term
        remove_dup= list(set(stem_tok)) # remove the duplicate terms for getting unique term value
        for item in remove_dup:
            if item in term:
                term[item] += 1
            else:
                term[item] = 1
        #print(term)

        r_tfs[i.split()[-1]] = raw_tf.copy()
        #print(tfs)
        raw_tf.clear()
    
    # creating the vectors for each document
    for rtf in r_tfs:
        vect[rtf] = Counter() # create empty vector dict for each document
        v = 0
        for t in r_tfs[rtf]:
            idf= getidf(t)          # get idf value
            z = 1 + log10(r_tfs[rtf][t]) 
            w = z * idf             # formulae to calculate
            vect[rtf][t] = w
            v += w**2
        #print(vect)
        lengths[rtf] = math.sqrt(v)
    # normalizing the vector by dividing the length
    for rtf in vect:
        for t in vect[rtf]:
            vect[rtf][t] = vect[rtf][t] / lengths[rtf]
            if t not in final:
                final[t] = Counter()
            final[t][rtf] = vect[rtf][t] # final vector normalized


def  main():
    #preprocess function to process the documents
    Preprocess()
    print("%.12f" % getidf('british'))
    print("%.12f" % getidf('union'))
    print("%.12f" % getidf('war'))
    print("%.12f" % getidf('military'))
    print("%.12f" % getidf('great'))
    print("--------------")
    print("%.12f" % getweight('02_washington_1793.txt','arrive'))
    print("%.12f" % getweight('07_madison_1813.txt','war'))
    print("%.12f" % getweight('12_jackson_1833.txt','union'))
    print("%.12f" % getweight('09_monroe_1821.txt','british'))
    print("%.12f" % getweight('05_jefferson_1805.txt','public'))
    print("--------------")
    print("(%s, %.12f)" % query("pleasing people"))
    print("(%s, %.12f)" % query("british war"))
    print("(%s, %.12f)" % query("false public"))
    print("(%s, %.12f)" % query("people institutions"))
    print("(%s, %.12f)" % query("violated willingly"))

main()