import csv
import os
import sys
import codecs
import nltk, re, pprint
from glob import glob
from xml.dom.minidom import parse, parseString
from nltk import WordPunctTokenizer

print("entered xmlparser.py")


def get_file_names(dirname):
    files = []
    os.chdir(dirname)
    start_dir = os.getcwd()
    pattern = "*.xml"
    for dir,_,_ in os.walk(start_dir):
        files = files + glob(os.path.join(dir, pattern))

    non_empty_files = []
    for fname in files:
        with codecs.open(fname, "r", encoding="utf-8") as f:
            ct = f.read()
            if len(ct) > 0:
                non_empty_files.append(fname)

    print("obtained " + str(len(non_empty_files)) + " files from " + start_dir) 
    print("returning to initial directory...")
    os.chdir(start_dir)
    os.chdir("..")
    print("current directory: " + os.getcwd())
    return non_empty_files

def parse_french_data(fname, non_empty_files):
    print("entered french xml parser")
    for fname in non_empty_files:
    #for fname in glob("*.xml"):
        try:
            dom=parse(fname)
            for node in dom.getElementsByTagName('ID'):
                idx = node.toxml()
            for node in dom.getElementsByTagName('BLOC_TEXTUEL'):
                contenu = node.toxml()
            for node in dom.getElementsByTagName('FORMATION'):
                formation = node.toxml()
            for node in dom.getElementsByTagName('DATE_DEC'):
                date = node.toxml()
            for node in dom.getElementsByTagName('DATE_DEC_ATT'):
                date_att = node.toxml()
            for node in dom.getElementsByTagName('SOLUTION'):
                solution=node.toxml()
            for node in dom.getElementsByTagName('LIEN'):
                citation = node.toxml()

            if ("<CONTENU/>" in contenu) and ("<CONTENU>" not in contenu):
                contenu = ""
            else: 
                contenu=contenu.replace("<BLOC_TEXTUEL>","")
                contenu=contenu.replace("<CONTENU>", "")
                contenu=contenu.replace("</BLOC_TEXTUEL>","")
                contenu=contenu.replace("</CONTENU>", "") 
                contenu=contenu.replace("<br/>","")
                contenu=contenu.replace("<p>","")
                contenu=contenu.replace("</p>","")
                contenu=contenu.replace("\n"," ")
                contenu=contenu.replace("\t"," ")
                

        # if contenu == "":
        #     for node in dom.getElementsByTagName('SOMAIRE')

            content = WordPunctTokenizer().tokenize(contenu)
            articles = []
    				
            for i in range(len(content)):
                if (content[i].lower() == "article") or (content[i] == "Article"):
                    articles.append(content[i+1])

            articles = " ".join(articles)
            content = ' '.join(content) # joins the elements of the list content 
            
            idx = idx.replace("<ID>", "")
            idx = idx.replace("</ID>", "") 

            if "<FORMATION/>" == formation:
                formation =""
            else:         
                formation = formation.replace("<FORMATION>","")
                formation = formation.replace("</FORMATION>","")
                formation = formation.replace("\t"," ")
                formation = formation.replace("\n"," ")

            if "<DATE_DEC/>" == date:
                date =""
            else:
                date = date.replace("<DATE_DEC>","")
                date = date.replace("</DATE_DEC>","")
                date = date.replace("\t"," ")
                date = date.replace("\n"," ")

            if "<DATE_DEC_ATT/>" == date_att:
                date_att = ""
            else:
                date_att = date_att.replace("<DATE_DEC_ATT>","")
                date_att = date_att.replace("</DATE_DEC_ATT>","")
                date_att = date_att.replace("\t"," ")
                date_att = date_att.replace("\n"," ")

            if solution == "<SOLUTION/>": # if solution field is empty
                solution = "" # replace field with "" to simplify
            else: # else clean up
                solution = solution.replace("<SOLUTION>","")
                solution = solution.replace("</SOLUTION>","")
                solution = solution.replace(".","")
                solution = solution.replace("non lieu","non-lieu")
                solution = solution.replace("\t"," ")
                solution = solution.replace("\n"," ")

            if re.search(">.+<", citation) == None:
                citation = ""
            else:
                citation = (re.search(">.+<", citation)).group()
                citation = citation[1:len(citation)-1]
                citation = citation.replace("\t", " ")
                citation = citation.replace("\n", " ")   
            
            if "2999" in date_att:
                csv = idx + "\t"+formation+"\t"+solution+"\t"+date+"\t"+content+"\t"+articles+"\t"+citation+"\n" 
            else:
                csv = idx + "\t"+formation+"\t"+solution+"\t"+date_att+"\t"+content+"\t"+articles+"\t"+citation+"\n"
            count = count + 1
            g.write(csv)
        except Exception as e:
            print(e)
        
    g.close()
    print("count: " + str(count) + " len non_empty_files: " + str(len(non_empty_files)) + "len files:" + str(len(files)) + "done!") 
