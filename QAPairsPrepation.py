import json
import csv
import corenlp
import os
import copy
import pip
import requests

os.environ["CORENLP_HOME"]="./CoreNLP/stanford-corenlp-full-2018-10-05/"
Data_DIR="./Data"


def downloadSQuADDataset():
    if not os.path.isdir(Data_DIR):
        os.mkdir(Data_DIR)
    print "Downloading SQuAD dataset..."
    response = requests.get('https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json')
    if response.status_code>=200 and response.status_code<300:
        file = open(Data_DIR+'/'+'train-v2.0.json','w')
        file.write(response.content)
        file.close()


def extractSentences():
    with corenlp.CoreNLPClient(timeout=100000) as client:
        with open(Data_DIR+"/"+'Sentences.csv', 'wb') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(["Sentence", "Processed Sentence ", "Answer", "question"])
            with open(Data_DIR+"/"+'train-v2.0.json') as json_file:
                data = json.load(json_file)
                topic_list = data['data']
                counter = 0
                dataset=[]
                for topic in topic_list:
                    for paragraph in topic['paragraphs']:
                        _paragraph=paragraph['context']
                        ann = client.annotate(_paragraph,annotators="tokenize,ssplit,pos,lemma,ner,coref".split())
                        sentences=ann.sentence
                        sentences_size=len(sentences)
                        for question in paragraph['qas']:
                            answer_position=question['answers'][0]['answer_start']
                            answer_text=question['answers'][0]['text']
                            len_accumelator=0
                            cluster_map={}
                            for sent in sentences:
                                modifiedSent=copy.deepcopy(sent)
                                for i in xrange(len(sent.token)):
                                    token=sent.token[i]
                                    if token.corefClusterID!= 0:
                                        if not cluster_map.has_key(token.corefClusterID):
                                            cluster_map[token.corefClusterID]=[token]
                                        else:
                                            if token.pos=="PRP" or token.pos=="PRP$":
                                                for coref in cluster_map[token.corefClusterID]:
                                                    if coref.ner!="O":
                                                        modifiedSent.token[i].word=coref.word if token.pos=="PRP" else coref.word+"'s"
                                                        break
                                            cluster_map[token.corefClusterID].append(token)

                                sentence_string = corenlp.to_text(sent).encode('ascii', 'ignore')
                                if len_accumelator + len(sentence_string) > answer_position:
                                    #dataset = (sentence_string,corenlp.to_text(modifiedSent),question['answers'][0]['text'], question['question'])
                                    writer.writerow([sentence_string,corenlp.to_text(modifiedSent).encode('ascii', 'ignore'),question['answers'][0]['text'].encode('ascii', 'ignore'), question['question'].encode('ascii', 'ignore')])

                                len_accumelator+=len(sentence_string)+1

            outcsv.close()
if __name__ == "__main__":
    downloadSQuADDataset()
    extractSentences()
