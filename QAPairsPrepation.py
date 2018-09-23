import numpy as np
import re
import nltk
import json
import csv

def main():
    paragraphs = []
    questions = []
    with open('SQUAD.csv', 'wb') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["Paragraph","Question", "Answer", "Start position","End position"])
        with open('train-v1.1.json') as json_file:
            data = json.load(json_file)
            topic_list = data['data']
            counter=0
            for topic in topic_list:
                for paragraph in topic['paragraphs']:
                    #sentence = re.sub('[^A-Za-z0-9|\'|\s]+', '', paragraph['context']).lower()
                    #tokens = nltk.word_tokenize(sentence)
                    #paragraphs.append([vocab.encode(token) for token in tokens])
                    index = len(paragraphs) - 1
                    _paragraph=re.sub('[^A-Za-z0-9|\'|\s]+', '', paragraph['context'].encode('ascii', 'ignore')).lower().strip()
                    for question in paragraph['qas']:
                        #sentence = re.sub('[^A-Za-z0-9|\'|\s]+', '', question['question']).lower()
                        #start_position=question['answers'][0]['answer_start']
                        answer=re.sub('[^A-Za-z0-9|\'|\s]+', '', question['answers'][0]['text'].encode('ascii', 'ignore')).lower().strip()
                        if counter==75201:
                            print "heej"
                        if answer=="" or answer=="NaN":
                            print "ERROR A HEAD!"
                            continue
                        start_position = _paragraph.find(answer)
                        end_position=start_position+len(answer)
                        #print _paragraph[start_position:end_position] + " | "+ answer
                        assert _paragraph[start_position:end_position]==answer
                        quos=re.sub('[^A-Za-z0-9|\'|\s]+', '',question['question'].encode('ascii', 'ignore')).lower().strip()
                        counter=counter+1
                        writer.writerow([_paragraph, quos
                                            , answer, start_position, end_position])
        outcsv.close()

if __name__=="__main__":
    main()