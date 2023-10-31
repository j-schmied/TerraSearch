import spacy
from spacy.tokens import DocBin
from sys import argv


def main():
    nlp = spacy.blank("de")
    
    if len(argv) != 2:
        print("[i] Usage: ./preprocess.py <txt file>")
        exit(0)
        
        
    print("[i] Start processing document...")    
    with open(argv[1], 'r') as training_data:
        db = DocBin()
        
        for text, annotations in training_data:
            doc = nlp(text)
            ents = []
            
            for start, end, label in annotations:
                span = doc.char_span(start, end, label=label)
                ents.append(span)
            
            doc.ents = ents
            db.add(doc)
    
    db.to_disk("./train.spacy")
    
    print("[i] Preprocessing done.\nYou may start training spacy now. Run\npython -m spacy train config.cfg --output ./output/output --paths.train ./train.spacy --gpu-id 0")
    
    exit(0)


if __name__ == '__main__':
    main()