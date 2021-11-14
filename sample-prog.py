import kenly

if __name__ == "__main__":
    kenly.load_embeddings()
    
    query_sentence = "dcp portable water line leaking infront of chillwater tank"
    df = kenly.find_topK_sentences(query_sentence, 10)

    print(df)