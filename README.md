#  Semantic Search PDF

### Preview with ability to do semantic search

I use the MiniLM_V6 model and the SWIFT BERT_TOKENIZER code by Julien Chaumond to convert PDF text into tokens, fetch embedding, and also search embeddings all on device. I store all embeddings in memory for now and use cosine similarity to find the most similar embeddings. 
I plan to write an optimized vector store to do the similarity ranking faster as the pdf page gets bigger. Also use TF-IDF and exact word search to make search results better. 



