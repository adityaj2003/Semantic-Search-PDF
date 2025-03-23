# Semantic Search PDF
### Preview with Ability to Perform Semantic Search
I use the MiniLM_V6 model and the SWIFT BERT_TOKENIZER code by Julien Chaumond to convert PDF text into tokens, fetch embeddings, and perform search operations, all on-device. Currently, I store all embeddings in memory and use cosine similarity to find the most similar embeddings.

#### Features
* On-Device Processing: All tokenization, embedding fetching, and search operations are performed on the device, ensuring privacy and reducing the need for internet connectivity.
* Cosine Similarity: This method is used to find the most similar embeddings, providing efficient and accurate search results.
* Implemented storing embeddings in HNSW to speed up indexing time by 3x
* Use batch processing to generate embeddings in CoreML leading to much 2x faster indexing overall for a 400 page pdf


