import 'dotenv/config'
import { QdrantVectorStore } from "@langchain/qdrant"
import { OpenAIEmbeddings } from "@langchain/openai";

export async function queryVectorStore(userQuery) {

  const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.API_KEY,
    model: "nvidia/nv-embed-v1",
    configuration: { baseURL: "https://integrate.api.nvidia.com/v1" },
  });

  const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
    url: process.env.QDRANT_URL,
    collectionName: "docs",
  });

  const results = await vectorStore.similaritySearch(userQuery, 4);
  return results;
}