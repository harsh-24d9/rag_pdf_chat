import 'dotenv/config'
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"
import { QdrantVectorStore } from "@langchain/qdrant"
import { OpenAIEmbeddings } from "@langchain/openai";
import { answerUserQuestion } from "./chat.js"

async function createEmbeddingsFromFile(filePath, query) {
  if (!filePath || filePath == '') {
    console.log("file path should not be empty")
    return null;
  }

  if (!query || query === '') {
    console.log("query should not be empty")
    return null;
  }

  const loader = new PDFLoader(filePath);
  const docs = await loader.load();

  // console.log("pdf loaded successfully")

  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 600, chunkOverlap: 60 });
  const chunkedDocs = await splitter.splitDocuments(docs)

  // console.log("data chunked successfully")

  const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.API_KEY,
    model: "nvidia/nv-embed-v1",
    configuration: {
      baseURL: "https://integrate.api.nvidia.com/v1",
    },
  });

  // console.log("embeddings initialized successfully")

  const vectorStore = await QdrantVectorStore.fromDocuments(chunkedDocs, embeddings, {
    url: process.env.QDRANT_URL,
    collectionName: "docs",
  });
  console.log("Embeddings successfully pushed to Qdrant!");

  answerUserQuestion(query)
}

createEmbeddingsFromFile('', '');