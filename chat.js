import { OpenAI } from "openai";
import { queryVectorStore } from "./queryStore.js";

export async function answerUserQuestion(question) {

  const relevantDocs = await queryVectorStore(question);
  const contextText = relevantDocs.map(doc => doc.pageContent).join("\n\n");

  const openai = new OpenAI({
    apiKey: process.env.API_KEY,
    baseURL: "https://integrate.api.nvidia.com/v1"
  });

  const response = await openai.chat.completions.create({
    model: "nvidia/nemotron-3-ultra-550b-a55b",
    messages: [
      {
        role: "system",
        content: `You are a helpful assistant. Answer the user's question accurately using ONLY the following context. If the answer cannot be found in the context, say "I cannot find that in the provided document."\n\nContext:\n${contextText}`
      },
      { role: "user", content: question }
    ],
    temperature: 1,
    top_p: 0.95,
    max_tokens: 16384,
    reasoning_budget: 16384,
    chat_template_kwargs: { "enable_thinking": true },
    stream: true
  });

  for await (const chunk of response) {
    process.stdout.write(chunk.choices[0]?.delta?.content || '')
  }
}