import express from "express"
import dotenv from "dotenv"

import fs from "fs"
import { PDFParse } from "pdf-parse"//text ko extract jarna
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"


import { QdrantVectorStore } from "@langchain/qdrant"
import { HumanMessage, SystemMessage } from "@langchain/core/messages"
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import{} from "@langchain/qdrant"

dotenv.config()
const app = express()
const port = 5000
app.use(express.json())

import { ChatOpenAI } from "@langchain/openai";

const llm = new ChatOpenAI({
  model: "qwen/qwen3-4b-2507",
  apiKey: "lm-studio", // koi bhi string
  configuration: {
    baseURL: "http://127.0.0.1:1234/v1",
  },
});


const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001", // 768 dimensions
  taskType: TaskType.RETRIEVAL_DOCUMENT,
  title: "Document title",
});

const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
  url: process.env.QDRANT_URL,
  collectionName: "langchainjs-testing",
});

 const upload= async()=>{
    const pdfpath="./knowledge.pdf"
    const buffer=fs.readFileSync(pdfpath)
   
    const pdfresult=new PDFParse({data:buffer})
     const result=await pdfresult.getText()
    const text=result.text
    const spilitter=new RecursiveCharacterTextSplitter({
        chunkSize:1000,
        chunkOverlap:90
    })
  const docs=await spilitter.createDocuments([text]) 
  await vectorStore.addDocuments(docs)
 }



app.post("/ai", async (req, res) => {
    const { input } = req.body

    const docs = await vectorStore.similaritySearch(input, 5)
    const context = docs.map((d) => d.pageContent).join("/n")//llm ko join arke bhe denge

    const response = await llm.invoke([
        new SystemMessage(`You are a RAG AI assistant.

STRICT RULES:
- Answer ONLY from context
- Do not use outside knowledge
- If answer not found say:
  "I don't know from uploaded PDF."

Context:
${context}`)
,
new HumanMessage(input)
    ])

console.log(response)


    return res.status(200).json({ai:response.content})
})




app.get("/", (req, res) => {
    return res.json({ message: "hello from level4" })
})


app.listen(port, () => {
    console.log("server started")
})