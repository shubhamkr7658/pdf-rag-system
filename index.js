import express from "express";
import dotenv from "dotenv";
import OpenAI from "openai";
import cors from "cors";
import multer from "multer";

import fs from "fs";
import { PDFParse } from "pdf-parse";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

import { QdrantVectorStore } from "@langchain/qdrant";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";


// ==================================================
// CONFIG
// ==================================================

dotenv.config();

const app = express();
const port = 5000;

app.use(cors());
app.use(express.json());


// ==================================================
// MULTER
// PDF RAM MEIN RAHEGA
// DISK PAR SAVE NAHI HOGA
// ==================================================

const upload = multer({
    storage: multer.memoryStorage(),
    limits: {
        fileSize: 10 * 1024 * 1024, // 10 MB
    },
});


// ==================================================
// LLM - OPENROUTER
// ==================================================

const llm = new OpenAI({
    apiKey: process.env.llm_api,
    baseURL: process.env.base_url_llm,
});


// ==================================================
// EMBEDDINGS
// ==================================================

const embeddings = new GoogleGenerativeAIEmbeddings({
    model: "gemini-embedding-001",
    taskType: TaskType.RETRIEVAL_DOCUMENT,
    title: "Document",
});


// ==================================================
// QDRANT
// ==================================================

const COLLECTION_NAME = "langchainjs-testing";

const vectorStore =
    await QdrantVectorStore.fromExistingCollection(
        embeddings,
        {
            url: process.env.QDRANT_URL,
            collectionName: COLLECTION_NAME,
        }
    );


// ==================================================
// GET QDRANT POINT COUNT
// ==================================================

const getPointCount = async () => {

    const response = await fetch(
        `${process.env.QDRANT_URL}/collections/${COLLECTION_NAME}/points/count`,
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json",

                ...(process.env.QDRANT_API_KEY && {
                    "api-key": process.env.QDRANT_API_KEY,
                }),
            },

            body: JSON.stringify({
                exact: true,
            }),
        }
    );

    if (!response.ok) {
        throw new Error("Could not get Qdrant point count");
    }

    const data = await response.json();

    return data.result.count;
};


// ==================================================
// DELETE ALL VECTORS FROM QDRANT
// ==================================================

const deleteAllVectors = async () => {

    const response = await fetch(
        `${process.env.QDRANT_URL}/collections/${COLLECTION_NAME}/points/delete?wait=true`,
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json",

                ...(process.env.QDRANT_API_KEY && {
                    "api-key": process.env.QDRANT_API_KEY,
                }),
            },

            body: JSON.stringify({
                filter: {},
            }),
        }
    );

    if (!response.ok) {
        const errorText = await response.text();

        throw new Error(
            `Failed to delete Qdrant data: ${errorText}`
        );
    }

    console.log("All old vectors deleted");
};


// ==================================================
// PDF BUFFER → QDRANT
// ==================================================

const processPDF = async (buffer) => {

    // ----------------------------------------------
    // PDF → TEXT
    // ----------------------------------------------

    const pdf = new PDFParse({
        data: buffer,
    });

    const result = await pdf.getText();

    const text = result.text;

    if (!text || !text.trim()) {
        throw new Error("PDF does not contain readable text");
    }


    // ----------------------------------------------
    // TEXT → CHUNKS
    // ----------------------------------------------

    const splitter =
        new RecursiveCharacterTextSplitter({

            chunkSize: 1000,

            chunkOverlap: 90,

        });


    const docs =
        await splitter.createDocuments([text]);


    console.log(
        `Created ${docs.length} chunks`
    );


    // ----------------------------------------------
    // CHUNKS → EMBEDDINGS → QDRANT
    // ----------------------------------------------

    await vectorStore.addDocuments(docs);


    console.log(
        "New PDF embeddings added to Qdrant"
    );


    return docs.length;
};


// ==================================================
// DEFAULT PDF
// ==================================================

const loadDefaultPDF = async () => {

    const count = await getPointCount();

    console.log(
        `Current Qdrant points: ${count}`
    );


    // Agar database already populated hai
    // to default PDF dobara insert nahi karenge.

    if (count > 0) {

        console.log(
            "Qdrant already contains data."
        );

        return;
    }


    console.log(
        "Qdrant is empty. Loading default PDF..."
    );


    const buffer =
        fs.readFileSync("./knowledge.pdf");


    await processPDF(buffer);


    console.log(
        "Default PDF loaded successfully"
    );
};


// ==================================================
// UPLOAD PDF API
// ==================================================

app.post(
    "/upload",
    upload.single("pdf"),

    async (req, res) => {

        try {

            if (!req.file) {

                return res.status(400).json({
                    error: "Please upload a PDF",
                });

            }


            // PDF validation

            if (
                req.file.mimetype !==
                "application/pdf"
            ) {

                return res.status(400).json({
                    error: "Only PDF files are allowed",
                });

            }


            console.log(
                `Received PDF: ${req.file.originalname}`
            );


            // ------------------------------------------
            // 1. DELETE OLD DATA
            // ------------------------------------------

            await deleteAllVectors();


            // ------------------------------------------
            // 2. PROCESS NEW PDF
            // ------------------------------------------

            const chunkCount =
                await processPDF(
                    req.file.buffer
                );


            // req.file.buffer ab request ke baad
            // memory se release ho jayega.
            // Koi PDF disk par save nahi hua.


            return res.status(200).json({

                message:
                    "PDF uploaded successfully",

                filename:
                    req.file.originalname,

                chunks:
                    chunkCount,

            });

        }

        catch (error) {

            console.error(
                "Upload error:",
                error
            );


            return res.status(500).json({

                error:
                    "Failed to process PDF",

            });

        }

    }
);


// ==================================================
// ASK AI
// ==================================================

app.post("/ai", async (req, res) => {

    try {

        const { input } = req.body;


        if (!input) {

            return res.status(400).json({

                error:
                    "Input is required",

            });

        }


        // ------------------------------------------
        // 1. QUESTION → QDRANT
        // ------------------------------------------

        const docs =
            await vectorStore.similaritySearch(
                input,
                5
            );


        // ------------------------------------------
        // 2. CHUNKS → CONTEXT
        // ------------------------------------------

        const context =
            docs
                .map(
                    (doc) =>
                        doc.pageContent
                )
                .join("\n");


        // ------------------------------------------
        // 3. CONTEXT + QUESTION → LLM
        // ------------------------------------------

        const response =
            await llm.chat.completions.create({

                model:
                    "openrouter/free",

                messages: [

                    {
                        role: "system",

                        content: `

You are a Retrieval-Augmented Generation (RAG) assistant.

Use ONLY the context provided below.

Rules:

- Answer ONLY the user's question.
- Do NOT repeat or summarize the entire context.
- Do NOT list unrelated products.
- If multiple items satisfy the query, list only those items.
- Keep answers concise.
- Use bullet points for lists.
- If the answer is not found, reply:
"I don't know from the uploaded PDF."

Context:

${context}

`,
                    },

                    {
                        role: "user",

                        content: input,

                    },

                ],

            });


        // ------------------------------------------
        // 4. ANSWER
        // ------------------------------------------

        const answer =
            response
                .choices[0]
                .message
                .content;


        return res.status(200).json({

            ai: answer,

        });

    }

    catch (error) {

        console.error(
            "RAG Error:",
            error
        );


        return res.status(500).json({

            error:
                "Something went wrong",

        });

    }

});


// ==================================================
// HEALTH CHECK
// ==================================================

app.get("/", (req, res) => {

    res.json({

        message:
            "RAG server is running",

    });

});


// ==================================================
// START SERVER
// ==================================================

const startServer = async () => {

    try {

        // Default PDF only if Qdrant is empty

        await loadDefaultPDF();


        app.listen(
            port,
            () => {

                console.log(
                    `Server running on http://localhost:${port}`
                );

            }
        );

    }

    catch (error) {

        console.error(
            "Server startup failed:",
            error
        );

        process.exit(1);

    }

};


startServer();