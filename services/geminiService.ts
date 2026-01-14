
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import { GoogleGenAI, GenerateContentResponse } from "@google/genai";

/**
 * Creates a fallback prompt to use when the primary one is blocked.
 */
function getFallbackPrompt(decade: string): string {
    return `Reimagine the person in this photo as if they were in the ${decade}. Focus on clothing and hairstyle. Ensure it looks like an authentic vintage photograph.`;
}

function extractDecade(prompt: string): string | null {
    const match = prompt.match(/(\d{4}s)/);
    return match ? match[1] : null;
}

function processGeminiResponse(response: GenerateContentResponse): string {
    const imagePartFromResponse = response.candidates?.[0]?.content?.parts?.find(part => part.inlineData);

    if (imagePartFromResponse?.inlineData) {
        const { mimeType, data } = imagePartFromResponse.inlineData;
        return `data:${mimeType};base64,${data}`;
    }

    const textResponse = response.text;
    console.error("API did not return an image. Response text:", textResponse);
    throw new Error(textResponse || "The AI returned a text response instead of an image.");
}

/**
 * A wrapper for the Gemini API call with retry logic.
 */
async function callGeminiWithRetry(imagePart: any, textPart: any): Promise<GenerateContentResponse> {
    const maxRetries = 2;
    const initialDelay = 1500;
    
    // The environment variable MUST be named exactly API_KEY.
    const apiKey = process.env.API_KEY;
    
    if (!apiKey) {
        // Specifically guide Vercel users who might have used GOOGLE_API_KEY instead of API_KEY
        throw new Error("API_KEY is missing. Please rename your Vercel environment variable from 'GOOGLE_API_KEY' to 'API_KEY' and re-deploy.");
    }

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            // New instance per call as per guidelines
            const ai = new GoogleGenAI({ apiKey });
            return await ai.models.generateContent({
                model: 'gemini-2.5-flash-image',
                contents: { parts: [imagePart, textPart] },
            });
        } catch (error) {
            console.error(`Error on attempt ${attempt}:`, error);
            const errorMessage = error instanceof Error ? error.message : String(error);
            const isRetriable = errorMessage.includes('500') || errorMessage.includes('INTERNAL') || errorMessage.includes('429');

            if (isRetriable && attempt < maxRetries) {
                await new Promise(resolve => setTimeout(resolve, initialDelay * attempt));
                continue;
            }
            throw error;
        }
    }
    throw new Error("Maximum retries reached for Gemini API.");
}

/**
 * Generates a decade-styled image.
 */
export async function generateDecadeImage(imageDataUrl: string, prompt: string): Promise<string> {
    const match = imageDataUrl.match(/^data:(image\/\w+);base64,(.*)$/);
    if (!match) {
        throw new Error("Invalid image format. Please try re-uploading.");
    }
    const [, mimeType, base64Data] = match;

    const imagePart = {
        inlineData: { mimeType, data: base64Data },
    };

    try {
        const textPart = { text: prompt };
        const response = await callGeminiWithRetry(imagePart, textPart);
        return processGeminiResponse(response);
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        
        // Handle potential safety blocks or non-image responses
        if (errorMessage.toLowerCase().includes("safety") || errorMessage.toLowerCase().includes("text response")) {
            const decade = extractDecade(prompt);
            if (decade) {
                const fallbackTextPart = { text: getFallbackPrompt(decade) };
                const fallbackResponse = await callGeminiWithRetry(imagePart, fallbackTextPart);
                return processGeminiResponse(fallbackResponse);
            }
        }
        throw error;
    }
}
