
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import { GoogleGenAI, GenerateContentResponse } from "@google/genai";

/**
 * Creates a fallback prompt to use when the primary one is blocked.
 * @param decade The decade string (e.g., "1950s").
 * @returns The fallback prompt string.
 */
function getFallbackPrompt(decade: string): string {
    return `Reimagine the person in this photo as if they were in the ${decade}. Focus on the clothing, hairstyle, and the photographic style of that era. Ensure the result is a clear, photorealistic image.`;
}

/**
 * Extracts the decade (e.g., "1950s") from a prompt string.
 */
function extractDecade(prompt: string): string | null {
    const match = prompt.match(/(\d{4}s)/);
    return match ? match[1] : null;
}

/**
 * Processes the Gemini API response, extracting the image part.
 */
function processGeminiResponse(response: GenerateContentResponse): string {
    const imagePartFromResponse = response.candidates?.[0]?.content?.parts?.find(part => part.inlineData);

    if (imagePartFromResponse?.inlineData) {
        const { mimeType, data } = imagePartFromResponse.inlineData;
        return `data:${mimeType};base64,${data}`;
    }

    const textResponse = response.text;
    console.error("API did not return an image. Response text:", textResponse);
    throw new Error(textResponse || "The AI returned a text response instead of an image. This might be due to safety filters.");
}

/**
 * A wrapper for the Gemini API call with retry logic.
 */
async function callGeminiWithRetry(imagePart: any, textPart: any): Promise<GenerateContentResponse> {
    const maxRetries = 2;
    const initialDelay = 1500;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            // Create a fresh instance for every call to ensure correct API key usage
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
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
    if (!process.env.API_KEY) {
        throw new Error("API Key is missing. Please check your environment configuration.");
    }

    const match = imageDataUrl.match(/^data:(image\/\w+);base64,(.*)$/);
    if (!match) {
        throw new Error("Invalid image data format.");
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
        
        // If it seems like a block (no image returned but text present), try the fallback
        if (errorMessage.toLowerCase().includes("text response") || errorMessage.toLowerCase().includes("safety")) {
            const decade = extractDecade(prompt);
            if (decade) {
                console.log(`Retrying with fallback prompt for ${decade}...`);
                const fallbackTextPart = { text: getFallbackPrompt(decade) };
                const fallbackResponse = await callGeminiWithRetry(imagePart, fallbackTextPart);
                return processGeminiResponse(fallbackResponse);
            }
        }
        throw error;
    }
}
