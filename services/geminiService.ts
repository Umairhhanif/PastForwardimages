
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import { GoogleGenAI, GenerateContentResponse } from "@google/genai";

/**
 * Creates a fallback prompt to use when the primary one is blocked.
 */
function getFallbackPrompt(decade: string): string {
    return `Reimagine the person in this photo as if they were in the ${decade}. Focus on era-appropriate clothing and hairstyle. Ensure it looks like an authentic vintage photograph from that time.`;
}

function extractDecade(prompt: string): string | null {
    const match = prompt.match(/(\d{4}s)/);
    return match ? match[1] : null;
}

/**
 * Extracts the image data from the Gemini response candidates.
 */
function processGeminiResponse(response: GenerateContentResponse): string {
    const candidates = response.candidates;
    if (!candidates || candidates.length === 0) {
        throw new Error("The AI returned an empty response. This usually happens due to safety filters.");
    }

    // Guidelines: iterate through all parts to find the image part; do not assume the first part is an image part.
    const imagePartFromResponse = candidates[0].content?.parts?.find(part => part.inlineData);

    if (imagePartFromResponse?.inlineData) {
        const { mimeType, data } = imagePartFromResponse.inlineData;
        return `data:${mimeType};base64,${data}`;
    }

    // Access the text property directly (not as a method).
    const textResponse = response.text || "No text returned";
    console.error("API did not return an image. Response content:", textResponse);
    throw new Error(textResponse.includes("safety") ? "Safety filter blocked this image." : "AI returned text instead of an image.");
}

/**
 * A wrapper for the Gemini API call with retry logic and fresh API client initialization.
 */
async function callGeminiWithRetry(imagePart: any, textPart: any): Promise<GenerateContentResponse> {
    const maxRetries = 2;
    const initialDelay = 1500;
    
    // Always use process.env.API_KEY directly as specified in the guidelines.
    const apiKey = process.env.API_KEY;
    
    if (!apiKey) {
        throw new Error("API_KEY is not defined. Please ensure an API key is selected via the activation dialog.");
    }

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            // Create a new GoogleGenAI instance right before making an API call to ensure it uses the most up-to-date key.
            const ai = new GoogleGenAI({ apiKey });
            
            // Use ai.models.generateContent to query GenAI with both the model name and prompt.
            return await ai.models.generateContent({
                model: 'gemini-3-pro-image-preview', // Requires API key selection in UI
                contents: { parts: [imagePart, textPart] },
                config: {
                    imageConfig: {
                        aspectRatio: "1:1",
                        imageSize: "1K"
                    }
                }
            });
        } catch (error) {
            console.error(`Error on attempt ${attempt}:`, error);
            const errorMessage = error instanceof Error ? error.message : String(error);
            
            // Critical errors that should prompt a key re-selection or immediate stop.
            if (errorMessage.includes('429') || errorMessage.includes('QUOTA') || errorMessage.includes('BILLING') || errorMessage.includes('Requested entity was not found')) {
                throw new Error(errorMessage);
            }

            const isRetriable = errorMessage.includes('500') || errorMessage.includes('INTERNAL');
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
 * Generates a decade-styled image using a provided image and prompt.
 */
export async function generateDecadeImage(imageDataUrl: string, prompt: string): Promise<string> {
    const match = imageDataUrl.match(/^data:(image\/\w+);base64,(.*)$/);
    if (!match) {
        throw new Error("Invalid image format. Try uploading a standard photo.");
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
        
        // Handle blocked content by attempting a fallback with a simpler, less likely to be blocked prompt.
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
