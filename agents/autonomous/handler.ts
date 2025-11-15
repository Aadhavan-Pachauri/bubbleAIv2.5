
import { GoogleGenAI } from "@google/genai";
import { AgentInput, AgentExecutionResult } from '../types';
import { getUserFriendlyError } from '../errorUtils';
import { generateImage } from '../../services/geminiService';
import { incrementThinkingCount } from '../../services/databaseService';
import { researchService } from "../../services/researchService";
import { BubbleSemanticRouter, RouterAction } from "../../services/semanticRouter";
import { Memory5Layer } from "../../services/memoryService";
import { autonomousInstruction } from './instructions';

const formatTimestamp = () => {
    return new Date().toLocaleString(undefined, { 
        weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', 
        hour: '2-digit', minute: '2-digit', second: '2-digit', timeZoneName: 'short' 
    });
};

export const runAutonomousAgent = async (input: AgentInput): Promise<AgentExecutionResult> => {
    const { prompt, files, apiKey, project, chat, history, supabase, user, profile, onStreamChunk } = input;
    
    try {
        const ai = new GoogleGenAI({ apiKey });
        // Router is now just a dummy/pass-through to keep architecture consistent
        const router = new BubbleSemanticRouter(supabase);
        const memory = new Memory5Layer(supabase, user.id);

        // 1. Default Route (Always SIMPLE now)
        const fileCount = files ? files.length : 0;
        let routing = await router.route(prompt, user.id, apiKey, fileCount);
        
        // 2. Gather Context
        // Explicitly fetch all relevant layers to ensure name/preferences are present
        const memoryContext = await memory.getContext(['inner_personal', 'outer_personal', 'interests', 'preferences', 'custom', 'codebase', 'aesthetic', 'project']);
        const dateTimeContext = `[CURRENT DATE & TIME]\n${formatTimestamp()}\n`;
        
        let finalResponseText = '';
        let metadataPayload: any = {};
        
        // We use a loop to handle potential re-routing via tags (e.g. SIMPLE -> SEARCH)
        let currentAction: RouterAction = routing.action;
        let currentPrompt = prompt;
        let loopCount = 0;
        const MAX_LOOPS = 2; // Prevent infinite loops

        while (loopCount < MAX_LOOPS) {
            loopCount++;
            console.log(`[Autonomous Loop ${loopCount}] Action: ${currentAction}`);

            switch (currentAction) {
                case 'SEARCH': {
                    onStreamChunk?.("\n\n🔍 Searching the web...");
                    const searchSystemPrompt = `${autonomousInstruction}\n\n${dateTimeContext}\n\nProvide a helpful, friendly answer to the user's query using Google Search. Maintain your persona (Bubble). Cite sources.`;
                    
                    const searchResponse = await ai.models.generateContentStream({
                        model: 'gemini-2.0-flash-exp',
                        contents: `User Query: ${currentPrompt}`,
                        config: {
                            systemInstruction: searchSystemPrompt,
                            tools: [{ googleSearch: {} }],
                            responseModalities: ['TEXT'],
                        }
                    });
                    
                    let searchResultText = "";
                    for await (const chunk of searchResponse) {
                        if (chunk.text) {
                            searchResultText += chunk.text;
                            finalResponseText += chunk.text;
                            onStreamChunk?.(chunk.text);
                        }
                        
                        // Capture grounding metadata for citations
                        // Access candidates safely as per SDK types
                        const candidate = chunk.candidates?.[0];
                        if (candidate && candidate.groundingMetadata) {
                             const groundingMetadata = candidate.groundingMetadata;
                             if (groundingMetadata.groundingChunks) {
                                if (!metadataPayload.groundingMetadata) {
                                    metadataPayload.groundingMetadata = [];
                                }
                                metadataPayload.groundingMetadata.push(...groundingMetadata.groundingChunks);
                             }
                        }
                    }
                    
                    return { messages: [{ project_id: project.id, chat_id: chat.id, sender: 'ai', text: finalResponseText, ...metadataPayload }] };
                }

                case 'DEEP_SEARCH': {
                    onStreamChunk?.("\n\n🔬 Deep Researching...");
                    const result = await researchService.deepResearch(currentPrompt, (msg) => {
                         onStreamChunk?.(`\n*${msg}*`);
                    });
                    const researchText = result.answer + `\n\n**Sources:**\n${result.sources.join('\n')}`;
                    finalResponseText += "\n\n" + researchText;
                    onStreamChunk?.(researchText);
                    return { messages: [{ project_id: project.id, chat_id: chat.id, sender: 'ai', text: finalResponseText, ...metadataPayload }] };
                }

                case 'THINK': {
                    onStreamChunk?.("\n\n🧠 Thinking...");
                    await incrementThinkingCount(supabase, user.id);
                    
                    const geminiHistory = history.map(msg => ({
                        role: msg.sender === 'user' ? 'user' : 'model' as 'user' | 'model',
                        parts: [{ text: msg.text }],
                    })).filter(msg => msg.parts[0].text.trim() !== '');
                    
                    // Inject the system instruction and memory into the prompt for the thinking model
                    // ensuring it stays in character.
                    const contextBlock = `
${autonomousInstruction}

${dateTimeContext}

[MEMORY]
${JSON.stringify(memoryContext)}

[TASK]
${currentPrompt}
`;
                    // We replace the last user message with the contextualized one for the thinking model
                    const contents = [...geminiHistory, { role: 'user', parts: [{ text: contextBlock }] }];

                    const response = await ai.models.generateContentStream({
                        model: 'gemini-2.0-flash-thinking-exp-1219',
                        contents,
                        // We pass system instruction in the prompt above because thinking models 
                        // handle it better as part of the user/system context in the prompt body for now.
                    });

                    for await (const chunk of response) {
                        if (chunk.text) {
                            finalResponseText += chunk.text;
                            onStreamChunk?.(chunk.text);
                        }
                    }
                    return { messages: [{ project_id: project.id, chat_id: chat.id, sender: 'ai', text: finalResponseText, ...metadataPayload }] };
                }

                case 'IMAGE': {
                    // Emit event to show UI loader immediately
                    onStreamChunk?.(JSON.stringify({ type: 'image_generation_start', text: finalResponseText }));

                    // The prompt might have been updated by tag extraction
                    const imagePrompt = routing.parameters?.prompt || currentPrompt;
                    
                    // If we switched from SIMPLE, finalResponseText already contains the intro (e.g., "Sure, here is a cat")
                    // We don't need to add more text, just attach the image.

                    // Generate image
                    try {
                        const { imageBase64 } = await generateImage(imagePrompt, apiKey, profile?.preferred_image_model);
                        return { 
                            messages: [{ 
                                project_id: project.id, 
                                chat_id: chat.id, 
                                sender: 'ai', 
                                text: finalResponseText, // Keep whatever text was generated before
                                image_base64: imageBase64, 
                                imageStatus: 'complete',
                                ...metadataPayload 
                            }] 
                        };
                    } catch (e) {
                        const errorMsg = `\n\n(Image generation failed: ${e instanceof Error ? e.message : 'Unknown error'})`;
                        finalResponseText += errorMsg;
                        onStreamChunk?.(errorMsg);
                        return { messages: [{ project_id: project.id, chat_id: chat.id, sender: 'ai', text: finalResponseText, ...metadataPayload }] };
                    }
                }

                case 'CANVAS': {
                    onStreamChunk?.("\n\n📝 Opening Canvas...");
                    const canvasPrompt = `Create a single-file solution for: ${currentPrompt}. 
                    If it's a web page, provide full HTML/CSS/JS in one file. 
                    If it's a script, provide the full code. 
                    Do not use markdown backticks for the main code block in the final output if possible, or ensure it's clean.`;
                    
                    const response = await ai.models.generateContentStream({
                        model: 'gemini-2.5-flash-exp',
                        contents: canvasPrompt,
                        config: { systemInstruction: "You are an expert coder. Generate concise, production-ready code." }
                    });

                    for await (const chunk of response) {
                        if (chunk.text) {
                            finalResponseText += chunk.text;
                            onStreamChunk?.(chunk.text);
                        }
                    }
                    return { messages: [{ project_id: project.id, chat_id: chat.id, sender: 'ai', text: finalResponseText, ...metadataPayload }] };
                }

                case 'PROJECT': {
                    onStreamChunk?.("\n\n📦 Building project structure...");
                     const response = await ai.models.generateContent({
                        model: 'gemini-2.5-flash-exp',
                        contents: `Build a complete file structure for a project: ${currentPrompt}. Return a JSON object with filenames and brief content descriptions.`,
                        config: { responseMimeType: 'application/json' }
                    });
                    
                    const projectMsg = `\nI've designed the project structure based on your request.\n\n${response.text}\n\n(Switch to Co-Creator mode to fully hydrate and edit these files.)`;
                    finalResponseText += projectMsg;
                    onStreamChunk?.(projectMsg);
                    return { messages: [{ project_id: project.id, chat_id: chat.id, sender: 'ai', text: finalResponseText, ...metadataPayload }] };
                }
                
                case 'STUDY': {
                    onStreamChunk?.("\n\n📚 Creating study plan...");
                    const response = await ai.models.generateContentStream({
                        model: 'gemini-2.0-flash-exp',
                        contents: `Create a structured study plan for: ${currentPrompt}. Include learning objectives and key concepts.`,
                    });
                    for await (const chunk of response) {
                        if (chunk.text) {
                            finalResponseText += chunk.text;
                            onStreamChunk?.(chunk.text);
                        }
                    }
                    return { messages: [{ project_id: project.id, chat_id: chat.id, sender: 'ai', text: finalResponseText, ...metadataPayload }] };
                }

                case 'SIMPLE':
                default: {
                    // Standard conversational flow
                    const systemPrompt = `${autonomousInstruction}
                    
                    [MEMORY]
                    ${JSON.stringify(memoryContext)}
                    
                    ${dateTimeContext}
                    `;
                    
                    const geminiHistory = history.map(msg => ({
                        role: msg.sender === 'user' ? 'user' : 'model' as 'user' | 'model',
                        parts: [{ text: msg.text }],
                    })).filter(msg => msg.parts[0].text.trim() !== '');

                    // Handle files if present
                    const userParts: any[] = [{ text: currentPrompt }];
                    if (files && files.length > 0) {
                        for (const file of files) {
                            const base64EncodedData = await new Promise<string>((resolve, reject) => {
                                const reader = new FileReader();
                                reader.onload = () => resolve((reader.result as string).split(',')[1]);
                                reader.onerror = reject;
                                reader.readAsDataURL(file);
                            });
                            userParts.unshift({ inlineData: { data: base64EncodedData, mimeType: file.type } });
                        }
                    }
                    const contents = [...geminiHistory, { role: 'user', parts: userParts }];

                    const response = await ai.models.generateContentStream({
                        model: 'gemini-2.0-flash-exp',
                        contents,
                        config: { systemInstruction: systemPrompt }
                    });

                    let generatedThisLoop = "";

                    for await (const chunk of response) {
                        if (chunk.text) {
                            generatedThisLoop += chunk.text;
                            finalResponseText += chunk.text;
                            onStreamChunk?.(chunk.text);
                            
                            // Check for grounding metadata in the stream (citations)
                            const candidate = chunk.candidates?.[0];
                            if (candidate && candidate.groundingMetadata) {
                                 const groundingMetadata = candidate.groundingMetadata;
                                 if (groundingMetadata.groundingChunks) {
                                     if (!metadataPayload.groundingMetadata) metadataPayload.groundingMetadata = [];
                                     metadataPayload.groundingMetadata.push(...groundingMetadata.groundingChunks);
                                 }
                            }
                        }
                    }
                    
                    // --- TAG DETECTION LOGIC ---
                    // We check the text generated in this loop for tags.
                    const searchMatch = generatedThisLoop.match(/<SEARCH>(.*?)<\/SEARCH>/);
                    const deepMatch = generatedThisLoop.match(/<DEEP>(.*?)<\/DEEP>/) || generatedThisLoop.match(/<SEARCH>deep\s+(.*?)<\/SEARCH>/i);
                    const thinkMatch = generatedThisLoop.match(/<THINK>(.*?)<\/THINK>/) || generatedThisLoop.match(/<THINK>/);
                    const imageMatch = generatedThisLoop.match(/<IMAGE>(.*?)<\/IMAGE>/);
                    const projectMatch = generatedThisLoop.match(/<PROJECT>(.*?)<\/PROJECT>/);
                    const canvasMatch = generatedThisLoop.match(/<CANVAS>(.*?)<\/CANVAS>/);
                    const studyMatch = generatedThisLoop.match(/<STUDY>(.*?)<\/STUDY>/);

                    if (deepMatch) {
                         currentAction = 'DEEP_SEARCH';
                         currentPrompt = deepMatch[1];
                         finalResponseText = finalResponseText.replace(deepMatch[0], '');
                         continue;
                    }
                    if (searchMatch) {
                         currentAction = 'SEARCH';
                         currentPrompt = searchMatch[1];
                         finalResponseText = finalResponseText.replace(searchMatch[0], '');
                         continue;
                    }
                    if (thinkMatch) {
                         currentAction = 'THINK';
                         // If the tag has content, use it. Otherwise, use the original prompt.
                         currentPrompt = thinkMatch[1] ? thinkMatch[1].trim() : prompt; 
                         finalResponseText = finalResponseText.replace(thinkMatch[0], '');
                         continue; 
                    }
                    if (imageMatch) {
                         currentAction = 'IMAGE';
                         currentPrompt = imageMatch[1]; 
                         routing.parameters = { prompt: imageMatch[1] };
                         finalResponseText = finalResponseText.replace(imageMatch[0], '');
                         continue;
                    }
                    if (projectMatch) {
                        currentAction = 'PROJECT';
                        currentPrompt = projectMatch[1];
                        finalResponseText = finalResponseText.replace(projectMatch[0], '');
                        continue;
                    }
                    if (canvasMatch) {
                        currentAction = 'CANVAS';
                        currentPrompt = canvasMatch[1];
                        finalResponseText = finalResponseText.replace(canvasMatch[0], '');
                        continue;
                    }
                    if (studyMatch) {
                        currentAction = 'STUDY';
                        currentPrompt = studyMatch[1];
                        finalResponseText = finalResponseText.replace(studyMatch[0], '');
                        continue;
                    }
                    
                    // Check if memory needs saving (simple heuristic)
                    if (generatedThisLoop.length > 50) {
                         metadataPayload.memoryToCreate = [{ layer: 'outer_personal', key: 'last_topic', value: prompt }];
                    }
                    
                    return { messages: [{ project_id: project.id, chat_id: chat.id, sender: 'ai', text: finalResponseText, ...metadataPayload }] };
                }
            }
        }
        
        return { messages: [{ project_id: project.id, chat_id: chat.id, sender: 'ai', text: finalResponseText, ...metadataPayload }] };

    } catch (error) {
        console.error("Error in runAutonomousAgent:", error);
        const errorMessage = getUserFriendlyError(error);
        return { messages: [{ project_id: project.id, chat_id: chat.id, sender: 'ai', text: `An error occurred: ${errorMessage}` }] };
    }
};
