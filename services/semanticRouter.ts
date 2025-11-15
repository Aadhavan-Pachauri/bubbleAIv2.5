
import { SupabaseClient } from '@supabase/supabase-js';
import { MemoryLayer } from '../types';

export type RouterAction = 'SEARCH' | 'DEEP_SEARCH' | 'THINK' | 'IMAGE' | 'CANVAS' | 'PROJECT' | 'STUDY' | 'SIMPLE';

export interface RoutingDecision {
    action: RouterAction;
    parameters: any;
    confidence: number;
    memoryLayers: MemoryLayer[];
    quotaOK: boolean;
    reasoning: string;
}

export class BubbleSemanticRouter {
    private supabase: SupabaseClient;

    constructor(supabaseClient: SupabaseClient) {
        this.supabase = supabaseClient;
    }

    async route(query: string, userId: string, apiKey: string, fileCount: number = 0): Promise<RoutingDecision> {
        // REVERTED: Complex semantic routing removed in favor of tag-based detection in the main agent.
        // We now default to SIMPLE to let the main Gemini 2.0 model decide when to use tools via tags.
        
        return {
            action: 'SIMPLE',
            parameters: {},
            confidence: 1,
            // Include ALL layers to ensure personal details (names) and preferences are always available
            memoryLayers: ['inner_personal', 'outer_personal', 'interests', 'preferences', 'custom', 'codebase', 'aesthetic', 'project'], 
            quotaOK: true,
            reasoning: "Defaulting to conversation. Tags will trigger specific actions."
        };
    }
}
