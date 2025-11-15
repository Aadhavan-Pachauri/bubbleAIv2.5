

import React from 'react';
import { Message, Task } from '../../types';
import { motion, AnimatePresence } from 'framer-motion';
import { CodeBlock } from '../ui/CodeBlock';
import { CheckCircleIcon, LightBulbIcon, CodeBracketSquareIcon, EyeIcon, ShareIcon, SparklesIcon } from '@heroicons/react/24/solid';
import { CpuChipIcon, ExclamationTriangleIcon, ChevronDownIcon } from '@heroicons/react/24/outline';
import { ClarificationForm } from './ClarificationForm';
import { useAuth } from '../../contexts/AuthContext';
import { useState } from 'react';
import { MessageContent } from './MessageContent';
import { ImageModal } from '../modals/ImageModal';
import { MermaidDiagram } from './MermaidDiagram';

const ImageLoadingPlaceholder: React.FC = () => {
    return (
        <div className="relative aspect-square w-full max-w-md my-4 p-4 rounded-lg bg-black/20 border border-white/10 overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-tr from-bg-tertiary via-bg-secondary to-bg-tertiary animate-pulse"></div>
            <div className="relative z-10 flex flex-col items-center justify-center h-full text-center">
                <SparklesIcon className="w-10 h-10 text-primary-start/50 mb-3" />
                <p className="font-semibold text-white/80">Generating Image...</p>
                <p className="text-sm text-white/50">The AI is creating your visual, this may take a moment.</p>
            </div>
        </div>
    );
};

interface ChatMessageProps {
  message: Message;
  onExecutePlan: (messageId: string) => void;
  onClarificationSubmit: (messageId: string, answers: string[]) => void;
  isDimmed?: boolean;
  isCurrentResult?: boolean;
  searchQuery?: string;
  isAdmin?: boolean;
  isTyping?: boolean;
}

const TaskStatusIcon: React.FC<{ status: Task['status'] }> = ({ status }) => {
    if (status === 'in-progress') {
        return (
            <svg className="animate-spin h-5 w-5 text-primary-start flex-shrink-0" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
        );
    }
    if (status === 'pending') {
        return (
            <div className="w-5 h-5 flex-shrink-0 flex items-center justify-center">
                <div className="w-2 h-2 rounded-full bg-gray-500"></div>
            </div>
        );
    }
    return null;
}

const TaskRenderer: React.FC<{ task: Task }> = ({ task }) => {
    const [isExpanded, setIsExpanded] = useState(false);
    
    const taskTextStyle = task.status === 'in-progress' ? 'text-white' : 'text-gray-300';

    if (task.status !== 'complete') {
        return (
            <div className="flex items-center space-x-3 p-3">
                <TaskStatusIcon status={task.status} />
                <span className={taskTextStyle}>{task.text}</span>
            </div>
        )
    }

    const hasError = !task.code;

    return (
        <div className={`rounded-lg transition-colors ${hasError ? 'bg-error/10' : 'bg-success/5'}`}>
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="w-full flex justify-between items-center p-3 text-left"
            >
                <div className="flex items-center space-x-3">
                    {hasError 
                        ? <ExclamationTriangleIcon className="w-5 h-5 text-error flex-shrink-0" />
                        : <CheckCircleIcon className="w-5 h-5 text-success flex-shrink-0" />
                    }
                    <span className={`text-sm ${hasError ? 'text-error/90' : 'text-gray-400'} line-through`}>{task.text}</span>
                </div>
                <ChevronDownIcon className={`w-5 h-5 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
            </button>
            <AnimatePresence>
                {isExpanded && (
                    <motion.div
                        // FIX: framer-motion props wrapped in a spread object to bypass type errors.
                        {...{
                          initial: { height: 0, opacity: 0 },
                          animate: { height: 'auto', opacity: 1 },
                          exit: { height: 0, opacity: 0 },
                        }}
                        className="overflow-hidden"
                    >
                        <div className="px-3 pb-3">
                            <div className="p-4 rounded-md border border-white/10 bg-black/20 space-y-4">
                                {hasError ? (
                                    <div>
                                        <h5 className="font-semibold text-error mb-2">Error Details</h5>
                                        <p className="text-sm text-error/80 whitespace-pre-wrap">{task.explanation}</p>
                                    </div>
                                ) : (
                                    <>
                                        <div>
                                            <div className="flex items-center gap-2 mb-2">
                                                <LightBulbIcon className="w-5 h-5 text-yellow-400" />
                                                <h5 className="font-semibold text-white">Explanation</h5>
                                            </div>
                                            <p className="text-sm text-gray-300 whitespace-pre-wrap">{task.explanation}</p>
                                        </div>
                                        {task.code && (
                                           <div>
                                                <div className="flex items-center gap-2 mb-2">
                                                    <CodeBracketSquareIcon className="w-5 h-5 text-gray-400" />
                                                    <h5 className="font-semibold text-white">Generated Code</h5>
                                                </div>
                                                <CodeBlock code={task.code} language="lua" />
                                           </div>
                                        )}
                                    </>
                                )}
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}

const PlanExecutionRenderer: React.FC<{ plan: Message['plan'] }> = ({ plan }) => {
    if (!plan) return null;
    return (
        <div className="mx-4 mb-4 p-4 rounded-lg bg-black/20 border border-white/10">
             {plan.mermaidGraph && (
                <div className="mb-4">
                     <MermaidDiagram graphDefinition={plan.mermaidGraph} />
                </div>
             )}
            <div className="flex items-center mb-4">
                <CpuChipIcon className="w-6 h-6 text-primary-start mr-3" />
                <div>
                    <h4 className="font-semibold text-white">Building: {plan.title}</h4>
                    <p className="text-sm text-gray-400">The AI is working on the tasks below.</p>
                </div>
            </div>
            <div className="space-y-2">
                {plan.tasks.map((task, index) => (
                    <TaskRenderer key={index} task={task} />
                ))}
            </div>
        </div>
    );
};


const PlanUIRenderer: React.FC<{ message: Message, onExecutePlan: (messageId: string) => void, isTyping?: boolean, searchQuery?: string }> = ({ message, onExecutePlan, isTyping, searchQuery }) => {
    const { plan } = message;
    if (!plan) return null;

    const isPlanEmpty = (!plan.features || plan.features.length === 0 || (plan.features.length === 1 && plan.features[0].includes("insufficient"))) && !plan.mermaidGraph;
    if (isPlanEmpty) {
        return <MessageContent content={message.text} searchQuery={searchQuery || ''} sender={message.sender} isTyping={isTyping} />;
    }

    const hasStartedExecution = plan.tasks.some(t => t.status !== 'pending');

    if (hasStartedExecution) {
        return (
            <>
                <MessageContent content={message.text} searchQuery={searchQuery || ''} sender={message.sender} isTyping={isTyping} />
                <PlanExecutionRenderer plan={plan} />
            </>
        );
    }
    
    return (
        <div className="space-y-4">
            <MessageContent content={message.text} searchQuery={searchQuery || ''} sender={message.sender} isTyping={isTyping} />
            
            <div className="p-4 rounded-lg bg-black/20 border border-white/10">
                <h4 className="font-semibold text-white mb-2">I'll include the following features:</h4>
                <ul className="space-y-1.5">
                    {plan.features.map((feature, index) => (
                        <li key={index} className="flex items-start">
                            <CheckCircleIcon className="w-5 h-5 text-primary-start/80 mr-2 mt-0.5 flex-shrink-0" />
                            <span className="text-gray-300">{feature}</span>
                        </li>
                    ))}
                </ul>
            </div>
            
            <div className="p-4 rounded-lg bg-black/20 border border-white/10">
                 <h4 className="font-semibold text-white mb-2 flex items-center gap-2">
                    <ShareIcon className="w-5 h-5 text-primary-start/80"/>
                    Project Blueprint
                </h4>
                 <p className="text-sm text-gray-400 mb-2">Here's a visual breakdown of how the components will work together.</p>
                 <div className="p-2 rounded-lg bg-bg-secondary/70">
                    {plan.mermaidGraph ? (
                        <MermaidDiagram graphDefinition={plan.mermaidGraph} />
                    ) : (
                        <div className="p-6 text-center text-gray-400 border-2 border-dashed border-bg-tertiary rounded-lg">
                            <p className="font-semibold">Please Provide Project Details</p>
                        </div>
                    )}
                 </div>
             </div>
            
            <div className="pb-3">
                 <button
                    onClick={() => onExecutePlan(message.id)}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-semibold bg-primary-start text-white rounded-lg shadow-lg hover:bg-primary-start/80 transition-all duration-150 ease-in-out transform hover:scale-[1.02] active:scale-95"
                >
                    <SparklesIcon className="w-5 h-5"/>
                    <span>Start Building</span>
                </button>
            </div>
        </div>
    )
}

const ClarificationRenderer: React.FC<{ message: Message, onClarificationSubmit: (messageId: string, answers: string[]) => void, isTyping?: boolean, searchQuery?: string }> = ({ message, onClarificationSubmit, isTyping, searchQuery }) => {
    const { clarification } = message;
    if (!clarification) return null;

    const handleSubmit = (answers: string[]) => {
        onClarificationSubmit(message.id, answers);
    }

    if (clarification.answers) {
        return <MessageContent content={message.text} searchQuery={searchQuery || ''} sender={message.sender} isTyping={isTyping} />;
    }

    return (
        <div className="space-y-4">
            <MessageContent content={message.text} searchQuery={searchQuery || ''} sender={message.sender} isTyping={isTyping} />
            <ClarificationForm 
                questions={clarification.questions}
                onSubmit={handleSubmit}
            />
        </div>
    )
}

const ThinkerRenderer: React.FC<{ message: Message; isTyping?: boolean; searchQuery?: string }> = ({ message, isTyping, searchQuery }) => {
    const [activeTab, setActiveTab] = useState<'final' | 'standing' | 'opposing'>('final');
    
    if (!message.standing_response || !message.opposing_response) {
        return <MessageContent content={message.text} searchQuery={searchQuery || ''} sender={message.sender} isTyping={isTyping} />;
    }

    const tabs = [
        { id: 'final', label: 'Final Plan' },
        { id: 'standing', label: 'Standing' },
        { id: 'opposing', label: 'Opposing' }
    ];

    const renderContent = () => {
        switch(activeTab) {
            case 'standing':
                return (
                    <div>
                        <h5 className="font-semibold text-white mb-1">Thought Process</h5>
                        <p className="text-sm text-gray-400 italic mb-3">"{message.standing_response?.thought}"</p>
                        <h5 className="font-semibold text-white mb-2">Proposed Plan</h5>
                        <MessageContent content={message.standing_response?.response ?? ''} searchQuery={searchQuery || ''} sender={message.sender} isTyping={false} />
                    </div>
                );
            case 'opposing':
                return (
                    <div>
                        <h5 className="font-semibold text-white mb-1">Thought Process</h5>
                        <p className="text-sm text-gray-400 italic mb-3">"{message.opposing_response?.thought}"</p>
                        <h5 className="font-semibold text-white mb-2">Critique & Alternatives</h5>
                        <MessageContent content={message.opposing_response?.response ?? ''} searchQuery={searchQuery || ''} sender={message.sender} isTyping={false} />
                    </div>
                );
            case 'final':
            default:
                 return <MessageContent content={message.text} searchQuery={searchQuery || ''} sender={message.sender} isTyping={isTyping} />;
        }
    }

    return (
        <div className="py-2">
            <div className="flex border-b border-white/10">
                {tabs.map(tab => (
                    <button 
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id as any)}
                        className={`-mb-px px-4 py-2 text-sm font-medium transition-colors border-b-2 ${activeTab === tab.id ? 'text-primary-start border-primary-start' : 'text-gray-400 hover:text-white border-transparent'}`}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>
            <div className="pt-4">
                 {renderContent()}
            </div>
        </div>
    )
}


export const ChatMessage: React.FC<ChatMessageProps> = ({ 
    message, 
    onExecutePlan, 
    onClarificationSubmit,
    isDimmed = false,
    isCurrentResult = false,
    searchQuery = '',
    isAdmin = false,
    isTyping = false,
}) => {
  const isUser = message.sender === 'user';
  const [showRaw, setShowRaw] = useState(false);
  const [isImageModalOpen, setIsImageModalOpen] = useState(false);

  const variants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  };

  // User Message Bubble
  if (isUser) {
    let images: string[] = [];
    if (message.image_base64) {
        try {
            // Try parsing as a JSON array (for multiple images)
            const parsed = JSON.parse(message.image_base64);
            images = Array.isArray(parsed) ? parsed : [message.image_base64];
        } catch {
            // If parsing fails, it's a single base64 string
            images = [message.image_base64];
        }
    }

    return (
        <motion.div
          // FIX: framer-motion props wrapped in a spread object to bypass type errors.
          {...{
            variants: variants,
            initial: "hidden",
            animate: "visible",
            transition: { duration: 0.3 },
          }}
          className={`flex justify-end mb-3 ${isDimmed ? 'opacity-30' : 'opacity-100'}`}
        >
            <div className={`bg-zinc-800 text-zinc-100 rounded-2xl px-4 py-3 max-w-[70%] break-words shadow-md ${isCurrentResult ? 'ring-2 ring-offset-2 ring-offset-bg-primary ring-yellow-400' : ''}`}>
                {images.length > 0 && (
                    <div className={`grid gap-2 mb-2 ${images.length > 1 ? 'grid-cols-2' : 'grid-cols-1'}`}>
                        {images.map((img, index) => (
                            <img
                                key={index}
                                src={`data:image/jpeg;base64,${img}`}
                                alt={`User upload ${index + 1}`}
                                className="rounded-lg w-full h-auto object-cover max-w-xs"
                            />
                        ))}
                    </div>
                )}
                <MessageContent content={message.text} searchQuery={searchQuery} sender={message.sender} />
            </div>
        </motion.div>
    );
  }
  
  // AI Full-Width Response
  return (
    <motion.div
      // FIX: framer-motion props wrapped in a spread object to bypass type errors.
      {...{
        variants: variants,
        initial: "hidden",
        animate: "visible",
        transition: { duration: 0.3 },
      }}
      className={`flex items-start gap-4 transition-opacity duration-300 ${isDimmed ? 'opacity-30' : 'opacity-100'}`}
    >
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-bg-secondary flex items-center justify-center border border-border-color">
            <span className="text-lg">🫧</span>
        </div>
        <div className={`flex-1 min-w-0 ${isCurrentResult ? 'rounded-lg ring-2 ring-offset-2 ring-offset-bg-primary ring-yellow-400' : ''}`}>
            <div className="w-full prose">
                {showRaw ? (
                    <pre className="p-4 text-xs bg-black/30 rounded-lg overflow-x-auto">
                        {JSON.stringify(message, null, 2)}
                    </pre>
                ) : message.imageStatus === 'generating' ? (
                    <ImageLoadingPlaceholder />
                ) : (
                    <>
                        {message.standing_response ? (
                            <ThinkerRenderer message={message} searchQuery={searchQuery} isTyping={isTyping} />
                        ) : message.plan ? (
                            <PlanUIRenderer message={message} onExecutePlan={onExecutePlan} searchQuery={searchQuery} isTyping={isTyping} />
                        ) : message.clarification ? (
                            <ClarificationRenderer message={message} onClarificationSubmit={onClarificationSubmit} searchQuery={searchQuery} isTyping={isTyping}/>
                        ) : (
                            <MessageContent content={message.text} searchQuery={searchQuery} sender={message.sender} isTyping={isTyping} />
                        )}
            
                        {message.image_base64 && (
                            <>
                                <div className="mt-4 not-prose">
                                    <button 
                                        onClick={() => setIsImageModalOpen(true)} 
                                        className="block w-full group"
                                        aria-label="Enlarge image"
                                    >
                                        <img
                                            src={`data:image/png;base64,${message.image_base64}`}
                                            alt="Generated content"
                                            className="rounded-lg max-w-md mx-auto h-auto shadow-lg transition-transform duration-200 group-hover:scale-[1.02] cursor-pointer"
                                        />
                                    </button>
                                </div>
                        
                                <AnimatePresence>
                                    {isImageModalOpen && (
                                        <ImageModal 
                                            src={`data:image/png;base64,${message.image_base64}`}
                                            onClose={() => setIsImageModalOpen(false)}
                                        />
                                    )}
                                </AnimatePresence>
                            </>
                        )}
            
                        {/* Standalone code blocks for backward compatibility */}
                        {message.code && !message.text.includes('```') && (
                          <div className="not-prose">
                            <CodeBlock code={message.code} language={message.language || 'lua'} />
                          </div>
                        )}
                    </>
                )}
            </div>

            {message.groundingMetadata && Array.isArray(message.groundingMetadata) && message.groundingMetadata.length > 0 && (
                <div className="mt-4 pt-3 border-t border-border-color not-prose">
                    <h5 className="text-xs font-semibold text-text-secondary mb-2">SOURCES</h5>
                    <div className="space-y-1.5">
                        {message.groundingMetadata.map((chunk: any, index: number) => (
                            chunk.web && (
                            <a 
                                key={index}
                                href={chunk.web.uri}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="block text-sm text-blue-400 hover:underline truncate"
                                title={chunk.web.title || chunk.web.uri}
                            >
                                {index + 1}. {chunk.web.title || chunk.web.uri}
                            </a>
                            )
                        ))}
                    </div>
                </div>
            )}

          {isAdmin && (
            <button
                onClick={() => setShowRaw(!showRaw)}
                className="mt-2 flex items-center gap-1.5 px-2 py-1 text-xs text-gray-500 hover:text-white bg-white/5 rounded-md transition-colors"
            >
                <EyeIcon className="w-3 h-3" />
                {showRaw ? "Show Formatted" : "Raw Output"}
            </button>
          )}
        </div>
    </motion.div>
  );
};