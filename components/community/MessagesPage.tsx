import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ChatBubbleLeftRightIcon, PaperAirplaneIcon, MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import { mockProfiles } from '../../data/users';
import { Profile } from '../../types';

const ConversationItem: React.FC<{ profile: Profile; selected: boolean; onSelect: () => void; }> = ({ profile, selected, onSelect }) => (
    <button
        onClick={onSelect}
        className={`w-full flex items-center gap-3 p-2 rounded-lg text-left transition-colors ${selected ? 'bg-white/10' : 'hover:bg-white/5'}`}
    >
        <img src={profile.avatar_url} alt={profile.roblox_username} className="w-10 h-10 rounded-full" />
        <div className="flex-1 min-w-0">
            <div className="flex justify-between items-center">
                <p className="font-semibold text-white truncate">{profile.roblox_username}</p>
                <p className="text-xs text-gray-500 flex-shrink-0">2h ago</p>
            </div>
            <p className="text-sm text-gray-400 truncate">Hey, I saw your new template! Looks awesome.</p>
        </div>
    </button>
);


export const MessagesPage: React.FC = () => {
    const [selectedConversation, setSelectedConversation] = useState<string | null>(mockProfiles[0].id);

    return (
        <div className="h-full flex text-white bg-bg-primary">
            {/* Conversation List */}
            <aside className="w-full md:w-96 flex-shrink-0 border-r border-border-color bg-bg-secondary flex flex-col">
                <header className="p-4 border-b border-border-color flex-shrink-0">
                    <h1 className="text-xl font-bold">Messages</h1>
                    <div className="relative mt-2">
                        <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                        <input
                            type="text"
                            placeholder="Search messages..."
                            className="w-full bg-bg-primary border border-border-color rounded-lg pl-10 pr-4 py-2 text-sm"
                        />
                    </div>
                </header>
                <div className="flex-1 overflow-y-auto p-2 space-y-1">
                    {mockProfiles.map(profile => (
                        <ConversationItem
                            key={profile.id}
                            profile={profile}
                            selected={selectedConversation === profile.id}
                            onSelect={() => setSelectedConversation(profile.id)}
                        />
                    ))}
                </div>
            </aside>

            {/* Message View */}
            <main className="flex-1 flex-col hidden md:flex">
                {selectedConversation ? (
                    <div className="h-full flex flex-col">
                        {/* Chat Header */}
                        <header className="p-4 border-b border-border-color flex-shrink-0">
                            <h2 className="font-bold text-white">
                                {mockProfiles.find(p => p.id === selectedConversation)?.roblox_username}
                            </h2>
                        </header>
                        {/* Chat Body (empty for now) */}
                        <div className="flex-1 flex items-center justify-center text-gray-500">
                            <p>Private messaging coming soon.</p>
                        </div>
                        {/* Chat Input */}
                        <div className="p-4 flex-shrink-0">
                            <div className="relative">
                                <input
                                    type="text"
                                    placeholder="Type a message..."
                                    className="w-full bg-bg-secondary border border-border-color rounded-lg pl-4 pr-12 py-3"
                                    disabled
                                />
                                <button className="absolute right-3 top-1/2 -translate-y-1/2 p-2 text-gray-400" disabled>
                                    <PaperAirplaneIcon className="w-6 h-6" />
                                </button>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="h-full flex flex-col items-center justify-center text-center p-8 text-gray-500">
                        <ChatBubbleLeftRightIcon className="w-16 h-16 mb-4" />
                        <h2 className="text-xl font-semibold text-gray-300">Select a Conversation</h2>
                        <p>Choose a conversation from the left to start chatting.</p>
                    </div>
                )}
            </main>
        </div>
    );
};