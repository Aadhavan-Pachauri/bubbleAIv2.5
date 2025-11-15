
import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { UsersIcon, MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import { mockProfiles } from '../../data/users'; 
import { Profile } from '../../types';

const CreatorCard: React.FC<{ profile: Profile }> = ({ profile }) => (
    <motion.div
        // FIX: framer-motion props wrapped in a spread object to bypass type errors.
        {...{
          layout: true,
          initial: { opacity: 0, scale: 0.9 },
          animate: { opacity: 1, scale: 1 },
        }}
        className="bg-bg-secondary rounded-xl p-4 border border-bg-tertiary flex flex-col items-center text-center transition-colors hover:border-primary-start"
    >
        <img src={profile.avatar_url} alt={profile.roblox_username} className="w-24 h-24 rounded-full mb-4 object-cover" />
        <h3 className="font-bold text-white text-lg">{profile.roblox_username}</h3>
        <p className="text-sm text-primary-start capitalize font-semibold">{profile.membership}</p>
        <p className="text-xs text-gray-400 mt-2 flex-grow line-clamp-2">
            {profile.bio || `An enthusiastic creator exploring Bubble AI.`}
        </p>
        <button className="mt-4 w-full px-4 py-2 text-sm font-semibold bg-white/5 text-gray-200 rounded-lg hover:bg-white/10 hover:text-white transition-colors">
            View Profile
        </button>
    </motion.div>
);

export const DiscoverPage: React.FC = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const filteredProfiles = useMemo(() => {
        if (!searchQuery) return mockProfiles;
        return mockProfiles.filter(p => p.roblox_username.toLowerCase().includes(searchQuery.toLowerCase()));
    }, [searchQuery]);

    return (
        <div className="p-4 md:p-8 h-full overflow-y-auto">
            <header className="mb-8">
                <h1 className="text-3xl md:text-4xl font-bold text-white">Discover Creators</h1>
                <p className="text-gray-400 mt-1">Find and connect with other talented developers.</p>
            </header>

            <div className="sticky top-0 bg-bg-primary/80 backdrop-blur-sm z-10 py-4 mb-6">
                <div className="relative max-w-md mx-auto">
                    <MagnifyingGlassIcon className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500 pointer-events-none" />
                    <input
                        type="text"
                        placeholder="Search for creators..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="bg-bg-secondary/50 border border-bg-tertiary rounded-full pl-12 pr-4 py-3 text-sm w-full focus:outline-none focus:ring-2 focus:ring-primary-start"
                    />
                </div>
            </div>

            <motion.div
                // FIX: framer-motion props wrapped in a spread object to bypass type errors.
                {...{ layout: true }}
                className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
                {filteredProfiles.map(profile => (
                    <CreatorCard key={profile.id} profile={profile} />
                ))}
            </motion.div>
        </div>
    );
};
