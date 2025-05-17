import React from 'react';

const Spinner = ({ size = 'medium', color = 'purple' }) => {
  const sizeClasses = {
    small: 'h-5 w-5 border-2',
    medium: 'h-8 w-8 border-2',
    large: 'h-12 w-12 border-3',
  };

  const colorClasses = {
    purple: 'border-t-purple-500 border-r-purple-500/30 border-b-purple-500/10 border-l-purple-500/50',
    white: 'border-t-white border-r-white/30 border-b-white/10 border-l-white/50',
    pink: 'border-t-pink-500 border-r-pink-500/30 border-b-pink-500/10 border-l-pink-500/50',
  };

  return (
    <div className="flex items-center justify-center">
      <div className={`${sizeClasses[size]} ${colorClasses[color]} rounded-full animate-spin`}></div>
    </div>
  );
};

export default Spinner;