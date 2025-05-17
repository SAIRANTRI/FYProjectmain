import React from 'react';
import { X, Eye } from 'react-feather';

const ImageCard = ({ image, onDelete, onView, type }) => {
  return (
    <div className="relative bg-gray-800 rounded-lg overflow-hidden h-32 group transform transition-all duration-300 hover:scale-105 hover:shadow-lg hover:shadow-purple-500/20">
      <img
        src={image.imageUrl}
        alt={type}
        className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
      />
      <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-40 transition-all duration-300 flex items-center justify-center">
        <div className="flex space-x-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
          {onView && (
            <button
              onClick={() => onView(image)}
              className="bg-blue-600 text-white rounded-full p-1.5 hover:bg-blue-700 transition-all duration-300 transform hover:scale-110"
            >
              <Eye size={16} />
            </button>
          )}
          <button
            onClick={() => onDelete(type, image._id)}
            className="bg-red-600 text-white rounded-full p-1.5 hover:bg-red-700 transition-all duration-300 transform hover:scale-110"
          >
            <X size={16} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default ImageCard;