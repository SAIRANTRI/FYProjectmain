// lib/streamupload.js
import cloudinary from './cloudinary.js';  // Import cloudinary configuration from lib
import streamifier from 'streamifier';

// Utility to stream buffer to Cloudinary
export const streamUpload = (buffer, folder, publicId) => {
  return new Promise((resolve, reject) => {
    const stream = cloudinary.uploader.upload_stream(
      {
        folder,
        public_id: publicId,
      },
      (error, result) => {
        if (result) resolve(result);
        else reject(error);
      }
    );
    streamifier.createReadStream(buffer).pipe(stream);
  });
};
