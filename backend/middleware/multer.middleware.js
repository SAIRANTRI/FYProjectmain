// middleware/multer.middleware.js
import multer from 'multer';
import path from 'path';

// Only accept image files
const fileFilter = (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (ext === '.jpg' || ext === '.jpeg' || ext === '.png' || ext === '.webp') {
        cb(null, true);
    } else {
        cb(new Error('Only image files are allowed'), false);
    }
};

// Memory storage (for uploading directly to Cloudinary)
const storage = multer.memoryStorage();

// Configure multer to store the file in memory and set size limits
const upload = multer({
    storage,
    fileFilter,
    limits: {
        fileSize: 10 * 1024 * 1024 // 5MB max size per file (optional)
    }
});

export default upload;
