import express from "express";
import { uploadReferenceImage, uploadPoolImages, getUploadedImages, deleteImage } from "../controllers/upload.controller.js";
import { protect } from "../middleware/auth.middleware.js";
import upload from "../middleware/multer.middleware.js";  // Import the upload middleware

const router = express.Router();

// Route to upload reference image (single file upload)
router.post("/reference", protect, upload.single("file"), uploadReferenceImage);

// Route to upload pool images (multiple files upload)
router.post("/pool", protect, upload.array("files", 10), uploadPoolImages);

// Route to fetch uploaded images
router.get("/images", protect, getUploadedImages);
router.delete("/delete/:imageId", protect, deleteImage);

export default router;
