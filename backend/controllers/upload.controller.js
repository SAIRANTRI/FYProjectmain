import { streamUpload } from '../lib/streamUpload.js'; // Importing the utility function
import mongoose from 'mongoose';
import Photo from '../models/photo.model.js';
import cloudinary from '../lib/cloudinary.js'; // Importing Cloudinary configuration
import dotenv from 'dotenv';  
dotenv.config();

// Function to upload reference image
export const uploadReferenceImage = async (req, res) => {
  try {
    const file = req.file;

    if (!file) {
      return res.status(400).json({ message: "No file uploaded" });
    }

    const publicId = `reference_${Date.now()}`;
    const folder = "albumify/references";

    // Use the streamUpload utility to upload the image to Cloudinary
    const uploadedImage = await streamUpload(file.buffer, folder, publicId);

    // Save the uploaded image details in the database
    const newReference = new Photo({
      userId: req.user.id,
      imageUrl: uploadedImage.secure_url,
      publicId: uploadedImage.public_id,
      imageType: "reference",
      uploadedAt: new Date(),
    });

    await newReference.save();

    res.status(200).json({
      message: "Reference image uploaded successfully",
      imageUrl: uploadedImage.secure_url,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Error uploading reference image", error: error.message });
  }
};

// Function to upload multiple pool images
export const uploadPoolImages = async (req, res) => {
  try {
    const files = req.files;

    if (!files || files.length === 0) {
      return res.status(400).json({ message: "No files uploaded" });
    }

    const referenceImage = await Photo.findOne({ userId: req.user.id, imageType: "reference" });

    if (!referenceImage) {
      return res.status(400).json({ message: "Reference image is required to upload pool images" });
    }

    const uploadedImages = [];

    // Loop through each file and upload it to Cloudinary
    for (let file of files) {
      const publicId = `pool_${Date.now()}`;
      const folder = "albumify/pool_images";

      // Use the streamUpload utility to upload the image to Cloudinary
      const uploadedImage = await streamUpload(file.buffer, folder, publicId);

      // Save uploaded image details in the database
      const newPoolImage = new Photo({
        userId: req.user.id,
        publicId: uploadedImage.public_id,
        imageUrl: uploadedImage.secure_url,
        imageType: "pool",
        uploadedAt: new Date(),
        referenceId: referenceImage._id,
      });

      await newPoolImage.save();
      uploadedImages.push(uploadedImage.secure_url);
    }

    res.status(200).json({
      message: "Pool images uploaded successfully",
      images: uploadedImages,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Error uploading pool images", error: error.message });
  }
};

// Get all uploaded images
export const getUploadedImages = async (req, res) => {
  try {
    const userId = req.user.id;

    const referenceImages = await Photo.find({ userId, imageType: "reference" });
    const poolImages = await Photo.find({ userId, imageType: "pool" });

    res.status(200).json({ referenceImages, poolImages });
  } catch (error) {
    console.error("Error fetching uploaded images:", error);
    res.status(500).json({ message: "Server error" });
  }
};

// Controller to delete an image

export const deleteImage = async (req, res) => {
  const { imageId } = req.params;

  try {
    if (!mongoose.Types.ObjectId.isValid(imageId)) {
      return res.status(400).json({ message: "Invalid image ID" });
    }

    // Retrieve the image from MongoDB by imageId
    const image = await Photo.findById(imageId);
    if (!image) {
      return res.status(404).json({ message: "Image not found" });
    }

    // Now delete the image from Cloudinary using the publicId
    await cloudinary.uploader.destroy(image.publicId);

    // Delete the image from MongoDB
    await Photo.findByIdAndDelete(imageId);

    res.status(200).json({ message: "Image deleted from MongoDB and Cloudinary successfully" });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Error deleting image" });
  }
};

