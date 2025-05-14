import express from 'express';
import axios from 'axios';
import Photo from '../models/photo.model.js';
import { protect } from "../Middleware/auth.middleware.js";

const router = express.Router();

const MODEL_API_URL = process.env.MODEL_API_URL || 'http://localhost:8000';

router.post('/process-images', protect, async (req, res) => {
  try {
    const userId = req.user._id;

    const referenceImages = await Photo.find({ userId, imageType: 'reference' });
    const poolImages = await Photo.find({ userId, imageType: 'pool' });

    if (referenceImages.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'No reference images found. Please upload at least one reference image.'
      });
    }

    if (poolImages.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'No pool images found. Please upload at least one pool image.'
      });
    }

    const response = await axios.post(`${MODEL_API_URL}/process-images/`, {
      userId: userId.toString()
    });

    res.status(200).json({
      success: true,
      taskId: response.data.taskId,
      status: response.data.status,
      message: response.data.message
    });
  } catch (error) {
    console.error('Error processing images:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to process images',
      error: error.response?.data?.detail || error.message
    });
  }
});

router.get('/processing-result/:taskId', protect, async (req, res) => {
  try {
    const { taskId } = req.params;

    const response = await axios.get(`${MODEL_API_URL}/processing-result/${taskId}`);

    res.status(200).json({
      success: true,
      result: response.data
    });
  } catch (error) {
    console.error('Error getting processing result:', error);
    res.status(error.response?.status || 500).json({
      success: false,
      message: 'Failed to get processing result',
      error: error.response?.data?.detail || error.message
    });
  }
});

router.get('/download-results/:taskId', protect, async (req, res) => {
  try {
    const { taskId } = req.params;
    
    res.redirect(`${MODEL_API_URL}/download-results/${taskId}`);
  } catch (error) {
    console.error('Error downloading results:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to download results',
      error: error.message
    });
  }
});

export default router;